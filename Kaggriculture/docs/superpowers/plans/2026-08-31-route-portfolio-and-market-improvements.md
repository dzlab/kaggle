# Kaggriculture Route Portfolio and Market Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ( - [ ] ) syntax for tracking.

**Goal:** Replace the current one-size-fits-most policy with a small, observable-state route portfolio that controls production, preserves needs solvency, and sells in a market-impact-aware order.

**Architecture:** A StrategySpec describes the economic route, asset mix, production caps, and market preferences. Policy selects one spec once after the opening observation and keeps it stable for the episode; the existing planner and legality checks execute the selected plan. A separate market controller ranks legal orders using current price, post-sale price impact, demand signals, inventory urgency, and terminal horizon.

**Tech Stack:** Python 3.11+, uv, pytest, standard-library dataclasses/mappings, existing observation/planner/routing modules.

---

### Task 1: Introduce strategy specifications without changing the current default

**Files:**
- Create: Kaggriculture/kagriculture_agent/strategy.py
- Modify: Kaggriculture/kagriculture_agent/policy.py
- Modify: Kaggriculture/kagriculture_agent/memory.py
- Test: Kaggriculture/tests/test_policy.py

- [ ] Step 1: Write failing tests

~~~python
def test_policy_defaults_to_current_strategy():
    from kagriculture_agent.policy import Policy
    assert Policy().strategy_name == "current"

def test_unknown_strategy_is_rejected():
    from kagriculture_agent.policy import Policy
    with pytest.raises(ValueError, match="unsupported strategy"):
        Policy(strategy="not-a-route")
~~~

- [ ] Step 2: Run the focused tests to verify they fail

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py -k strategy -q

Expected: FAIL because the constructor and strategy registry do not exist.

- [ ] Step 3: Add the strategy dataclass and memory field

~~~python
# Kaggriculture/kagriculture_agent/strategy.py
from dataclasses import dataclass

@dataclass(frozen=True)
class StrategySpec:
    name: str
    crops: tuple[str, ...]
    animals: tuple[str, ...]
    max_crop_units: int
    max_animal_units: int
    reserve_wheat: int
    avoid_price_floor_sales: bool = True
    terminal_liquidation_hour: int = 22

STRATEGIES = {
    "current": StrategySpec("current", ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"), ("GOOSE", "COW", "SHEEP"), 10000, 10000, 8),
    "melon": StrategySpec("melon", ("WHEAT", "MELON"), ("COW", "SHEEP"), 80, 9, 12),
    "premium": StrategySpec("premium", ("WHEAT", "MELON"), ("COW", "SHEEP"), 48, 12, 16),
    "mixed": StrategySpec("mixed", ("WHEAT", "CARROT", "MELON"), ("COW", "SHEEP"), 64, 9, 12),
}

def get_strategy(name: str) -> StrategySpec:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported strategy: {name}") from exc
~~~

Add selected_strategy: str | None = None to PolicyMemory and clear it in reset(). Make Policy(strategy="current") validate the name while preserving current behavior for current.

- [ ] Step 4: Run policy and memory tests

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py tests/test_observation.py -q

Expected: PASS.

- [ ] Step 5: Commit the strategy boundary

~~~bash
git add Kaggriculture/kagriculture_agent/strategy.py Kaggriculture/kagriculture_agent/policy.py Kaggriculture/kagriculture_agent/memory.py Kaggriculture/tests/test_policy.py
git commit -m "feat: add stable route strategy boundary"
~~~

### Task 2: Select a route from early observable state

**Files:**
- Modify: Kaggriculture/kagriculture_agent/strategy.py
- Modify: Kaggriculture/kagriculture_agent/policy.py
- Test: Kaggriculture/tests/test_policy.py

- [ ] Step 1: Write tests for public opening signals

~~~python
def test_select_strategy_prefers_melon_without_shop_signal():
    from kagriculture_agent.strategy import select_strategy
    state = {"cash": 3000, "town": {"unlocked_shops": []}, "market": {"prices": {}, "inventory": {}}}
    assert select_strategy(state).name == "melon"

def test_select_strategy_uses_pet_cafe_signal():
    from kagriculture_agent.strategy import select_strategy
    state = {"cash": 3000, "town": {"unlocked_shops": ["PET_CAFE"]}, "market": {"prices": {}, "inventory": {}}}
    assert select_strategy(state).name == "mixed"
~~~

- [ ] Step 2: Run the tests to verify they fail

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py -k select_strategy -q

Expected: FAIL because select_strategy does not exist.

- [ ] Step 3: Implement a sparse selector with a conservative fallback

~~~python
def select_strategy(state):
    town = state.get("town", {}) if isinstance(state, dict) else {}
    shops = {str(shop).upper() for shop in town.get("unlocked_shops", ())}
    if "PET_CAFE" in shops:
        return STRATEGIES["mixed"]
    if "YARN_STORE" in shops or "ICE_CREAM_SHOP" in shops:
        return STRATEGIES["premium"]
    return STRATEGIES["melon"]
~~~

Call this once at episode start only when Policy(strategy="auto"). Store the result in memory.selected_strategy. Use only the current observation, never future replay state. Keep the default constructor value current until a candidate passes the evaluation gate.

- [ ] Step 4: Prove that an auto policy does not switch routes mid-episode

~~~python
def test_auto_policy_keeps_opening_route(monkeypatch):
    from kagriculture_agent.policy import Policy
    policy = Policy(strategy="auto")
    policy.act({"day": 0, "hour": 0, "town": {"unlocked_shops": []}, "market": {}})
    chosen = policy.memory.selected_strategy
    policy.act({"day": 0, "hour": 1, "town": {"unlocked_shops": ["YARN_STORE"]}, "market": {}})
    assert policy.memory.selected_strategy == chosen
~~~

- [ ] Step 5: Run tests and commit

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py -q

Expected: PASS.

~~~bash
git add Kaggriculture/kagriculture_agent/strategy.py Kaggriculture/kagriculture_agent/policy.py Kaggriculture/tests/test_policy.py
git commit -m "feat: select route from opening public state"
~~~

### Task 3: Make production and asset purchases follow the selected route

**Files:**
- Modify: Kaggriculture/kagriculture_agent/planner.py
- Modify: Kaggriculture/kagriculture_agent/policy.py
- Modify: Kaggriculture/kagriculture_agent/strategy.py
- Test: Kaggriculture/tests/test_policy.py
- Test: Kaggriculture/tests/test_economics.py

- [ ] Step 1: Write route-cap tests

~~~python
def test_melon_strategy_excludes_crash_prone_ongoing_crops():
    from kagriculture_agent.strategy import get_strategy
    spec = get_strategy("melon")
    assert "TOMATO" not in spec.crops
    assert "STRAWBERRY" not in spec.crops

def test_premium_strategy_reserves_more_wheat():
    from kagriculture_agent.strategy import get_strategy
    assert get_strategy("premium").reserve_wheat > get_strategy("mixed").reserve_wheat
~~~

- [ ] Step 2: Run the tests to verify route constraints are not wired

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py tests/test_economics.py -k "melon_strategy or premium_strategy" -q

Expected: FAIL until planner decisions consume StrategySpec.

- [ ] Step 3: Thread StrategySpec through macro planning

Change build_autonomous_macro_plan(state, memory, strategy=None) and build_daily_plan(state, memory, strategy=None). When present, filter crop tasks to strategy.crops, animal purchases to strategy.animals, cap planned crop and animal counts, and pass strategy.reserve_wheat into the existing feed-reserve calculation. Do not bypass affordability, legality, deadlines, or routing.

~~~python
allowed_crops = set(strategy.crops) if strategy is not None else set(CROPS)
crop_candidates = [crop for crop in crop_candidates if crop in allowed_crops]
allowed_animals = set(strategy.animals) if strategy is not None else set(ANIMALS)
animal_candidates = [animal for animal in animal_candidates if animal in allowed_animals]
~~~

- [ ] Step 4: Run planner, economics, and policy tests

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py tests/test_economics.py tests/test_routing.py -q

Expected: PASS with current-route legality unchanged.

- [ ] Step 5: Commit route-aware planning

~~~bash
git add Kaggriculture/kagriculture_agent/planner.py Kaggriculture/kagriculture_agent/policy.py Kaggriculture/kagriculture_agent/strategy.py Kaggriculture/tests/test_policy.py Kaggriculture/tests/test_economics.py
git commit -m "feat: constrain production by route economics"
~~~

### Task 4: Add market-impact-aware batching and order priority

**Files:**
- Modify: Kaggriculture/kagriculture_agent/policy.py
- Modify: Kaggriculture/kagriculture_agent/strategy.py
- Test: Kaggriculture/tests/test_policy.py
- Test: Kaggriculture/tests/test_economics.py

- [ ] Step 1: Write tests for post-sale quotes and safe order priority

~~~python
def test_market_order_score_penalizes_large_price_impact():
    from kagriculture_agent.strategy import market_order_score
    state = {"market": {"prices": {"TOMATO": 10, "MELON": 250}, "inventory": {"TOMATO": 200, "MELON": 20}}}
    assert market_order_score("MELON", 4, state, urgency=0) > market_order_score("TOMATO", 40, state, urgency=0)

def test_sell_orders_precede_buys_when_cash_is_needed():
    from kagriculture_agent.policy import order_market_intents
    intents = [{"kind": "BUY_SEED", "item": "MELON", "quantity": 1}, {"kind": "SELL", "item": "MELON", "quantity": 2}]
    assert order_market_intents(intents, cash_needed=True)[0]["kind"] == "SELL"
~~~

- [ ] Step 2: Implement sequential quote impact and bounded batches

~~~python
def market_order_score(item, quantity, state, urgency):
    inventory = float(state["market"]["inventory"].get(item, 0))
    params = state["market"].get("params")
    quotes = [market_price(item, inventory - offset - 1, params)
              for offset in range(max(1, int(quantity)))]
    average = sum(quotes) / len(quotes)
    impact = max(0.0, quotes[0] - average) * max(1, int(quantity))
    return average * max(1, int(quantity)) - impact + float(urgency)
~~~

Add order_market_intents() that sorts sells before buys when cash is constrained, caps one sell batch using the selected strategy, and leaves BUY_PRODUCT restricted to WHEAT and FERTILIZER.

- [ ] Step 3: Add terminal liquidation without violating needs

At day >= season_days - 1 and hour >= terminal_liquidation_hour, sell saleable shed goods after carried goods are dropped. Never sell wheat below reserve_wheat while a live animal still requires feed. Preserve terminal cleanup and _remove_pickup_sale_conflicts().

- [ ] Step 4: Run market and policy tests

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py tests/test_economics.py -q

Expected: PASS, including existing price-floor and terminal-boundary tests.

- [ ] Step 5: Commit the market controller

~~~bash
git add Kaggriculture/kagriculture_agent/policy.py Kaggriculture/kagriculture_agent/strategy.py Kaggriculture/tests/test_policy.py Kaggriculture/tests/test_economics.py
git commit -m "feat: add market impact aware order control"
~~~

### Task 5: Add conservative opponent signals as an optional branch

**Files:**
- Modify: Kaggriculture/kagriculture_agent/strategy.py
- Modify: Kaggriculture/kagriculture_agent/policy.py
- Modify: Kaggriculture/kagriculture_agent/memory.py
- Test: Kaggriculture/tests/test_policy.py

- [ ] Step 1: Write tests requiring strong evidence before pre-sale

~~~python
def test_opponent_signal_requires_three_consistent_observations():
    from kagriculture_agent.strategy import OpponentMarketSignal
    signal = OpponentMarketSignal()
    assert signal.observe({"market": {"inventory": {"MELON": 20}}}) is None
    assert signal.observe({"market": {"inventory": {"MELON": 16}}}) is None
    assert signal.observe({"market": {"inventory": {"MELON": 12}}}) == "MELON"

def test_price_floor_disables_front_running():
    from kagriculture_agent.strategy import should_front_run
    assert not should_front_run(item="TOMATO", current_price=1, evidence=3, town_refill=False)
~~~

- [ ] Step 2: Run tests to verify they fail

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py -k "opponent_signal or front_running" -q

Expected: FAIL because the signal tracker is not implemented.

- [ ] Step 3: Implement signal tracking with no guessed rival identity

Track public market inventory deltas and town-demand refreshes only. Require three monotonic observations, a price above the floor, no intervening town refill, and an owned sellable batch before adding a pre-sale intent. Clear the history on episode reset. Do not infer which opponent sold and never front-run when a prerequisite is missing.

- [ ] Step 4: Run the focused tests and the full suite

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_policy.py -q

Expected: PASS.

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest -q

Expected: PASS with zero framework-related regressions.

- [ ] Step 5: Commit the optional branch

~~~bash
git add Kaggriculture/kagriculture_agent/strategy.py Kaggriculture/kagriculture_agent/policy.py Kaggriculture/kagriculture_agent/memory.py Kaggriculture/tests/test_policy.py
git commit -m "feat: add conservative opponent market signals"
~~~

### Task 6: Evaluate, compare, and promote only a surviving route

**Files:**
- Modify: Kaggriculture/kagriculture_agent/candidates.py
- Modify: Kaggriculture/README.md
- Test: Kaggriculture/tests/test_packaging.py
- Test: Kaggriculture/tests/test_agent_smoke.py

- [ ] Step 1: Register explicit policy factories

Map current, melon, premium, and mixed to Policy(strategy=...). Keep production main.py as Policy() until a candidate passes the evaluation plan’s promotion gate.

- [ ] Step 2: Run package and smoke tests

~~~bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_packaging.py tests/test_agent_smoke.py -q
~~~

Expected: PASS and the archive still contains root main.py plus runtime package files only.

- [ ] Step 3: Run the development matrix and discard failed candidates

Run the evaluation command from 2026-08-31-evaluation-and-discard-gates.md with all candidates. Do not promote on mean bank alone. Require zero framework errors, zero missed basic needs, valid both-seat pairs, a non-negative fifth-percentile bank differential, and improvement over current on holdout seeds.

- [ ] Step 4: Re-run holdout from a clean state

Run the same command with only surviving candidates and holdout seeds. Record the selected candidate and every discard reason. If code changes after seeing holdout results, create a new candidate name and repeat the development/holdout split.

- [ ] Step 5: Update main.py only after promotion evidence exists

If and only if a candidate passes, replace the current constructor with Policy(strategy="PROMOTED_CANDIDATE"), using the exact report-selected name. Otherwise leave the current default unchanged.

- [ ] Step 6: Run final verification and commit only a promoted candidate

~~~bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv lock --check
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest -q
tar --exclude='__pycache__' --exclude='*.pyc' -czf /tmp/kaggriculture-promoted-agent.tar.gz -C . main.py kagriculture_agent
tar -tzf /tmp/kaggriculture-promoted-agent.tar.gz
~~~

Expected: lockfile check succeeds, all tests pass, and the archive contains main.py at its root with no tests, reports, or documentation.

~~~bash
git add Kaggriculture/main.py Kaggriculture/kagriculture_agent Kaggriculture/tests Kaggriculture/README.md
git commit -m "feat: promote evaluated Kaggriculture route"
~~~
