# Kaggriculture Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, locally validate, and submit a robust model-based Kaggriculture agent that maximizes the probability of beating another agent over 720 turns under randomized shops and a shared dynamic market.

**Architecture:** Use a deterministic observation-to-action policy with a small per-episode memory, an explicit daily macro planner, and a shortest-path task scheduler for the farmer and temporary farm hands. The planner will score crop, animal, land, hiring, fertilizer, and selling decisions with the exact published game mechanics, while the policy layer converts the selected tasks into one legal action per unit and up to ten ordered market orders per turn.

**Tech Stack:** Python 3, `kaggle-environments>=1.32.7`, pytest, standard library (`dataclasses`, `heapq`, `collections`, `math`, `json`), optional pandas only for offline replay analysis.

---

## Research findings and implications

- The competition is a two-player simulation, not a train/test prediction problem. Each game lasts 30 days × 24 turns = 720 turns; the winner is the player with the most money in the bank at the end. Unsold shed/inventory goods do not count.
- The agent receives public farms, the shared market, and the unlocked shop list, plus its own shed, seeds, and worker inventories. It returns `{"farmer": [...], "hands": [...], "market": [...]}`.
- The initial board is a 10×10 grid with only NW unlocked. `BUY_LAND` unlocks quadrants in fixed `NE`, `SW`, `SE` order at `$1,000`, `$2,000`, and `$4,000`. Building a coop or pasture costs a turn and does not deduct cash in the published engine.
- Plants must be watered before two consecutive missed refreshes; the planting day counts as the first miss. Animals survive their first newly placed day without feed, then require one wheat per day. `CARE` creates a bonus only when that day is also fed.
- The shared market starts at inventory 10,000 per product. Sale prices use product-specific scarcity/glut curves, sales are processed one unit at a time in lockstep across players, premium products can hit the `$1` floor under a glut, and only wheat/fertilizer can be bought back.
- Shops unlock every three days, with replacement and up to eight instances. The shop draw is the main strategic uncertainty visible to the agent over time; its demand should update the crop/animal mix rather than be treated as a fixed prior.
- The most recent published balance change is in engine version `1.32.7`: scarcity-side curves make carrots, tomatoes, and eggs substantially more valuable in some high-demand games. Local tests must pin or assert this version so a stale environment does not produce misleading results.
- Kaggle’s current public Code page shows that strong public submissions use economics-driven rules, route portfolios, and adaptive shop handling. Their displayed scores are volatile and should be treated as directional evidence only, not acceptance criteria.
- The requested workspace is empty at `/Users/bachir/co/github/dzlab/kaggle/Kaggriculture`; implementation must bootstrap the project there. The parent directory is not currently a Git repository, so implementation should initialize version control before using commit-based checkpoints.

## File map

Create the following focused files under `/Users/bachir/co/github/dzlab/kaggle/Kaggriculture`:

- `main.py` — submission-safe entry point; exposes only `agent(obs)` and imports the package.
- `kaggriculture_agent/constants.py` — crop, animal, shop, land, market, timing, and action constants copied from the published rules with version metadata.
- `kaggriculture_agent/types.py` — typed dataclasses/value objects for positions, tasks, worker assignments, economic estimates, and episode memory.
- `kaggriculture_agent/observation.py` — defensive parsing of public/private observations, tile classification, coordinates, shed capacity, and episode-boundary detection.
- `kaggriculture_agent/economics.py` — exact market price curves, production forecasts, feed costs, care/fertilizer effects, and opportunity-cost scoring.
- `kaggriculture_agent/routing.py` — grid distances and deterministic movement actions; locked tiles remain passable but are never selected for tile actions.
- `kaggriculture_agent/planner.py` — daily macro plan and intraday task assignment for crops, animals, land, hires, and liquidation.
- `kaggriculture_agent/policy.py` — worker action selection, market-order construction, shed/inventory handling, and safe fallback actions.
- `kaggriculture_agent/memory.py` — resettable per-episode/per-day state for assignments, planned sell batches, and diagnostics.
- `tests/test_economics.py` — exact prices and production/revenue calculations.
- `tests/test_observation.py` — observation parsing and edge cases.
- `tests/test_routing.py` — movement and assignment routing.
- `tests/test_policy.py` — legal action decisions and market-order safeguards.
- `tests/test_agent_smoke.py` — local environment integration against built-in opponents.
- `scripts/run_local.py` — one-game runner with replay output and final bank/status summary.
- `scripts/evaluate.py` — seeded batch evaluation against `pass`, `random`, `starter`, and internal policy variants.
- `pyproject.toml` — package metadata, dependency pin, pytest configuration, and lint/test commands.
- `README.md` — setup, local evaluation, submission packaging, and operational notes.
- `replays/.gitkeep` — output directory marker; generated replay files remain local.
- `logs/.gitkeep` — output directory marker; generated diagnostics remain local.

## Strategy design

### 1. Exact state reconstruction

At every call, derive all decisions from `obs["day"]`, `obs["hour"]`, `obs["farms"]`, `obs["market"]`, `obs["town"]`, and `obs["private"]`. Do not depend on `obs["step"]` because the competition discussion reports a seat-1 step inconsistency. Reset memory when `(day, hour)` is `(0, 0)` or time moves backward.

Track:

- cash, unlocked quadrants, available empty/weed/plant/structure tiles, worker positions, and current-day worker count;
- seed counts, shed product counts, worker inventories, and remaining shed capacity;
- plant crop, age, watering status, fertilization expiry, current yield units, and decay risk;
- animal species, feed/care status, product buffer, pending care bonus, and fertilizer availability;
- market prices/inventory, shop multiplicities, expected town consumption, opponent farm production, and the current market regime.

All action decisions must be legal for the current tile and inventory. A malformed or unavailable task falls back to `PASS` or one legal movement step; it must never emit a guessed quantity or a market order after the ten-order limit.

### 2. Production and macro economics

Implement exact functions for:

- `market_price(item, inventory)` using linear, square, square-root, log, log10, and hinge shapes;
- one-time crop yield by age, watering history, fertilizer window, and harvest timing;
- ongoing crop scheduled yields and decay;
- animal production, feed consumption, care banking, held-yield caps, and fertilizer collection;
- expected net cash over the remaining days, including seed/animal costs, wheat feed, worker costs, land cost, expected sell price, and travel/action capacity.

Use the following initial priors only until observations provide evidence: prioritize fast one-time crops for opening cash; use melons selectively because they have high base value but glut to `$1`; treat carrots/tomatoes/eggs as conditional scarcity plays after shop demand is observed; add animals only when projected fed-and-cared revenue exceeds the best crop/worker alternative; reserve cash for the next profitable land or worker expansion.

The macro planner should compare a small set of portfolio candidates each day rather than optimize every tile continuously:

1. fast-cash crop rotation (wheat/carrot),
2. melon batch with controlled harvest/sale,
3. demand-matched tomato/strawberry batch,
4. egg/cow/sheep infrastructure with wheat feed reserve,
5. mixed portfolio with one risk-limited premium batch.

Score each candidate against at least 16 deterministic shop/market scenarios sampled from the current observed shop state and plausible future unlocks. Select by expected win proxy: expected bank differential divided by estimated variance, with a conservative penalty for inventory overflow, missed watering/feed, and sales at the price floor.

### 3. Worker scheduling and routing

At day start, create a task list with deadlines and value per action:

- water plants before their daily window closes;
- feed and care animals before end-of-day;
- harvest crops at or after the earliest positive-yield time, with one-time crops harvested before decay;
- collect animal fertilizer once per surviving animal/day when travel and shed capacity justify it;
- build structures, place animals, dig weeds, or buy land only when the macro plan has approved the investment;
- move harvested goods to the shed and sell only goods currently in the shed;
- use spare turns for planting, shed logistics, or market timing.

Assign tasks greedily by urgency first, then `expected_cash_delta / (movement_turns + action_turns)`, with a stable tie-break by worker index and coordinate. Use Manhattan routing and avoid claiming the same exclusive tile/action in one turn. Workers may share a tile, so overlapping routes are valid when they do not cause duplicate no-op actions.

Use farm hands as one-day capacity, not as persistent assets. Hire while the marginal value of an extra 24-turn worker exceeds the Fibonacci hire cost plus any shed overflow risk. The main farmer should remain responsible for shed-adjacent pickup/drop and critical macro transitions; hands should handle repetitive field or animal tasks.

### 4. Market execution

Every turn, build an ordered market queue with at most ten entries:

- reserve wheat for all animals through the next feed deadline before selling wheat;
- buy seeds/animals/land/hands only when cash reserves and the macro plan permit;
- sell shed goods in bounded batches, recomputing from the observed price and inventory each turn;
- for high-glut-risk goods, sell smaller batches and prefer demand windows immediately after town consumption;
- for scarcity-sensitive carrot/tomato/egg, retain inventory for a later observed demand spike when the price curve offers more than the planned baseline;
- never buy products merely to manufacture a round trip; buy wheat/fertilizer only when operationally required or when the expected avoided loss exceeds the price.

The market planner must model per-unit lockstep impact: a `SELL item n` order can lower the quote on later units and interact with the opponent’s same-turn orders. It should therefore return a batch quantity, not just a binary sell decision. At season end, liquidate all shed goods in the final available turns even if the price is poor because banked cash is the objective.

### 5. Submission and adaptation loop

Keep the submitted agent deterministic for a fixed observation sequence. Use separate local variants for A/B testing: conservative mixed portfolio, melon-heavy, demand-reactive, and animal-heavy. Submit only after seeded local tests pass; use the daily five-submission allowance to test variants while keeping only the latest two active in mind. Preserve replay/log metadata for every variant so leaderboard results can be attributed to a policy change.

## Implementation tasks

### Task 1: Bootstrap the runnable project

**Files:**
- Create: `pyproject.toml`
- Create: `main.py`
- Create: `kaggriculture_agent/__init__.py`
- Create: `README.md`
- Create: `replays/.gitkeep`
- Create: `logs/.gitkeep`

- [ ] **Step 1: Add the dependency and test configuration**

Use this minimum project configuration:

```toml
[project]
name = "kaggriculture-agent"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["kaggle-environments==1.32.7", "pytest>=8,<10"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-q"
```

- [ ] **Step 2: Add the submission entry point**

`main.py` must expose exactly this import-compatible function:

```python
from kaggriculture_agent.policy import Policy

_policy = Policy()


def agent(obs):
    return _policy.act(obs)
```

- [ ] **Step 3: Add setup and submission instructions**

Document `python -m venv .venv`, dependency installation, local smoke execution, replay output, and the two supported Kaggle submission forms: a single `main.py` during early smoke tests and a tarball with `main.py` at its root for the multi-file agent.

- [ ] **Step 4: Run the import check**

Run `python3 -c 'from main import agent; print(callable(agent))'`.
Expected: `True` after dependencies are installed; before installation, the failure should identify the missing dependency rather than a syntax error.

### Task 2: Encode the published rules and typed state

**Files:**
- Create: `kaggriculture_agent/constants.py`
- Create: `kaggriculture_agent/types.py`
- Create: `kaggriculture_agent/observation.py`
- Test: `tests/test_observation.py`

- [ ] **Step 1: Define immutable rule tables**

Add exact crop fields (`seed`, `first_yield_day`, `max_yield_day`, `interval`, `max_yield`, `ongoing`), animal fields (`cost`, `structure`, `first_yield_day`, `interval`, `max_held`, `product`), shop demand lists, land order/prices, `turns_per_day=24`, `season_days=30`, `shed_capacity=100`, and `max_market_orders=10`. Include `ENGINE_VERSION = "1.32.7"`.

- [ ] **Step 2: Define state value objects**

Create dataclasses with explicit fields and no hidden mutable defaults: `Position(x, y)`, `TileRef(position, tile)`, `Task(kind, target, priority, deadline, value)`, `WorkerAssignment(worker_index, task, route)`, `EconomicEstimate(cash_delta, turns, risk)`, and `EpisodeMemory(last_day, last_hour, assignments, sell_batches, diagnostics)`.

- [ ] **Step 3: Implement defensive observation parsing**

Add `parse_observation(obs)`, `iter_tiles(farm)`, `shed_total(private)`, `shed_access_tiles(board_size)`, `is_shed_adjacent(position, board_size)`, and `is_episode_start(obs, memory)`. Treat absent optional fields as empty collections, preserve `tiles[y][x]` orientation, and derive the active player from `obs["player"]`.

- [ ] **Step 4: Write parsing tests first**

Cover: the NW-only initial board, locked tiles being passable but not actionable, the four shed-access coordinates for a 10×10 board, absent `hands`, a full shed, and a reset when time changes from a later hour to `(0, 0)`.

- [ ] **Step 5: Run tests**

Run `pytest tests/test_observation.py -q`.
Expected: all parsing tests pass without importing the Kaggle engine.

### Task 3: Implement exact economics

**Files:**
- Create: `kaggriculture_agent/economics.py`
- Test: `tests/test_economics.py`

- [ ] **Step 1: Write price-curve tests**

Assert the published reference values at `I0`, `I0-T`, `I0+T`, and `I0+2T` for wheat, carrot, tomato, strawberry, melon, egg, milk, wool, and fertilizer. Assert the `$1` floor and the tomato/carrot/egg hinge behavior.

- [ ] **Step 2: Implement the market functions**

Implement `shape_value`, `market_price`, `sell_batch_value`, `project_inventory_after_town`, and `market_regime`. Round to the nearest dollar only at the same boundary as the engine and keep batch simulation per unit so later units receive later prices.

- [ ] **Step 3: Write production tests**

Cover the planting-day watering miss, one-time crop bonus windows, fertilizer doubling for three days, ongoing crop scheduled production, animal first-yield timing, one wheat feed per animal/day, care bonus reset on production, held-yield cap, and fertilizer availability reset.

- [ ] **Step 4: Implement forecast/scoring functions**

Add `forecast_crop`, `forecast_animal`, `feed_reserve`, `expected_portfolio_cash`, and `opportunity_score`. Include purchase costs, projected feed, land/hire costs, movement/action turns supplied by the caller, and a risk penalty for price-floor exposure and shed overflow.

- [ ] **Step 5: Run economics tests**

Run `pytest tests/test_economics.py -q`.
Expected: exact curve fixtures and production fixtures pass; any mismatch is treated as a rules/version bug before strategy tuning continues.

### Task 4: Add routing and worker task scheduling

**Files:**
- Create: `kaggriculture_agent/routing.py`
- Create: `kaggriculture_agent/planner.py`
- Test: `tests/test_routing.py`

- [ ] **Step 1: Write route tests**

Assert deterministic Manhattan paths, no movement off the board, pass-through of locked tiles, rejection of locked tile-action targets, stable tie-breaking, and no duplicate assignment of two workers to the same exclusive tile task.

- [ ] **Step 2: Implement routing primitives**

Add `distance`, `next_move`, `route_to`, `nearest_target`, and `route_action`. Use `NORTH/SOUTH/EAST/WEST` with y increasing downward. Return `PASS` when already at target but the target has no legal action.

- [ ] **Step 3: Implement daily task generation**

Add `build_daily_plan(state, memory)` that emits tasks for urgent water/feed/care work, positive-yield harvests, structure/animal placement, weed clearing, planting, shed logistics, and selling. Set hard deadlines before end-of-day refresh and assign economic values from `economics.py`.

- [ ] **Step 4: Implement assignment**

Add `assign_tasks(plan, workers, state)` using urgency, deadline slack, and cash per action. Keep the farmer available for shed access and reserve at least one worker for unmet basic-needs tasks.

- [ ] **Step 5: Run routing tests**

Run `pytest tests/test_routing.py -q`.
Expected: all path and scheduling invariants pass.

### Task 5: Implement policy and legal action emission

**Files:**
- Create: `kaggriculture_agent/memory.py`
- Create: `kaggriculture_agent/policy.py`
- Test: `tests/test_policy.py`

- [ ] **Step 1: Write policy contract tests**

For representative observations, assert that policy output contains one farmer command, exactly one command per visible hand, a list of no more than ten market orders, valid command shapes, no selling above shed inventory, no buying above cash/reserve, no feeding without wheat, and no tile action on a locked/occupied/incompatible tile.

- [ ] **Step 2: Implement memory reset and daily planning**

`Policy.act(obs)` should parse state, reset memory at an episode boundary, rebuild the macro plan at hour 0 or when the observed shop/market regime changes, and otherwise continue an assignment only if its target and prerequisites are still valid.

- [ ] **Step 3: Implement market order construction**

Add `build_market_orders(state, plan)` with explicit order priority: required feed/wheat purchase, approved land/hire/seed/animal purchases, bounded sells, and final-turn liquidation. Truncate to ten and avoid duplicate orders for the same resource in one queue unless a deliberate batch split is selected.

- [ ] **Step 4: Implement worker action selection**

Add `worker_action(worker_index, state, assignment)` that moves toward the target, performs the exact action when adjacent/on-target, and falls back to `PASS`. Handle `PICKUP`, `DROP`, and `PLACE` only with verified shed adjacency/occupancy; never assume a purchased animal is in a worker inventory because the engine places it in the shed.

- [ ] **Step 5: Run policy tests**

Run `pytest tests/test_policy.py -q`.
Expected: output contracts pass for initial, mid-season, full-shed, animal, locked-land, and final-day fixtures.

### Task 6: Integrate the real local engine

**Files:**
- Create: `scripts/run_local.py`
- Create: `tests/test_agent_smoke.py`

- [ ] **Step 1: Add the single-game runner**

Use `make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)`, run `[agent, opponent]`, print both final rewards/statuses, and write `env.toJSON()` to `replays/seed-<seed>-<opponent>.json`.

- [ ] **Step 2: Add smoke tests against built-ins**

Run short 96-turn games against `pass`, `random`, and `starter`, then one full 720-turn seeded game against `starter`. Assert both agents finish without errors, final observations contain money, and the custom agent emits no framework error.

- [ ] **Step 3: Inspect replay invariants**

Scan the replay after each smoke game for malformed action dictionaries, accidental repeated no-op loops, unwatered plants reaching two misses, unfed animals reaching two misses, shed totals above 100, or harvests after the relevant decay point.

- [ ] **Step 4: Run the integration tests**

Run `pytest tests/test_agent_smoke.py -q`.
Expected: short games and the seeded full game complete with status `DONE` for both seats.

### Task 7: Tune with seeded evaluation and public competition feedback

**Files:**
- Create: `scripts/evaluate.py`
- Modify: `README.md`

- [ ] **Step 1: Implement batch evaluation**

Run at least 30 seeds per opponent and report mean/median/fifth-percentile final bank, win rate, error rate, average shed overflow, average price-floor sales, and missed basic-needs events. Store JSON summaries beside replays.

- [ ] **Step 2: Add controlled variants**

Evaluate the conservative, melon-heavy, demand-reactive, and animal-heavy portfolio settings against identical seed/opponent sets. Select the default by win rate first, then median bank and failure rate; do not select by one lucky seed.

- [ ] **Step 3: Add ablations**

Measure the contribution of route scheduling, market batch sizing, shop adaptation, land purchase, and animals by disabling one component at a time. Keep only changes that improve win rate without increasing invalid actions or basic-needs failures.

- [ ] **Step 4: Add submission checks**

Package with `tar -czf submission.tar.gz main.py kaggriculture_agent` and inspect the archive so `main.py` is at its root. Run the exact packaged entry point locally before any upload.

- [ ] **Step 5: Run the final local gate**

Run `pytest -q`, then `python3 scripts/evaluate.py --seeds 30 --opponents pass random starter`.
Expected: zero test failures, zero framework errors, zero malformed actions, and a recorded JSON report for the selected policy.

### Task 8: Submit and monitor safely

**Files:**
- Modify: `README.md` with the selected variant and command history
- Generated locally: `submission.tar.gz`

- [ ] **Step 1: Accept the competition rules in the Kaggle UI**

This is required before submission and is an external account action; complete it manually if not already done.

- [ ] **Step 2: Verify the installed CLI and membership**

Run `kaggle competitions list -s kaggriculture` and `kaggle competitions list --group entered`.

- [ ] **Step 3: Submit one validated variant**

Run `kaggle competitions submit kaggriculture -f submission.tar.gz -m "model-based route-aware agent v1"`.

- [ ] **Step 4: Monitor validation and episodes**

Run `kaggle competitions submissions kaggriculture`, then use the returned submission ID with `kaggle competitions episodes <SUBMISSION_ID>` and download logs/replays for any error episode.

- [ ] **Step 5: Record outcomes before changing policy**

For every submission, record engine/version, variant name, local seed results, validation status, episode count, win/loss/tie trend, and the exact code archive. Keep the best two active submissions in mind because only the latest two are tracked for final evaluation.

## Verification checklist

Before calling the agent ready:

- `pytest -q` passes.
- `kaggle-environments` is at least `1.32.7`.
- A 720-turn run against each built-in opponent reaches `DONE` with no framework error.
- Every output has one action per worker and no more than ten market orders.
- Plants are never intentionally allowed to miss two refreshes; animals are never intentionally allowed to miss two feeds.
- Shed capacity is tracked and overflow is treated as lost cash, not ignored.
- Land/hire/animal/seed purchases respect cash reserves and fixed engine ordering.
- Market sells use observed prices and bounded batches; final liquidations occur before the episode ends.
- Replay summaries and logs can explain each major macro decision.
- The archive contains `main.py` at its root and imports no files outside the archive.

## Sources consulted

- [Kaggriculture competition page](https://www.kaggle.com/competitions/kaggriculture) — current rules, evaluation, timeline, object tables, observation format, and submission instructions.
- [Kaggle kaggle-environments Kaggriculture README](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/README.md) — published Python-kit rules.
- [Kaggriculture engine source](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.py) — exact action handling, market lockstep, land order/prices, shop demands, and refresh mechanics.
- [Kaggriculture AGENTS.md](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/AGENTS.md) — local test and submission workflow.

## Completion update (2026-08-29)

Tasks 1–7 are implemented and locally reviewed. The submission entry point, defensive observation model, exact economics, routing/planning, legal policy actions, local runner, seeded evaluator, variant matrix, isolated ablations, replay sidecars, and packaging smoke checks are present in the repository. Task 8 remains intentionally manual because competition acceptance, submission, and monitoring require external Kaggle account actions.

The evaluator review follow-ups are also complete:

- replay actions are checked against the observation before the action and their deterministic post-state effects, including worker movement, board changes, seeds, inventories, cash, land, builds, and market purchases/sales;
- both players’ market queues are simulated in shared per-unit lockstep for validation and floor-sale metrics;
- malformed replay metadata, info, configuration, status, steps, and action structures become framework failures rather than evaluator exceptions;
- default variant selection prioritizes framework reliability before win rate and bank tie-breakers;
- isolated one-component ablations, pre-transition overflow, real 23→0 need boundaries, final-state need confirmation, and deterministic batch reporting are covered by regression tests.

The maintained verification command is:

    UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest -q
- Kaggle Discussion topic “Small balance change” — rationale and observed effects of the `1.32.7` carrot/tomato/egg scarcity update.
