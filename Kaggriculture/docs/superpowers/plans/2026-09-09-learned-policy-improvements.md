# Learned-Policy and Core-Policy Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the confirmed trajectory, deadline, inventory, terminal-market, and train/runtime-parity defects that can teach or preserve bad behavior in Kaggriculture policies, then establish evidence gates before promoting a learned artifact.

**Architecture:** Keep the deterministic planner as the safety fallback and improve it first. Add regression tests for every confirmed behavioral defect. Then make the learned runtime obey the same terminal and safety semantics, cache/load artifacts safely, and enforce a single numerical/action contract. Training-metric and action-space changes are isolated behind explicit compatibility decisions so an existing artifact is never silently reinterpreted.

**Tech Stack:** Python 3.11+, uv, pytest, NumPy, PyTorch training/export code, JSON learned-policy artifacts.

---

## Scope and confirmed findings

The reports are treated as review inputs, not as instructions. This plan includes confirmed issues that can change training trajectories or learned-policy behavior:

- planner assignment starvation when one worker is reserved for basic needs (C1);
- destructive bulk DROP when shed capacity is insufficient (C2);
- late-season crop/animal purchases and planting that cannot mature (W2);
- missing drop/feed detours in task budgets (W5, W12);
- shed work assigned to an empty worker (W6, W11);
- flat-state seed fallback that turns hands into phantom seeds (W13);
- feed-buy orders removed by market-direction filtering (W10);
- learned terminal liquidation divergence (W9);
- learned artifact load retry and first-turn loading (learned W1, W4);
- learned inference timing without an effective safety gate (learned C1);
- GELU mismatch between torch, NumPy, and pure-Python inference (learned C3);
- dead/unsupported MOVE and PASS action labels (learned W3);
- misleading PPO KL metric naming (learned W5).

Keep these as explicit decisions or cleanup, not automatic behavioral changes:

- Policy(strategy="current") remaining deterministic/legacy (W1);
- double market-impact penalty (W7);
- buy-only learned market head (W2);
- auto-mode crop mapping (W8);
- the no-wheat animal wait path (W14);
- memory pruning, hard-coded 24-hour constants, positional hand indexing, duplicate checks, and dead locals (N1-N8);
- market-history reconciliation and terminal wheat-reserve semantics (W4/W15) unless the product owner chooses the desired economic objective.

The current baseline is not green: uv run pytest -q reported 1327 passed, 6 failed, 1 skipped. The six failures are an existing smoke/evaluation regression cluster, including missing FERTILIZE behavior and seed-17 conservative replay failures. No promotion gate may be marked green until that baseline cluster is resolved or explicitly quarantined with an owner and rationale.

## Files to touch

- Kaggriculture/kagriculture_agent/planner.py
- Kaggriculture/kagriculture_agent/policy.py
- Kaggriculture/kagriculture_agent/candidates.py
- Kaggriculture/kagriculture_agent/learned_policy.py
- Kaggriculture/kagriculture_agent/model.py
- Kaggriculture/scripts/train_policy.py
- Kaggriculture/tests/test_policy_improvements.py (new)
- Kaggriculture/tests/test_policy.py
- Kaggriculture/tests/test_candidates.py
- Kaggriculture/tests/test_export_policy.py
- Kaggriculture/tests/test_train_policy.py
- Kaggriculture/tests/test_evaluate.py
- Kaggriculture/tests/test_agent_smoke.py
- Kaggriculture/README.md

## Task 1: Establish the regression harness

Files: Kaggriculture/tests/test_policy_improvements.py (new), with local helpers in that file.

- [ ] Add local state builders for one worker, two workers, shed inventory, carried inventory, day/hour, market quotes, and private seeds. Do not import helpers from another test module; keep each regression independently runnable.

- [ ] Add these tests before changing production code. Each must fail for the current behavior and assert the direct policy/planner output:

    def test_assign_tasks_falls_back_to_reserved_worker_when_only_farmer():
        assignments = assign_tasks(state_with_one_farmer_and_tasks("ANIMAL", "WATER"))
        assert assignments[0].kind == "ANIMAL"
        assert assignments[1].kind == "WATER"

    def test_drop_carried_goods_never_bulk_drops_over_capacity():
        actions = act_at_shed_with_inventory(shed={"CARROT": 99}, carried={"MELON": 2}, force=True)
        assert actions[0] in {"PLACE MELON 1", "PASS"}
        assert actions[0] != "DROP"

    def test_day_27_does_not_plan_melon_purchase_or_planting():
        plan = plan_for_day(27, selected_crop="MELON", empty_tiles=2)
        assert all(task.kind not in {"BUY_SEED", "PLANT"} or task.item != "MELON" for task in plan)

    def test_shed_assignment_prefers_worker_with_inventory():
        assignments = assign_tasks(state_with_carried_goods(worker=1, shed_task=True))
        assert assignments[1].kind == "SHED"

    def test_animal_budget_includes_feed_after_shed_pickup():
        budget = task_turn_budget(animal_task_with_shed_wheat_and_worker_inventory())
        assert budget == 11

    def test_normalize_planner_state_never_uses_hands_as_seeds():
        state = normalize_state(farm={"hands": {"WHEAT": 5}, "seeds": {}}, private={})
        assert state.seeds == {}

    def test_basic_need_wheat_buy_survives_direction_filter():
        market = guarded_market_at_turn_with_reversal_memory(item="WHEAT")
        assert ["BUY_PRODUCT", "WHEAT", 2] in market

    def test_sell_assignment_validates_carried_inventory():
        assignments = assign_tasks(state_with_carried_product_and_empty_shed("EGG"))
        assert assignments[0].kind == "SELL"

    def test_learned_policy_liquidates_shed_inventory_in_terminal_window():
        deterministic = terminal_actions(strategy="current")
        learned = terminal_actions(learned_artifact="models/learned_v1.json")
        assert learned.market == deterministic.market

- [ ] Add a load-cache test in Kaggriculture/tests/test_candidates.py with a counting loader: two calls to the same candidate policy load the artifact once, and a missing/corrupt artifact produces one cached failure diagnostic rather than retrying every turn.

- [ ] Add a torch/NumPy/pure-runtime parity test in Kaggriculture/tests/test_export_policy.py. Compare logits with rtol=1e-5 and atol=1e-6, not only argmax actions.

- [ ] Run only the new tests:

    cd /Users/bachir/co/github/dzlab/kaggle/Kaggriculture
    uv run pytest -q tests/test_policy_improvements.py tests/test_candidates.py tests/test_export_policy.py

  Expected result before fixes: the new behavioral tests fail; unrelated existing tests are not evidence for this task.

- [ ] Commit the red regression harness with message test: add policy and learned-runtime regressions.

## Task 2: Fix planner assignment and logistics accounting

Files: Kaggriculture/kagriculture_agent/planner.py and Kaggriculture/tests/test_policy_improvements.py.

- [ ] In assign_tasks.choose, retain the reserved basic-needs worker only when another candidate exists. For a non-basic task, use filtered candidates if non-empty and otherwise restore the original candidates:

    non_reserved = [info for info in candidates if info[0] != reserved_basic[0]]
    candidates = non_reserved or candidates

- [ ] For a generic SHED task, rank workers carrying positive inventory ahead of empty workers. Derive carried quantity from normalized worker state and then use existing distance/availability tie-breakers. If a legacy fixture has no represented inventory, preserve the current candidate set.

- [ ] Correct _task_turn_budget: when a worker must pick up shed wheat before feeding an animal, count pickup and feed turns in addition to both movement legs. The existing geometry repro must remain 11 turns.

- [ ] Change planner-state normalization so an explicitly present empty seeds mapping stays empty. Only accept the private/farm seed mapping when the primary field is absent or is not a mapping; never fall back to hands for seeds.

- [ ] Run:

    uv run pytest -q tests/test_policy_improvements.py -k 'assign_tasks or shed_assignment or animal_budget or normalize_planner_state'

  Expected result: all selected tests pass, including the one-worker ANIMAL then WATER assignment.

- [ ] Commit as fix: preserve planner assignments and logistics deadlines.

## Task 3: Make carried-goods cleanup capacity-safe

Files: Kaggriculture/kagriculture_agent/policy.py, Kaggriculture/tests/test_policy_improvements.py, and relevant existing tests in Kaggriculture/tests/test_policy.py.

- [ ] Replace unconditional terminal/forced DROP behavior in _drop_carried_goods with capacity-aware placement:

  1. Calculate free shed room from configured capacity and current shed inventory.
  2. If all carried goods fit, retain full-drop behavior only where the engine contract is lossless.
  3. If only part fits, emit bounded PLACE item quantity for a deterministic product and leave the remainder carried.
  4. If no room exists, emit no destructive cleanup action and let normal fallback preserve the goods.
  5. force=True may bypass cleanup preference, but never the capacity check.

- [ ] Cover shed occupancy 99/100 with two carried units, 100/100, mixed carried products, and terminal cleanup. Assert that goods represented by the resulting state never decrease merely because cleanup was requested.

- [ ] Preserve animal handling: never turn a carried animal into product placement. Add a full-shed carried-animal regression.

- [ ] Run:

    uv run pytest -q tests/test_policy_improvements.py -k 'drop_carried_goods or terminal_cleanup'
    uv run pytest -q tests/test_policy.py -k 'drop or shed or terminal'

  Expected result: no bulk DROP under insufficient capacity and prior lossless cleanup tests pass.

- [ ] Commit as fix: make carried-goods cleanup lossless.

## Task 4: Enforce crop and animal maturation horizons

Files: Kaggriculture/kagriculture_agent/planner.py, Kaggriculture/tests/test_policy_improvements.py, and late-season tests in Kaggriculture/tests/test_policy.py.

- [ ] Add one planner helper that answers whether a new crop can produce before season end using CROPS[crop]["first_yield_day"] and the day indexing already used by economics.py. Add an animal helper using ANIMALS[animal]["first_yield_day"] and include existing build/structure action turns in the start-day calculation:

    def _can_produce_before_season_end(first_yield_day, start_day, season_days):
        return start_day + int(first_yield_day) <= season_days - 1

- [ ] Apply the helper consistently to _portfolio_scenarios, macro BUY_SEED and BUY_ANIMAL tasks, daily empty-tile planting, and animal/structure purchase/build tasks.

- [ ] Add tests for day-27 MELON (no BUY_SEED or PLANT), day-27 WHEAT/CARROT (still allowed when existing rules permit), day 28 (no new crop), and a late animal whose build plus first-yield horizon exceeds season end.

- [ ] Keep already planted crops and existing animals harvestable/feedable; the horizon gate applies only to new commitments.

- [ ] Run:

    uv run pytest -q tests/test_policy_improvements.py -k 'day_27 or late_season or horizon'
    uv run pytest -q tests/test_policy.py -k 'late_season or portfolio or animal'

  Expected result: no impossible late-season commitments and no regression in day-28 no-new-crops behavior.

- [ ] Commit as fix: gate new production by remaining season horizon.

## Task 5: Preserve safety-critical feed purchases and learned terminal parity

Files: Kaggriculture/kagriculture_agent/policy.py, Kaggriculture/tests/test_policy_improvements.py, and Kaggriculture/tests/test_policy.py.

- [ ] Refactor the basic-needs guard and market-direction filter to use an explicit protected-direction set:

    market, protected = self._basic_need_guard(state, market)
    market = self._filter_market_direction(state, market, protected=protected)

  The guard adds ("WHEAT", "BUY_PRODUCT") only when the order covers projected feed need and passes existing cash/market-capacity checks. Discretionary reversed wheat remains filterable.

- [ ] Update direct guard/filter tests to assert both the protected feed order and removal of an unprotected reversed order.

- [ ] Unify terminal liquidation in Policy.act. Learned and deterministic proposals must both permit the final liquidation window when there are no carried goods, including day 29/hour 22. Learned inference may add/override actions, but may not remove deterministic terminal liquidation.

- [ ] Add a test comparing deterministic and learned market orders for the same terminal observation with an artifact that proposes no market orders. Learned output must contain the deterministic SELL order.

- [ ] Run:

    uv run pytest -q tests/test_policy_improvements.py -k 'basic_need or learned_policy_liquidates'
    uv run pytest -q tests/test_policy.py -k 'market_direction or basic_need or terminal_liquidation'

- [ ] Commit as fix: preserve feed safety and terminal liquidation parity.

## Task 6: Make surplus-selling and terminal inventory semantics explicit

Files: Kaggriculture/kagriculture_agent/planner.py, Kaggriculture/kagriculture_agent/policy.py, and policy/economics tests.

- [ ] Change generic SELL_ALL planning so it does not sell wheat reserved for projected live-animal feed. Generate deterministic surplus quantities per product and do not create a sell task when the only inventory is protected feed.

- [ ] Validate SELL and SELL_ALL against worker carried inventory at execution time. A shed-only inventory check is insufficient because the task action sells carried goods. Keep pickup/sale conflict removal and test a worker carrying EGG with zero EGG in the shed.

- [ ] Decide and encode the final-turn economic contract: if terminal cash is the objective, sell saleable wheat beyond consumed feed; if terminal inventory has value, retain the documented reserve. Do not change the existing final-turn reserve test without recording this decision.

- [ ] Run the economics/policy tests, inspect one full replay action trace, and commit as fix: sell only safe carried and surplus inventory.

## Task 7: Harden learned artifact loading and latency behavior

Files: Kaggriculture/kagriculture_agent/learned_policy.py, Kaggriculture/kagriculture_agent/candidates.py, Kaggriculture/tests/test_candidates.py, and Kaggriculture/tests/test_policy_improvements.py.

- [ ] Change candidate_policy to load and validate the artifact once, then pass the resulting DependencyFreePolicy object into Policy. Match artifact_candidate_policy one-load behavior and expose the loaded object through the existing __self__ test hook consistently.

- [ ] Add LearnedPolicy load state: unloaded, loaded, or failed. Cache the first terminal load exception and return deterministic fallback on later calls without reopening or revalidating the path. Include path and exception type in a non-sensitive diagnostic.

- [ ] Measure every actual learned inference with time.monotonic(). If it exceeds the configured budget, record slow_inference, return no learned override for that turn, and disable learned overrides for the rest of the episode so deterministic policy owns the trajectory. Keep artifact-load time separate from steady-state inference time.

- [ ] Do not claim hard preemption for in-process NumPy/Python calls. Test the observable safety contract: slow/failed inference cannot suppress deterministic actions, and known terminal load failures are not retried.

- [ ] Add a full-observation benchmark (100 tiles, 10 workers, 9 market rows) reporting p95 and max. The release threshold is the configured budget; the small-fixture performance test is not release proof.

- [ ] Run:

    uv run pytest -q tests/test_candidates.py tests/test_policy_improvements.py -k 'load or timeout or latency'
    uv run pytest -q tests/test_export_policy.py -m performance

- [ ] Commit as fix: cache learned artifacts and enforce inference safety.

## Task 8: Make neural activation numerically identical across runtimes

Files: Kaggriculture/kagriculture_agent/model.py, Kaggriculture/kagriculture_agent/learned_policy.py, and export tests.

- [ ] Make exact GELU the compatibility contract for learned_v1. Implement the NumPy path with the same erf definition as torch using a tested vectorized math.erf adapter if the supported NumPy version has no erf ufunc; keep pure Python mathematically identical.

- [ ] Run identical weights and inputs through torch, NumPy runtime, and pure-Python runtime. Assert logits with rtol=1e-5 and atol=1e-6 and assert identical selected actions.

- [ ] Run parity and performance together. If exact NumPy GELU breaches the full-state latency budget, optimize the exact kernel or create an explicitly versioned artifact/training contract using tanh-GELU everywhere; never mix a new activation with learned_v1 weights.

- [ ] Commit as fix: align learned activation across runtimes.

## Task 9: Close the learned action-vocabulary contract

Files: Kaggriculture/kagriculture_agent/learned_policy.py, Kaggriculture/scripts/train_policy.py, and train/export/runtime tests.

- [ ] Add one exported vocabulary validator used by training, export, artifact load, and proposal compilation. It rejects a class the compiler cannot execute and reports vocabulary/version in artifact diagnostics.

- [ ] Preserve backward loading for learned_v1, but mark MOVE and PASS as non-overriding classes: when selected, ignore that proposal and leave the deterministic planner authoritative for that worker. Test that such a proposal cannot erase a valid deterministic task.

- [ ] For newly trained artifacts, mask MOVE/PASS rows out of task-intent classification loss unless the runtime contract is deliberately expanded. Store the mask in exported metadata.

- [ ] Add a train/export/load/compile round-trip test proving every emitted proposal kind is in the compiler valid set.

- [ ] Do not remove or renumber classes in learned_v1; a class-table change requires a new artifact version and retraining/export.

- [ ] Commit as fix: enforce learned action vocabulary contract.

## Task 10: Correct training telemetry and defer objective changes behind evidence

Files: Kaggriculture/scripts/train_policy.py, Kaggriculture/tests/test_train_policy.py, and Kaggriculture/README.md.

- [ ] Rename the pre-update ratio metric to pre_step_approx_kl and retain post_step_kl measured after the optimizer step. If compatibility requires approx_kl, alias it to pre_step_approx_kl:

    metrics["pre_step_approx_kl"] = float(pre_step_approx_kl)
    metrics["post_step_kl"] = float(post_step_kl)
    metrics["approx_kl"] = metrics["pre_step_approx_kl"]

- [ ] Update tests with known old/new log probabilities and assert both values are reported under the correct update phase.

- [ ] Before changing market_active or joint-action objective weighting, run an identical-seed ablation reporting validation reward, feed-deadline violations, terminal cash, artifact latency, and action-contract violations. The runtime currently does not consume the exported market-active head, so an objective change without runtime consumption is not an automatic fix.

- [ ] Record the chosen objective in README.md; do not silently change loss weighting while repairing telemetry.

- [ ] Commit as fix: label policy training metrics by update phase.

## Task 11: Resolve the pre-existing baseline failures and verify the complete gate

Files: Kaggriculture/tests/test_agent_smoke.py, Kaggriculture/tests/test_evaluate.py, and only production files implicated by their traces.

- [ ] Reproduce the six known failures independently and capture the first failing action/observation transition:

    cd /Users/bachir/co/github/dzlab/kaggle/Kaggriculture
    uv run pytest -q tests/test_agent_smoke.py::test_full_pass_exercises_autonomous_macro_action_and_market_flows
    uv run pytest -q tests/test_evaluate.py -k 'seed17 and conservative'

- [ ] Fix root causes or update stale assertions only when the current engine contract proves the assertion obsolete. Add a regression for the chosen behavior; never mark these tests xfail merely to make the suite green.

- [ ] Run the complete verification sequence:

    uv run pytest -q
    uv run pytest -q -m performance
    uv run python scripts/evaluate_policy.py --help
    git diff --check
    git status --short

  Expected result: zero unexpected test failures, p95/max evidence for the full learned observation shape, clean whitespace, and only intentional source/test/doc changes.

- [ ] Run the documented holdout evaluation from README.md for deterministic current and learned candidates. Save comparison of reward, basic-need deadline failures, framework errors, terminal liquidation, inference p95/max, and artifact SHA.

- [ ] Promotion gate: select learned only with zero framework errors, zero safety/deadline regressions against deterministic baseline, valid action vocabulary, activation parity, cached loading, and latency within budget. Otherwise keep current as default and retain the learned artifact for further training.

- [ ] Commit verification/doc updates as docs: define learned-policy promotion evidence.

## Explicit decisions required before optional follow-up work

1. Whether terminal inventory has value after the final action; this determines W15 wheat-reserve behavior.
2. Whether to retain double market-impact penalty; if not, change the economics test and document the risk objective.
3. Whether to expand the learned market head beyond feed purchases and whether runtime will consume market_active.
4. Whether auto mode should include strawberry/tomato route mappings.
5. Whether market-direction history should reconcile executed rather than emitted orders.
6. Whether to introduce a new artifact version for redesigned task-intent vocabulary instead of backward-compatible MOVE/PASS masking.

Make these decisions after Tasks 1-8 produce clean behavior and parity evidence; do not hide them inside a bug-fix commit.
