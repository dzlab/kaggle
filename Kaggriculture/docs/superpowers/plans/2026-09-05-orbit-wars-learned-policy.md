# Orbit Wars-Inspired Learned Kaggriculture Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a compact, target-based self-play policy to Kaggriculture while keeping the current deterministic policy as the safety baseline and promotion fallback.

**Architecture:** Work in four independently testable layers: clean up the deterministic baseline and evaluator; add compact tile/worker/market features and a target-based proposal compiler; train a small PPO policy against a checkpoint league; export dependency-light quantized inference and promote it only through the existing holdout gates. The production entrypoint remains the current policy until a learned candidate is proven better.

**Tech Stack:** Python 3.11+, kaggle-environments 1.32.7, pytest, optional PyTorch/NumPy training dependencies, standard-library runtime inference, JSON/array model artifacts, and the existing seeded replay evaluator. The target-action and checkpoint/self-play ideas come from the Orbit Wars article at https://tufalabs.ai/research/orbit-wars/; its 200M-parameter scale and Rust rewrite are not assumed initially.

---

## File and module boundaries

- Modify Kaggriculture/scripts/evaluate.py for per-game safety diagnostics and learned-candidate evaluation.
- Modify Kaggriculture/kagriculture_agent/policy.py for market safety, learned-policy integration, proposal compilation, and fallback behavior.
- Modify Kaggriculture/kagriculture_agent/memory.py for episode-local market cooldown and learned diagnostics.
- Modify Kaggriculture/kagriculture_agent/strategy.py only for shared market scoring/signals.
- Create Kaggriculture/kagriculture_agent/features.py for deterministic compact feature extraction; it must not import PyTorch.
- Create Kaggriculture/kagriculture_agent/learned_policy.py for runtime proposal types, artifact loading, quantized inference, and fallback behavior.
- Create Kaggriculture/kagriculture_agent/model.py for the training-time neural architecture and export format.
- Create Kaggriculture/kagriculture_agent/trajectory.py for replay-to-transition conversion and schema validation.
- Create Kaggriculture/scripts/collect_trajectories.py, train_policy.py, export_policy.py, and benchmark_rollouts.py for offline workflows.
- Modify Kaggriculture/kagriculture_agent/candidates.py to register learned_v1 only after an artifact exists; leave current, melon, premium, and mixed unchanged.
- Modify Kaggriculture/main.py only after holdout promotion.
- Add focused tests under Kaggriculture/tests. Keep generated reports, trajectories, checkpoints, and model artifacts outside source control unless explicitly requested.

## Task 1: Establish a clean, diagnosable baseline

**Files:**
- Modify: Kaggriculture/scripts/evaluate.py
- Test: Kaggriculture/tests/test_evaluate.py
- Create: Kaggriculture/reports/README.md

- [ ] **Step 1: Add explicit per-game failure reasons.**

Extend each replay record with framework_error_reasons, preserving the existing framework_error boolean. Use this fixed order and vocabulary:

~~~python
FRAMEWORK_REASON_ORDER = (
    "engine_error",
    "malformed_replay",
    "missed_basic_needs",
    "market_churn",
    "market_transaction_cap",
    "terminal_cash_floor",
    "terminal_inventory_floor",
    "incomplete_pairing",
)
~~~

Add each reason at most once; set framework_error if the list is non-empty. Do not remove or rename existing report fields.

- [ ] **Step 2: Write tests for ordered diagnostics.**

Add tests with synthetic records that assert:

~~~python
assert record["framework_error"] is True
assert record["framework_error_reasons"] == [
    "missed_basic_needs",
    "market_churn",
]
~~~

Also assert that a clean legacy fixture gets framework_error_reasons equal to an empty list and remains valid.

- [ ] **Step 3: Document and run the baseline.**

Create Kaggriculture/reports/README.md with the command below and the rule that a candidate is ineligible until both seats have complete records, zero framework failures, and zero missed basic-needs events.

~~~bash
Kaggriculture/.venv/bin/python Kaggriculture/scripts/evaluate.py \
  --seeds 30 --start-seed 0 --steps 720 \
  --opponents pass random starter --seats 0 1 \
  --candidates current melon premium mixed \
  --output /private/tmp/kaggriculture-baseline.json
~~~

Run it, inspect reason counts and the replay sidecar, and record the source commit SHA and command in report metadata. Do not overwrite the checked-in legacy report.

- [ ] **Step 4: Test and commit.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests/test_evaluate.py -q
git -C Kaggriculture add scripts/evaluate.py tests/test_evaluate.py reports/README.md
git -C Kaggriculture commit -m "test: make baseline evaluation failures diagnosable"
~~~

## Task 2: Eliminate deterministic-policy safety regressions

**Files:**
- Modify: Kaggriculture/kagriculture_agent/memory.py
- Modify: Kaggriculture/kagriculture_agent/policy.py
- Modify: Kaggriculture/kagriculture_agent/planner.py
- Test: Kaggriculture/tests/test_policy.py
- Test: Kaggriculture/tests/test_agent_smoke.py

- [ ] **Step 1: Add episode-local market direction memory.**

Add this field to PolicyMemory, clear it from reset(), and add the helper:

~~~python
market_history: dict[str, list[tuple[int, str]]] = field(default_factory=dict)

def market_order_allowed(
    memory: PolicyMemory,
    *,
    item: str,
    direction: str,
    turn: int,
    window: int = 2,
    terminal: bool = False,
) -> bool:
    """Reject only an opposite buy/sell direction within window turns."""
~~~

Allow the first order, repeated orders in the same direction, and terminal liquidation. Reject only an opposite BUY_PRODUCT/SELL direction within the window.

- [ ] **Step 2: Write churn tests.**

Test first order, same-direction repeat, reverse order inside the two-turn window, reverse order after the window, terminal override, and reset between episodes.

- [ ] **Step 3: Apply the guard at the market boundary.**

In Policy.act, filter approved orders immediately before _remove_pickup_sale_conflicts, and record accepted orders in memory.market_history. Preserve affordability, shed-capacity, max-order, and price-floor checks.

- [ ] **Step 4: Add a hard basic-needs reservation invariant.**

Before accepting BUY_LAND, BUY_ANIMAL, HIRE, or nonessential crop purchases, calculate remaining feed with _feed_purchase_needed(). Reject the purchase if projected cash falls below the wheat reserve or if a due WATER/FEED task lacks a valid assignment. Purchases needed for an already-due feed deadline remain allowed.

Record:

~~~python
memory.diagnostics["reserved_wheat"] = int(required_wheat)
memory.diagnostics["basic_need_guard"] = "pass"  # or "blocked"
~~~

- [ ] **Step 5: Add regression tests.**

Use a live-animal state with insufficient wheat and assert land/hire/animal purchases are omitted. Use sufficient wheat and assert the purchase remains. Use an overdue watering task and assert its assignment survives replanning.

- [ ] **Step 6: Run safety checks.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest \
  Kaggriculture/tests/test_policy.py \
  Kaggriculture/tests/test_agent_smoke.py -q
Kaggriculture/.venv/bin/python Kaggriculture/scripts/evaluate.py \
  --seeds 5 --start-seed 0 --steps 720 \
  --opponents pass random starter --seats 0 1 \
  --candidates current mixed \
  --output /private/tmp/kaggriculture-safety-regression.json
~~~

Continue only when the report has no missed_basic_needs or market_churn reasons for either candidate. Fix any engine legality error before learned-policy work.

- [ ] **Step 7: Commit.**

~~~bash
git -C Kaggriculture add kagriculture_agent/memory.py kagriculture_agent/policy.py kagriculture_agent/planner.py tests/test_policy.py tests/test_agent_smoke.py
git -C Kaggriculture commit -m "fix: prevent market churn and need-solvency regressions"
~~~

## Task 3: Define compact, versioned feature extraction

**Files:**
- Create: Kaggriculture/kagriculture_agent/features.py
- Test: Kaggriculture/tests/test_features.py
- Modify: Kaggriculture/pyproject.toml

- [ ] **Step 1: Define the immutable feature batch.**

~~~python
FEATURE_SCHEMA_VERSION = 1

@dataclass(frozen=True)
class FeatureBatch:
    tile_tokens: tuple[tuple[float, ...], ...]
    worker_tokens: tuple[tuple[float, ...], ...]
    market_tokens: tuple[tuple[float, ...], ...]
    global_tokens: tuple[float, ...]
    tile_positions: tuple[Position, ...]
    schema_version: int = FEATURE_SCHEMA_VERSION
~~~

extract_features(state) must be deterministic, non-mutating, and return correctly shaped empty features for malformed input.

- [ ] **Step 2: Implement fixed token layouts.**

Document and implement:

- Tile: normalized coordinates, lock/empty flags, crop one-hot, crop age, yield timing, decay timing, watering/fertilizer flags, structure/animal one-hot, animal needs, expected harvest units, and task-deadline slack.
- Worker: normalized position, farmer/hand role, held-item one-hot, worker index, current task kind, target distance, and deadline slack.
- Market: product one-hot, quote, inventory delta from MARKET_I0, sequential post-sale quote, town-demand flag, and price-floor flag.
- Global: day/hour, cash, shed utilization, wheat reserve, free workers, unlocked land, production, terminal horizon, and strategy one-hot.

Sort tile positions and product names deterministically. Keep opponent private state out of the features.

- [ ] **Step 3: Write feature tests.**

Test stable ordering, no mutation, fixed shapes for a 10x10 board, private-state isolation, market quote changes, malformed-observation handling, and feature-schema mismatch rejection.

- [ ] **Step 4: Add optional training dependencies.**

Add an optional training group containing NumPy and PyTorch. Keep base runtime dependencies unchanged and update uv.lock.

- [ ] **Step 5: Test and commit.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests/test_features.py -q
git -C Kaggriculture add kagriculture_agent/features.py tests/test_features.py pyproject.toml uv.lock
git -C Kaggriculture commit -m "feat: add compact versioned Kaggriculture features"
~~~

## Task 4: Add a target-based proposal and deterministic compiler

**Files:**
- Create: Kaggriculture/kagriculture_agent/learned_policy.py
- Modify: Kaggriculture/kagriculture_agent/policy.py
- Test: Kaggriculture/tests/test_learned_policy.py
- Test: Kaggriculture/tests/test_policy.py

- [ ] **Step 1: Define proposal types.**

~~~python
@dataclass(frozen=True)
class WorkerProposal:
    worker_index: int
    kind: str
    target: Position | None
    item: str | None
    score: float

@dataclass(frozen=True)
class PolicyProposal:
    workers: tuple[WorkerProposal, ...]
    market_orders: tuple[tuple[str, str | None, int], ...]
    confidence: float
    model_version: str
~~~

The policy proposes a task target, not a path. Market output is an intent and still passes through build_market_orders().

- [ ] **Step 2: Implement the no-model fallback.**

LearnedPolicy(model_path=None).propose(state, features) returns an empty proposal. Missing, corrupt, incompatible, or slow models must cause deterministic fallback, never an exception from the agent entrypoint.

- [ ] **Step 3: Implement compile_proposal.**

The compiler must discard unknown workers, out-of-bounds/locked targets, invalid task kinds, and unavailable items; resolve duplicate workers by proposal score then worker index; preserve carried delivery assignments; route valid targets through worker_action()/route_to(); pass market intents through build_market_orders(); and use deterministic assignments for omitted workers.

- [ ] **Step 4: Write compiler tests.**

Test movement compilation, invalid and locked targets, duplicate resolution, carried-item preservation, legal market compilation, and complete deterministic fallback.

- [ ] **Step 5: Add a disabled-by-default seam.**

Add Policy(strategy="current", learned_model: str | None = None). With learned_model=None, existing deterministic tests and action outputs must remain unchanged. Do not change main.py.

- [ ] **Step 6: Test and commit.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests -q
git -C Kaggriculture add kagriculture_agent/learned_policy.py kagriculture_agent/policy.py tests/test_learned_policy.py tests/test_policy.py
git -C Kaggriculture commit -m "feat: add target-based learned policy compiler"
~~~

## Task 5: Build replay-to-trajectory collection

**Files:**
- Create: Kaggriculture/kagriculture_agent/trajectory.py
- Create: Kaggriculture/scripts/collect_trajectories.py
- Test: Kaggriculture/tests/test_trajectory.py
- Modify: Kaggriculture/scripts/run_local.py

- [ ] **Step 1: Define transitions.**

~~~python
TRANSITION_SCHEMA_VERSION = 1

@dataclass(frozen=True)
class Transition:
    observation: dict[str, Any]
    action: dict[str, Any]
    next_observation: dict[str, Any]
    done: bool
    reward: float
    final_bank: float | None
    opponent_final_bank: float | None
    safety_flags: tuple[str, ...]
~~~

Use zero rewards except on the final transition. Set terminal reward to tanh((candidate_bank - opponent_bank) / 1000.0). Never add the opponent’s private shed or inventory.

- [ ] **Step 2: Implement replay conversion.**

transitions_from_replay(replay, candidate_player) pairs each action with the preceding observation, uses the following observation only as next_observation, and rejects incomplete/malformed sequences. Reuse evaluator validation rather than duplicating engine rules.

- [ ] **Step 3: Add the collector CLI.**

Support:

~~~bash
Kaggriculture/.venv/bin/python Kaggriculture/scripts/collect_trajectories.py \
  --seeds 100 --start-seed 0 --steps 720 \
  --opponents current random starter --seats 0 1 \
  --output /private/tmp/kagriculture-trajectories.jsonl
~~~

Run games in isolated processes and write one validated transition per JSONL line plus a manifest with engine version, feature schema, seed matrix, seats, opponents, and source-policy identity.

- [ ] **Step 4: Write trajectory tests.**

Test terminal reward, candidate seat selection, private-state preservation, malformed replay rejection, deterministic JSON serialization, and one transition per engine step.

- [ ] **Step 5: Smoke-test and commit.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests/test_trajectory.py -q
Kaggriculture/.venv/bin/python Kaggriculture/scripts/collect_trajectories.py \
  --seeds 2 --start-seed 0 --steps 96 --opponents pass random --seats 0 1 \
  --output /private/tmp/kagriculture-trajectories-smoke.jsonl
git -C Kaggriculture add kagriculture_agent/trajectory.py scripts/collect_trajectories.py scripts/run_local.py tests/test_trajectory.py
git -C Kaggriculture commit -m "feat: collect validated Kaggriculture trajectories"
~~~

## Task 6: Implement the training-time model and PPO league

**Files:**
- Create: Kaggriculture/kagriculture_agent/model.py
- Create: Kaggriculture/scripts/train_policy.py
- Test: Kaggriculture/tests/test_model.py
- Test: Kaggriculture/tests/test_train_policy.py

- [ ] **Step 1: Define the small network.**

Use separate projections for tile, worker, market, and global tokens into width 128; four residual attention blocks with four heads and a 256-wide MLP. Add heads for worker act/idle, worker target tile, worker task kind, market item, market quantity, and scalar normalized final-bank-margin value.

The forward contract is:

~~~python
model(feature_batch) -> {
    "worker_act_logits": Tensor,
    "worker_target_logits": Tensor,
    "worker_kind_logits": Tensor,
    "market_item_logits": Tensor,
    "market_quantity_logits": Tensor,
    "value": Tensor,
}
~~~

- [ ] **Step 2: Add behavior-cloning smoke training.**

Train one epoch from deterministic-policy transitions and emit a checkpoint containing model version, feature schema version, action vocabulary, and engine version. This validates data and action shapes before long self-play.

- [ ] **Step 3: Add PPO configuration.**

Start with:

~~~python
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.20
value_coef = 0.50
entropy_coef = 0.01
target_kl = 0.03
rollout_steps = 64
~~~

Use terminal normalized bank-margin reward, advantage normalization, clipped policy/value/entropy losses, and KL/cross-entropy regularization against the previous-best checkpoint. Compare gamma 1.0 only as an explicit experiment because the article observed training-time stalling with undiscounted returns.

- [ ] **Step 4: Implement the opponent pool.**

Sample current at .40, mixed at .15, random at .10, starter at .10, and checkpoint at .25. For checkpoint opponents, sample uniformly from the previous five promoted checkpoints. Alternate seats 0/1. Replace the previous best only after a fixed match exceeds 70% win rate.

- [ ] **Step 5: Write training tests.**

Test output shapes, deterministic seeds, terminal reward, PPO ratio clipping, advantage normalization, checkpoint metadata, and opponent probabilities summing to one.

- [ ] **Step 6: Smoke-test and commit.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests/test_model.py Kaggriculture/tests/test_train_policy.py -q
Kaggriculture/.venv/bin/python Kaggriculture/scripts/train_policy.py \
  --input /private/tmp/kagriculture-trajectories-smoke.jsonl \
  --steps 2 --batch-size 8 --output /private/tmp/learned-v1-smoke.pt
git -C Kaggriculture add kagriculture_agent/model.py scripts/train_policy.py tests/test_model.py tests/test_train_policy.py
git -C Kaggriculture commit -m "feat: add compact PPO training pipeline"
~~~

## Task 7: Benchmark rollout throughput and add simulator parity only if needed

**Files:**
- Create: Kaggriculture/scripts/benchmark_rollouts.py
- Test: Kaggriculture/tests/test_benchmark_rollouts.py
- Modify: Kaggriculture/scripts/collect_trajectories.py
- Optional create: Kaggriculture/kagriculture_agent/simulator.py

- [ ] **Step 1: Benchmark the real engine.**

Report games/hour, environment steps/second, policy inference milliseconds/turn, and p95 latency:

~~~bash
Kaggriculture/.venv/bin/python Kaggriculture/scripts/benchmark_rollouts.py \
  --games 10 --steps 720 --workers 1 2 4 8
~~~

- [ ] **Step 2: Apply the throughput gate.**

Keep the real engine if four workers reach at least 100,000 environment steps/minute and inference is below 10 ms/turn. If either threshold fails, implement a fast simulator matching the public configuration, market, end-of-day transitions, worker legality, and terminal bank.

- [ ] **Step 3: Add parity tests for a simulator.**

Run identical recorded action sequences through the real engine and simulator for seeds 0–9. Assert equality of public state, candidate private state, market, worker positions, cash, inventory, and terminal status. Do not use the simulator for promotion until all ten seeds pass.

- [ ] **Step 4: Test and commit separately.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests/test_benchmark_rollouts.py -q
git -C Kaggriculture add scripts/benchmark_rollouts.py scripts/collect_trajectories.py tests/test_benchmark_rollouts.py
git -C Kaggriculture commit -m "perf: benchmark self-play rollout throughput"
~~~

If a simulator is required, commit it with parity tests in a separate commit from policy changes.

## Task 8: Export dependency-light quantized inference

**Files:**
- Create: Kaggriculture/scripts/export_policy.py
- Modify: Kaggriculture/kagriculture_agent/learned_policy.py
- Test: Kaggriculture/tests/test_export_policy.py
- Modify: Kaggriculture/tests/test_packaging.py
- Modify: Kaggriculture/.gitignore

- [ ] **Step 1: Define the artifact format.**

Export:

~~~json
{
  "format_version": 1,
  "model_version": "learned_v1",
  "feature_schema_version": 1,
  "engine_version": "1.32.7",
  "hidden_width": 128,
  "quantization": "int8-per-row",
  "weights": {}
}
~~~

Quantize linear weights per row with an fp32 scale and int8 values. Runtime loading must use only standard-library modules such as array and math.

- [ ] **Step 2: Validate artifacts and fallback.**

Reject unsupported formats, mismatched feature/engine versions, missing tensors, non-finite scales, or invalid action vocabularies. LearnedPolicy catches load errors and returns deterministic fallback.

- [ ] **Step 3: Test numerical agreement and latency.**

Compare pure-Python inference with the training model on a fixed fixture: at least 99% action-logit rank agreement and value difference below 1e-2. Require p95 inference below 50 ms over 1000 states on the development machine.

- [ ] **Step 4: Test packaging and commit.**

Assert that the submission archive contains only the runtime package and selected artifact, and excludes tests, reports, trajectories, training scripts, checkpoints, PyTorch, and NumPy.

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests/test_export_policy.py Kaggriculture/tests/test_packaging.py -q
git -C Kaggriculture add scripts/export_policy.py kagriculture_agent/learned_policy.py tests/test_export_policy.py tests/test_packaging.py .gitignore
git -C Kaggriculture commit -m "feat: export dependency-light learned inference"
~~~

## Task 9: Register and evaluate learned_v1 without promotion

**Files:**
- Modify: Kaggriculture/kagriculture_agent/candidates.py
- Modify: Kaggriculture/scripts/evaluate.py
- Test: Kaggriculture/tests/test_agent_smoke.py
- Test: Kaggriculture/tests/test_evaluate.py

- [ ] **Step 1: Register the candidate behind an artifact check.**

Add learned_v1 only when its artifact exists. If absent, candidate_policy("learned_v1") raises a clear ValueError; it must not silently alter production behavior.

- [ ] **Step 2: Include reproducibility metadata.**

Use the stable candidate namespace, preserve both-seat execution, and include model identity, artifact hash, feature schema, and engine version in the manifest. Run current on the exact same matrix as the learned candidate.

- [ ] **Step 3: Add smoke tests.**

Test import without training dependencies, exact action schema, corrupt-artifact fallback, and separation from legacy variant="mixed" semantics.

- [ ] **Step 4: Run development evaluation.**

~~~bash
Kaggriculture/.venv/bin/python Kaggriculture/scripts/evaluate.py \
  --seeds 30 --start-seed 0 --steps 720 \
  --opponents pass random starter --seats 0 1 \
  --candidates current learned_v1 \
  --output /private/tmp/kaggriculture-learned-development.json
~~~

Inspect safety reasons before comparing bank or win rate. Do not modify main.py from development results.

- [ ] **Step 5: Test and commit.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests/test_agent_smoke.py Kaggriculture/tests/test_evaluate.py -q
git -C Kaggriculture add kagriculture_agent/candidates.py scripts/evaluate.py tests/test_agent_smoke.py tests/test_evaluate.py
git -C Kaggriculture commit -m "feat: evaluate learned_v1 beside deterministic candidates"
~~~

## Task 10: Holdout promotion and production packaging

**Files:**
- Modify: Kaggriculture/main.py only after promotion
- Modify: Kaggriculture/README.md
- Test: Kaggriculture/tests/test_import.py
- Test: Kaggriculture/tests/test_packaging.py

- [ ] **Step 1: Run disjoint development and holdout matrices.**

~~~bash
Kaggriculture/.venv/bin/python Kaggriculture/scripts/evaluate.py \
  --seeds 30 --start-seed 0 \
  --holdout-seeds 100 101 102 103 104 105 106 107 108 109 \
  --steps 720 --min-valid-games 20 \
  --opponents pass random starter --seats 0 1 \
  --candidates current learned_v1 \
  --max-same-item-churn 0 --max-market-transactions 500 \
  --min-terminal-cash 100 \
  --output /private/tmp/kaggriculture-learned-holdout.json
~~~

- [ ] **Step 2: Apply the promotion gate.**

Promote only if learned_v1 has complete valid records for every requested pair, zero framework failures, zero missed basic-needs events, non-negative fifth-percentile paired bank differential, and strictly better seat-balanced win rate and median paired bank differential than current.

- [ ] **Step 3: Switch production only when selected.**

Only after the report explicitly selects learned_v1, change:

~~~python
_policy = Policy()
~~~

to:

~~~python
_policy = Policy(strategy="current", learned_model="models/learned_v1.json")
~~~

Keep deterministic fallback on model load or inference failure.

- [ ] **Step 4: Update documentation and archive contents.**

Document artifact hash, model/feature schema versions, holdout command, and fallback behavior. Package only main.py, kagriculture_agent/, and the selected model artifact.

- [ ] **Step 5: Verify and commit.**

~~~bash
Kaggriculture/.venv/bin/python -m pytest Kaggriculture/tests -q
tar --exclude='__pycache__' -czf /private/tmp/kaggriculture-submission.tar.gz \
  -C Kaggriculture main.py kagriculture_agent models/learned_v1.json
tar -tzf /private/tmp/kaggriculture-submission.tar.gz
Kaggriculture/.venv/bin/python -c "import sys; sys.path.insert(0, 'Kaggriculture'); from main import agent; print(agent({'step': 0}))"
~~~

Confirm the archive excludes tests, reports, trajectories, training code, checkpoints, PyTorch, and NumPy. Commit the production switch only after these checks pass.

## Self-review and stopping rules

- The deterministic policy remains usable after every task; no learned component is required for import until Task 9.
- The plan covers target-based actions, compact entity/global observations, self-play PPO, checkpoint leagues, throughput measurement, parity tests, and quantized inference.
- The plan does not copy Orbit Wars’ 200M-parameter/15B-step scale because Kaggriculture has a smaller state/action space and a one-second turn budget.
- No simulator is trusted without replay parity, and no learned candidate is promoted without safety and holdout gates.
- Generated reports, trajectories, checkpoints, and model artifacts remain outside source control unless explicitly requested.

