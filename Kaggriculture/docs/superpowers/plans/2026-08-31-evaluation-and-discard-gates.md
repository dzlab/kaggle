# Kaggriculture Evaluation and Discard Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox ( - [ ] ) syntax for tracking.

**Goal:** Build a fresh-process, both-seat, seed-holdout evaluation harness that reliably identifies illegal, fragile, and economically weak Kaggriculture strategies before submission.

**Architecture:** Keep the existing replay validator as the correctness layer, add an isolated worker process for each game, and make evaluation outputs a versioned manifest containing per-game records plus aggregate promotion decisions. Candidate strategies are evaluated against identical seeds and opponents in both seating orders; discard gates run before score comparisons.

**Tech Stack:** Python 3.11+, uv, pytest, kaggle-environments==1.32.7, JSON reports, subprocess isolation.

---

### Task 1: Define the evaluation manifest and candidate interface

**Files:**
- Create: Kaggriculture/kagriculture_agent/candidates.py
- Modify: Kaggriculture/scripts/evaluate.py
- Test: Kaggriculture/tests/test_evaluate.py

- [ ] Step 1: Write failing tests for candidate names and manifest fields

~~~python
def test_candidate_registry_exposes_stable_names():
    from kagriculture_agent.candidates import CANDIDATES, candidate_policy
    assert tuple(CANDIDATES) == ("current", "melon", "premium", "mixed")
    assert callable(candidate_policy("current"))

def test_manifest_contains_engine_and_reproducibility_fields():
    from scripts.evaluate import build_manifest
    manifest = build_manifest(
        candidates=["current"], opponents=["pass"], seeds=[3],
        steps=720, seats=[0, 1], command=["scripts/evaluate.py"],
    )
    assert manifest["schema_version"] == 2
    assert manifest["engine_version"] == "1.32.7"
    assert manifest["seeds"] == [3]
    assert manifest["seats"] == [0, 1]
    assert manifest["candidates"] == ["current"]

def test_unknown_candidate_is_rejected():
    from kagriculture_agent.candidates import candidate_policy
    with pytest.raises(ValueError, match="unsupported candidate"):
        candidate_policy("unknown")
~~~

- [ ] Step 2: Run the focused tests to verify they fail

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_evaluate.py -k "candidate_registry or manifest or unknown_candidate" -q

Expected: FAIL because the registry and manifest builder do not exist.

- [ ] Step 3: Add the stable candidate registry and manifest builder

~~~python
# Kaggriculture/kagriculture_agent/candidates.py
from collections.abc import Callable, Mapping
from typing import Any
from .policy import Policy

CANDIDATES = ("current", "melon", "premium", "mixed")

def candidate_policy(name: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    if name not in CANDIDATES:
        raise ValueError(f"unsupported candidate: {name}")
    return Policy(strategy=name).act
~~~

Add build_manifest beside the existing report builders. It returns only JSON-compatible values: schema_version, engine_version, steps, seeds, seats, opponents, candidates, Python version, and a normalized command. Never write absolute output paths into the report.

- [ ] Step 4: Run the focused tests to verify they pass

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_evaluate.py -k "candidate_registry or manifest or unknown_candidate" -q

Expected: PASS.

- [ ] Step 5: Commit the interface

~~~bash
git add Kaggriculture/kagriculture_agent/candidates.py Kaggriculture/scripts/evaluate.py Kaggriculture/tests/test_evaluate.py
git commit -m "test: define evaluation candidate manifest"
~~~

### Task 2: Run every game in a fresh interpreter and support both seats

**Files:**
- Create: Kaggriculture/scripts/evaluation_worker.py
- Modify: Kaggriculture/scripts/evaluate.py
- Modify: Kaggriculture/scripts/run_local.py
- Test: Kaggriculture/tests/test_evaluate.py

- [ ] Step 1: Write tests for seat order and worker serialization

~~~python
def test_worker_result_must_be_one_json_object():
    from scripts.evaluation_worker import decode_worker_result
    with pytest.raises(ValueError, match="worker result"):
        decode_worker_result("not-json\n")

def test_run_matrix_contains_both_seat_orders(monkeypatch):
    from scripts.evaluate import run_matrix
    calls = []
    def fake_run_game(**kwargs):
        calls.append(kwargs)
        return {"framework_error": False, "outcome": "tie", "final_bank": 0,
                "bank_differential": 0, "missed_basic_needs": 0, **kwargs}
    monkeypatch.setattr("scripts.evaluate.run_game", fake_run_game)
    run_matrix(candidates=["current"], opponents=["pass"], seeds=[4], steps=16, seats=[0, 1])
    assert [call["seat"] for call in calls] == [0, 1]
~~~

- [ ] Step 2: Run the tests to verify they fail

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_evaluate.py -k "worker_result or seat_orders" -q

Expected: FAIL because worker decoding and seat-aware execution are absent.

- [ ] Step 3: Implement the worker protocol and seat-aware execution

The worker reads one JSON object from stdin and prints exactly one JSON object to stdout. The payload contains candidate, opponent, seed, steps, and seat. It creates the engine with episodeSteps and seed, orders agents as candidate/opponent for seat 0 and opponent/candidate for seat 1, and returns a normalized record or structured framework failure.

~~~python
def decode_worker_result(output: str) -> dict[str, object]:
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("worker result must contain one JSON object")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("worker result is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("worker result must be an object")
    return value
~~~

Change run_game and run_matrix to accept candidate and seat, while preserving variant as a compatibility alias for existing fixtures. Pass candidate_player=seat into replay extraction. The parent invokes subprocess.run([sys.executable, "scripts/evaluation_worker.py"], input=payload, text=True, capture_output=True, timeout=120). A timeout or non-zero exit becomes framework_error=True with the error capped at 1000 characters.

- [ ] Step 4: Run focused and existing replay tests

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_evaluate.py -q

Expected: PASS.

- [ ] Step 5: Commit the isolated runner

~~~bash
git add Kaggriculture/scripts/evaluation_worker.py Kaggriculture/scripts/evaluate.py Kaggriculture/scripts/run_local.py Kaggriculture/tests/test_evaluate.py
git commit -m "feat: isolate both-seat evaluation games"
~~~

### Task 3: Add discard metrics and confidence-preserving aggregation

**Files:**
- Modify: Kaggriculture/scripts/evaluate.py
- Test: Kaggriculture/tests/test_evaluate.py

- [ ] Step 1: Write failing tests for hard gates and paired results

~~~python
def test_candidate_is_discarded_on_any_framework_failure():
    from scripts.evaluate import promotion_decision
    records = [{"candidate": "melon", "opponent": "pass", "seat": 0, "seed": 1,
                "outcome": "win", "final_bank": 100, "bank_differential": 10,
                "framework_error": True, "missed_basic_needs": 0}]
    decision = promotion_decision(records, baseline_records=[])
    assert decision["status"] == "discard"
    assert "framework_error" in decision["reasons"]

def test_paired_seed_summary_has_both_seats():
    from scripts.evaluate import paired_seed_summary
    records = [
        {"candidate": "melon", "opponent": "pass", "seat": 0, "seed": 1,
         "outcome": "win", "bank_differential": 10, "framework_error": False,
         "missed_basic_needs": 0},
        {"candidate": "melon", "opponent": "pass", "seat": 1, "seed": 1,
         "outcome": "loss", "bank_differential": -4, "framework_error": False,
         "missed_basic_needs": 0},
    ]
    summary = paired_seed_summary(records)
    assert summary["paired_games"] == 1
    assert summary["seat_balanced_win_rate"] == 0.5
    assert summary["mean_paired_bank_differential"] == 3
~~~

- [ ] Step 2: Run the tests to verify they fail

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest tests/test_evaluate.py -k "promotion_decision or paired_seed" -q

Expected: FAIL because the discard and paired-summary functions do not exist.

- [ ] Step 3: Implement deterministic metrics and gates

Add gates in this order: discard on any framework error; discard on any missed basic need; discard when either seat has fewer than the configured valid games; discard when the fifth-percentile bank differential is below zero; only then compare paired win rate and median bank differential against baseline. Pair records by candidate, opponent, and seed across seats. Missing pairs are invalid. Store Wilson win-rate bounds and deterministic seed-based bootstrap bank bounds in JSON.

~~~python
def promotion_decision(records, baseline_records, *, min_valid_games=20):
    if any(record.get("framework_error") for record in records):
        return {"status": "discard", "reasons": ["framework_error"]}
    if any(record.get("missed_basic_needs", 0) for record in records):
        return {"status": "discard", "reasons": ["missed_basic_needs"]}
    valid = [r for r in records if r.get("outcome") in {"win", "loss", "tie"}]
    if len(valid) < min_valid_games:
        return {"status": "discard", "reasons": ["insufficient_valid_games"]}
    candidate = paired_seed_summary(valid)
    baseline = paired_seed_summary(baseline_records)
    if candidate["fifth_percentile_bank_differential"] < 0:
        return {"status": "discard", "reasons": ["negative_tail"],
                "candidate": candidate, "baseline": baseline}
    improved = (
        candidate["seat_balanced_win_rate"] > baseline["seat_balanced_win_rate"]
        and candidate["mean_paired_bank_differential"] > baseline["mean_paired_bank_differential"]
    )
    return {"status": "promote" if improved else "discard",
            "reasons": [] if improved else ["no_paired_improvement"],
            "candidate": candidate, "baseline": baseline}
~~~

- [ ] Step 4: Run the full test suite

Run: UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest -q

Expected: PASS with 290 or more tests.

- [ ] Step 5: Commit the discard gates

~~~bash
git add Kaggriculture/scripts/evaluate.py Kaggriculture/tests/test_evaluate.py
git commit -m "feat: add strategy discard gates"
~~~

### Task 4: Add the reproducible evaluation CLI and report artifacts

**Files:**
- Modify: Kaggriculture/scripts/evaluate.py
- Modify: Kaggriculture/README.md
- Test: Kaggriculture/tests/test_evaluate.py

- [ ] Step 1: Add CLI parsing tests

~~~python
def test_cli_accepts_candidate_seat_and_holdout_options():
    from scripts.evaluate import parse_args
    args = parse_args([
        "--candidates", "current", "melon",
        "--opponents", "pass", "starter",
        "--seats", "0", "1",
        "--seeds", "4", "--holdout-seeds", "100", "101",
        "--min-valid-games", "8",
    ])
    assert args.candidates == ["current", "melon"]
    assert args.seats == [0, 1]
    assert args.holdout_seeds == [100, 101]
    assert args.min_valid_games == 8
~~~

- [ ] Step 2: Implement fixed development and holdout partitions

Support this command:

~~~bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run python scripts/evaluate.py \
  --candidates current melon premium mixed \
  --opponents pass random starter --seats 0 1 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --holdout-seeds 100 101 102 103 104 --steps 720 \
  --min-valid-games 20 --output reports/strategy-gate.json
~~~

The report includes engine version, the exact matrix, per-game records, aggregate summaries, paired summaries, discard reasons, confidence bounds, and selected candidate. A failed candidate can never be selected.

- [ ] Step 3: Document the promotion contract

Document that development and holdout seeds are never mixed; both seats are mandatory; framework errors and missed basic needs are hard failures; and full-season publication requires zero failures. Keep raw replays and credentials outside Git.

- [ ] Step 4: Run the full test suite and commit

~~~bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run pytest -q
git add Kaggriculture/scripts/evaluate.py Kaggriculture/README.md Kaggriculture/tests/test_evaluate.py
git commit -m "docs: define reproducible strategy evaluation gate"
~~~

Expected: all tests pass and report metadata is byte-stable apart from intentional result content.

### Task 5: Establish the baseline discard floor

**Files:**
- Create: Kaggriculture/reports/README.md
- Create: Kaggriculture/reports/baseline-gate.json
- Do not commit: raw replay files or replay sidecars.

- [ ] Step 1: Run the short smoke gate

~~~bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run python scripts/evaluate.py \
  --candidates current --opponents pass random starter --seats 0 1 \
  --seeds 0 1 --steps 96 --min-valid-games 1 \
  --output /private/tmp/kagriculture-smoke.json
~~~

Expected: exit zero and schema-valid records; this is only an execution smoke test.

- [ ] Step 2: Run the full baseline gate

Run the Task 4 matrix with current as the sole candidate. Retain the report only after checking it contains no credentials, raw replays, or absolute paths.

- [ ] Step 3: Inspect discard reasons before comparing money

~~~bash
UV_CACHE_DIR=/private/tmp/kagriculture-uv-cache uv run python -c "import json; p=json.load(open('reports/baseline-gate.json')); print(json.dumps({'selected': p['selected_candidate'], 'decisions': p['decisions']}, sort_keys=True))"
~~~

Expected: every decision is explicit; a failed baseline remains discard rather than being silently selected.

- [ ] Step 4: Commit only the report index and baseline summary

~~~bash
git add Kaggriculture/reports/README.md Kaggriculture/reports/baseline-gate.json
git commit -m "test: record baseline strategy gate"
~~~
