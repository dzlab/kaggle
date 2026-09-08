# Dependency-Free Artifact Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, reproducible CLI that compares a dependency-free artifact with the current policy on an identical two-seat game matrix.

**Architecture:** Keep all new orchestration in `scripts/evaluate_artifact.py`. Validate and hash the artifact in the parent, build ordered `(opponent, seed, seat)` requests, execute them in a capped `ProcessPoolExecutor`, normalize replay JSON with `scripts.evaluate.replay_record`, and apply `promotion_decision` with the explicit matrix. Inject a game runner in tests so no game is started.

**Tech Stack:** Python 3.11+, standard-library argparse/hashlib/json/os/tempfile/concurrent.futures, existing `run_local.run_episode`, existing evaluator gates, pytest.

---

### Task 1: Define the test-facing matrix, CLI, and artifact validation contract

**Files:**
- Create: `tests/test_evaluate_artifact.py`
- Create: `scripts/evaluate_artifact.py`

- [ ] **Step 1: Write failing tests for deterministic matrix construction and validation**

```python
def test_build_matrix_is_ordered_and_has_both_default_seats():
    from scripts.evaluate_artifact import build_matrix

    assert build_matrix(opponents=["starter", "pass"], seeds=[7, 8], seats=[0, 1]) == [
        {"opponent": "starter", "seed": 7, "seat": 0},
        {"opponent": "starter", "seed": 7, "seat": 1},
        {"opponent": "starter", "seed": 8, "seat": 0},
        {"opponent": "starter", "seed": 8, "seat": 1},
        {"opponent": "pass", "seed": 7, "seat": 0},
        {"opponent": "pass", "seed": 7, "seat": 1},
        {"opponent": "pass", "seed": 8, "seat": 0},
        {"opponent": "pass", "seed": 8, "seat": 1},
    ]

def test_validate_seats_requires_unique_zero_or_one_values():
    from scripts.evaluate_artifact import validate_seats
    assert validate_seats([0, 1]) == [0, 1]
    with pytest.raises(ValueError, match="unique"):
        validate_seats([0, 0])
    with pytest.raises(ValueError, match="0 or 1"):
        validate_seats([2])

def test_validate_artifact_returns_identity_and_sha256(tmp_path):
    from scripts.evaluate_artifact import validate_artifact
    artifact = tmp_path / "ppo16.json"
    artifact.write_text(json.dumps(_valid_learned_artifact()), encoding="utf-8")
    result = validate_artifact(artifact, "ppo16")
    assert result["identity"] == "ppo16"
    assert result["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
```

- [ ] **Step 2: Run the new tests to verify they fail for missing functions**

Run: `pytest -q tests/test_evaluate_artifact.py`

Expected: collection or test failures because `scripts.evaluate_artifact` and its functions do not yet exist.

- [ ] **Step 3: Implement validation and deterministic request construction**

Implement `validate_seats(values)`, `validate_opponents(values)`, and
`build_matrix(opponents, seeds, seats)`. `build_matrix` must iterate opponents,
then seeds, then seats and return dictionaries containing only string opponent,
integer seed, and integer seat. Implement `validate_artifact(path, identity)` by
resolving a regular file, requiring a non-empty string identity, calling
`load_exported_policy(path)`, and returning `{"path": str(path), "name": path.name,
"identity": identity, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}`.

Add `parse_args` with required `--artifact`, default identity `learned_artifact`,
`--seeds` default 30, `--start-seed` default 0, `--steps` default 720,
opponents defaulting to `pass random starter`, seats defaulting to `[0, 1]`,
workers defaulting to `min(2, 8)`, positive `--min-valid-games` default 20,
output default `reports/artifact-evaluation.json`, and `--quick`. Enforce
workers in `1..8`; when quick mode leaves defaults untouched, change seeds to 2
and steps to 96.

- [ ] **Step 4: Run the validation tests and confirm they pass**

Run: `pytest -q tests/test_evaluate_artifact.py -k 'matrix or seats or artifact'`

Expected: PASS.

### Task 2: Add mocked execution tests and the bounded evaluation implementation

**Files:**
- Modify: `tests/test_evaluate_artifact.py`
- Modify: `scripts/evaluate_artifact.py`

- [ ] **Step 1: Write a failing test for two identical candidate matrices**

```python
def test_evaluate_runs_current_and_artifact_on_identical_matrix(tmp_path):
    from scripts.evaluate_artifact import evaluate
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_valid_learned_artifact()), encoding="utf-8")
    calls = []

    def fake_game(request):
        calls.append(request)
        return _valid_record(request["candidate"], request["opponent"], request["seed"], request["seat"])

    result = evaluate(artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
                      steps=4, workers=1, min_valid_games=1, game_runner=fake_game)
    assert result["expected_matrix"] == [["pass", 3, 0], ["pass", 3, 1]]
    assert [(r["candidate"], r["opponent"], r["seed"], r["seat"]) for r in result["records"]] == [
        ("current", "pass", 3, 0), ("current", "pass", 3, 1),
        ("learned_artifact", "pass", 3, 0), ("learned_artifact", "pass", 3, 1),
    ]
```

- [ ] **Step 2: Run the matrix test to verify the orchestration is absent**

Run: `pytest -q tests/test_evaluate_artifact.py -k identical_matrix`

Expected: FAIL because `evaluate` is not implemented.

- [ ] **Step 3: Implement isolated game execution and bounded ordering**

Implement `_run_game(request)` as a top-level picklable function. It must use
`run_local.run_episode` with `candidate_identity="current"` for the current
candidate or `candidate_artifact=artifact_path` for the artifact candidate,
never pass the artifact as an opponent, and call `replay_record(env.toJSON(),
variant=candidate, opponent=..., seed=..., seat=...)`. Catch all execution and
normalization exceptions and return `_framework_error_record` with the request
coordinates.

Implement `evaluate(...)` to validate inputs, validate the artifact, construct
one expected matrix, create current requests followed by artifact requests, and
run them with `ProcessPoolExecutor(max_workers=workers).map(_run_game, requests)`
when no `game_runner` is injected. An injected runner is called directly in
request order for tests. Add the candidate label to every returned record and
return ordered records, expected matrix, artifact metadata, configuration,
summaries from `paired_seed_summary`, and a decision from
`promotion_decision(artifact_records, current_records, min_valid_games=...,
expected_matrix=...)`.

- [ ] **Step 4: Run the mocked orchestration test and focused suite**

Run: `pytest -q tests/test_evaluate_artifact.py`

Expected: PASS with no Kaggriculture game started.

### Task 3: Add fail-closed reporting, CLI wiring, and degraded-candidate coverage

**Files:**
- Modify: `tests/test_evaluate_artifact.py`
- Modify: `scripts/evaluate_artifact.py`

- [ ] **Step 1: Write failing tests for degraded-gate discard and atomic report shape**

```python
def test_degraded_candidate_is_discarded_by_existing_gates(tmp_path):
    from scripts.evaluate_artifact import evaluate
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_valid_learned_artifact()), encoding="utf-8")

    def fake_game(request):
        return _valid_record(request["candidate"], request["opponent"], request["seed"],
                             request["seat"], bank_differential=-10 if request["candidate"] != "current" else 10)

    result = evaluate(artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
                      steps=4, workers=1, min_valid_games=1, game_runner=fake_game)
    assert result["decision"]["status"] == "discard"
    assert "negative_tail" in result["decision"]["reasons"]

def test_incomplete_matrix_fails_closed(tmp_path):
    from scripts.evaluate_artifact import evaluate
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(_valid_learned_artifact()), encoding="utf-8")

    def fake_game(request):
        if request["candidate"] != "current" and request["seat"] == 1:
            return {"candidate": request["candidate"], "opponent": request["opponent"],
                    "seed": request["seed"], "seat": 0, "framework_error": False}
        return _valid_record(request["candidate"], request["opponent"], request["seed"], request["seat"])

    result = evaluate(artifact=artifact, seeds=[3], opponents=["pass"], seats=[0, 1],
                      steps=4, workers=1, min_valid_games=1, game_runner=fake_game)
    assert result["decision"]["status"] == "discard"
    assert result["decision"]["reasons"]
```

- [ ] **Step 2: Run the gate tests to verify they fail**

Run: `pytest -q tests/test_evaluate_artifact.py -k 'degraded or incomplete'`

Expected: FAIL because fail-closed report/decision handling is not complete.

- [ ] **Step 3: Implement report construction and atomic writing**

Implement `build_report(result, configuration)` with JSON-compatible
configuration, artifact metadata, expected matrix, per-game records grouped by
candidate, summaries, and decision. Always expose matrix completeness for both
candidate record sets. If either set is not exactly one record per expected
coordinate, force the final decision to `{"status": "discard", ...}` and add an
incomplete-matrix reason while retaining the existing gate result.

Implement `write_report(path, report)` using a sibling temporary file opened
with UTF-8 JSON and `os.replace`. `main(argv)` validates the artifact before
running, writes the report even for evaluation failures when possible, prints a
compact status line, and returns 0 only for a complete evaluation (the decision
may still be `discard` for a valid degraded artifact); invalid artifact,
framework-error, or incomplete-matrix outcomes return 1.

- [ ] **Step 4: Run focused tests and inspect the diff**

Run: `pytest -q tests/test_evaluate_artifact.py`

Expected: PASS; no test should invoke a long game.

Run: `git diff --check && git status --short`

Expected: no whitespace errors and only the new script, its test, and the approved plan remain as implementation changes.

- [ ] **Step 5: Commit the implementation**

```bash
git add scripts/evaluate_artifact.py tests/test_evaluate_artifact.py
git commit -m "feat: add bounded artifact evaluator"
```
