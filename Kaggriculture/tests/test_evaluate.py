import json
import io
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from kaggle_environments import make
except ModuleNotFoundError:
    make = None


def _engine_envelope(replay, seed=1, *, legacy_compact_fixture=True):
    from kagriculture_agent.constants import ENGINE_VERSION

    replay.update({
        "id": "test-replay",
        "name": "kaggriculture",
        "version": "0.1.0",
        "module_version": ENGINE_VERSION,
        "schema_version": 1,
        "title": "Kaggriculture",
        "description": "test",
        "metadata": {"legacy_compact_fixture": legacy_compact_fixture},
        "info": {"seed": seed},
        "configuration": {
            "episodeSteps": 720, "seed": None, "actTimeout": 1, "runTimeout": 1200,
            "boardSize": 10, "startingMoney": 3000, "maxMarketOrdersPerTurn": 10,
            "turnsPerDay": 24, "shedCapacity": 100, "weedSpawnChance": 0.005,
            "townShopUnlockInterval": 3, "townShopSellInterval": 4,
            "townCenterSellInterval": 24, "farmHandCostMult": 1, "marketParams": {},
        },
        "specification": {"action": {}, "agents": [2], "configuration": {}},
    })
    return replay


def _artifact_checksum(value):
    payload = {key: item for key, item in value.items() if key != "checksum"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _valid_learned_artifact():
    from kagriculture_agent import learned_policy as runtime

    weights = {}
    for name, shape in runtime.artifact_tensor_shapes().items():
        if len(shape) == 1:
            weights[name] = {"shape": list(shape), "values": [0.0] * shape[0]}
        else:
            weights[name] = {
                "shape": list(shape),
                "scales": [1.0] * shape[0],
                "values": [[0] * shape[1] for _row in range(shape[0])],
            }
    artifact = {
        "format_version": 1,
        "model_version": "learned_v1",
        "feature_schema_version": 1,
        "engine_version": "1.32.7",
        "hidden_width": 128,
        "quantization": "int8-per-row",
        "action_vocab": {
            "worker_kinds": list(runtime._ARTIFACT_WORKER_KINDS),
            "market_items": sorted(runtime.PRODUCTS),
            "market_quantities": list(runtime._ARTIFACT_MARKET_QUANTITIES),
        },
        "weights": weights,
    }
    artifact["checksum"] = _artifact_checksum(artifact)
    return artifact


def _reload_candidate_modules():
    import kagriculture_agent.candidates as candidates
    import scripts.evaluate as evaluate

    importlib.reload(candidates)
    return importlib.reload(evaluate)


def _strict_two_turn_replay():
    """Return a complete two-turn envelope for integrity-only regressions."""
    tiles = [[None for _ in range(4)] for _ in range(4)]
    p0_farm = {
        "money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
        "tiles": tiles, "unlocked_quadrants": ["NW"],
    }
    p1_farm = {
        "money": 90, "farmer": [0, 0], "hands": [], "hires_today": 0,
        "tiles": [[None for _ in range(4)] for _ in range(4)], "unlocked_quadrants": ["NW"],
    }
    market = {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}}

    def observation(player, step, hour):
        farms = [p0_farm] if player == 0 else [{"money": 100}, p1_farm]
        return {
            "player": player, "step": step, "day": 0, "hour": hour,
            "farms": farms,
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": market,
        }

    def state(player, step, hour, status):
        return {
            "observation": observation(player, step, hour),
            "action": {"farmer": ["PASS"], "hands": [], "market": []},
            "status": status, "info": {},
        }

    replay = {
        "steps": [[state(0, 0, 0, "ACTIVE"), state(1, 0, 0, "ACTIVE")],
                  [state(0, 1, 1, "DONE"), state(1, 1, 1, "DONE")]],
        "rewards": [100, 90], "statuses": ["DONE", "DONE"], "info": {},
    }
    replay = _engine_envelope(replay, legacy_compact_fixture=False)
    replay["configuration"]["episodeSteps"] = 2
    replay["configuration"]["boardSize"] = 4
    replay["specification"]["action"] = {"type": "object"}
    return replay


def _full_season_replay_with_terminal_shed(item):
    """Return a strict full-season replay with one persistent shed residue."""
    from kagriculture_agent.constants import season_days

    episode_steps = season_days * 24
    farm = {
        "money": 100, "farmer": [1, 1], "hands": [], "hires_today": 0,
        "tiles": [[None for _ in range(4)] for _ in range(4)],
        "unlocked_quadrants": ["NW"],
    }

    def observation(player, step):
        farms = [farm] if player == 0 else [{"money": 100}, farm]
        return {
            "player": player, "step": step, "day": step // 24, "hour": step % 24,
            "farms": farms,
            "private": {"seeds": {}, "shed": {item: 1}, "inventories": [{}]},
            "market": {"prices": {}, "inventory": {}},
        }

    steps = []
    for step in range(episode_steps):
        status = "DONE" if step == episode_steps - 1 else "ACTIVE"
        steps.append([
            {"observation": observation(0, step), "action": {"farmer": ["PASS"], "hands": [], "market": []},
             "status": status, "info": {}},
            {"observation": observation(1, step), "action": {"farmer": ["PASS"], "hands": [], "market": []},
             "status": status, "info": {}},
        ])

    replay = _engine_envelope({
        "steps": steps, "rewards": [100, 100], "statuses": ["DONE", "DONE"], "info": {},
    }, legacy_compact_fixture=False)
    replay["configuration"]["episodeSteps"] = episode_steps
    replay["configuration"]["boardSize"] = 4
    replay["specification"]["action"] = {"type": "object"}
    return replay


def test_cli_parses_seed_opponent_variant_and_quick_options():
    from scripts.evaluate import parse_args

    args = parse_args([
        "--seeds", "3",
        "--start-seed", "40",
        "--opponents", "pass", "starter",
        "--steps", "12",
        "--output", "results.json",
        "--variants", "mixed", "melon-heavy",
        "--quick",
    ])

    assert args.seeds == 3
    assert args.start_seed == 40
    assert args.opponents == ["pass", "starter"]
    assert args.steps == 12
    assert args.output == Path("results.json")
    assert args.variants == ["mixed", "melon-heavy"]
    assert args.quick is True


def test_cli_default_output_is_under_reports():
    from scripts.evaluate import parse_args

    assert parse_args([]).output == Path("reports/evaluation.json")
    assert parse_args([]).seats == [0, 1]


def test_cli_parses_requested_seats():
    from scripts.evaluate import parse_args

    assert parse_args(["--seats", "0", "1"]).seats == [0, 1]


def test_cli_accepts_explicit_holdout_seeds_and_minimum_valid_games():
    from scripts.evaluate import parse_args

    args = parse_args([
        "--candidates", "current", "melon",
        "--seeds", "4", "--start-seed", "10",
        "--holdout-seeds", "100", "101",
        "--min-valid-games", "8",
    ])

    assert args.holdout_seeds == [100, 101]
    assert args.min_valid_games == 8


def test_cli_rejects_overlapping_development_and_holdout_seeds():
    from scripts.evaluate import parse_args

    with pytest.raises(SystemExit):
        parse_args([
            "--seeds", "3", "--start-seed", "10",
            "--holdout-seeds", "12", "20",
        ])


def test_build_manifest_is_schema_v3_json_compatible_and_normalized():
    from scripts.evaluate import build_manifest

    manifest = build_manifest(
        candidates=["current"], opponents=["pass"], seeds=[3],
        steps=720, seats=[0, 1],
        command=["/tmp/project/scripts/evaluate.py", "--output", "/tmp/project/report.json"],
    )

    assert manifest == {
        "schema_version": 3,
        "engine_version": "1.32.7",
        "steps": 720,
        "seeds": [3],
        "seats": [0, 1],
        "opponents": ["pass"],
        "candidates": ["current"],
        "candidate_models": {
            "current": {
                "model_identity": "deterministic:current",
                "artifact_sha256": None,
                "feature_schema_version": 1,
                "engine_version": "1.32.7",
            },
        },
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "command": ["scripts/evaluate.py", "--output", "<report>"],
    }
    json.dumps(manifest, allow_nan=False)


def test_build_manifest_carries_experiment_identity():
    from scripts.evaluate import build_manifest

    manifest = build_manifest(
        candidates=["current"], opponents=["pass"], seeds=[3], steps=720,
        seats=[0, 1], experiment_id="orbit-context-test",
        feature_variant="experimental_context_v1",
        training_mode="reduced_behavior_clone_then_ppo",
    )

    assert {
        key: manifest[key]
        for key in ("experiment_id", "feature_variant", "training_mode")
    } == {
        "experiment_id": "orbit-context-test",
        "feature_variant": "experimental_context_v1",
        "training_mode": "reduced_behavior_clone_then_ppo",
    }


def test_build_manifest_includes_learned_v1_artifact_metadata(monkeypatch, tmp_path):
    artifact_path = tmp_path / "learned_v1.json"
    artifact_path.write_text(json.dumps(_valid_learned_artifact()), encoding="utf-8")
    with monkeypatch.context() as local:
        local.setenv("KAGRICULTURE_LEARNED_V1_ARTIFACT", str(artifact_path))
        evaluate = _reload_candidate_modules()
        manifest = evaluate.build_manifest(
            candidates=["current", "learned_v1"], opponents=["pass"], seeds=[3],
            steps=720, seats=[0, 1], command=["scripts/evaluate.py"],
        )

        assert manifest["candidates"] == ["current", "learned_v1"]
        assert manifest["candidate_models"]["learned_v1"] == {
            "model_identity": "learned_v1",
            "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "feature_schema_version": 1,
            "engine_version": "1.32.7",
        }
        assert manifest["candidate_models"]["current"]["artifact_sha256"] is None

    _reload_candidate_modules()


def test_build_manifest_names_legacy_and_stable_mixed_paths_separately():
    from scripts.evaluate import build_manifest

    common = {
        "candidates": ["mixed"], "opponents": ["pass"], "seeds": [3],
        "steps": 720, "seats": [0, 1], "command": ["scripts/evaluate.py"],
    }
    legacy = build_manifest(**common, selection_path="legacy_variant")
    stable = build_manifest(**common, selection_path="candidate")

    assert legacy["candidate_models"]["mixed"]["model_identity"] == "legacy_variant:mixed"
    assert stable["candidate_models"]["mixed"]["model_identity"] == "deterministic:mixed"
    assert legacy != stable
    assert legacy == build_manifest(**common, selection_path="legacy_variant")
    assert stable == build_manifest(**common, selection_path="candidate")


def test_result_manifest_uses_actual_legacy_or_candidate_selection_path():
    from scripts.evaluate import build_result_document

    common = {"opponents": ["pass"], "min_valid_games": 1}
    legacy = build_result_document(
        config={**common, "variants": ["mixed"]}, records=[]
    )
    stable = build_result_document(
        config={**common, "candidates": ["mixed"]}, records=[]
    )

    legacy_manifest = legacy["metadata"]["manifest"]
    stable_manifest = stable["metadata"]["manifest"]
    assert legacy_manifest["candidate_models"]["mixed"]["model_identity"] == "legacy_variant:mixed"
    assert stable_manifest["candidate_models"]["mixed"]["model_identity"] == "deterministic:mixed"
    assert legacy_manifest != stable_manifest
    assert legacy_manifest == build_result_document(
        config={**common, "variants": ["mixed"]}, records=[]
    )["metadata"]["manifest"]
    assert stable_manifest == build_result_document(
        config={**common, "candidates": ["mixed"]}, records=[]
    )["metadata"]["manifest"]


def test_run_matrix_uses_identical_matrix_for_current_and_learned_candidate(monkeypatch, tmp_path):
    artifact_path = tmp_path / "learned_v1.json"
    artifact_path.write_text(json.dumps(_valid_learned_artifact()), encoding="utf-8")
    captured = []

    def fake_run_game(**kwargs):
        captured.append(kwargs)
        candidate = kwargs["candidate"]
        return {
            **_complete_worker_record(),
            "candidate": candidate,
            "variant": candidate,
            "opponent": kwargs["opponent"],
            "seed": kwargs["seed"],
            "seat": kwargs["seat"],
        }

    with monkeypatch.context() as local:
        local.setenv("KAGRICULTURE_LEARNED_V1_ARTIFACT", str(artifact_path))
        evaluate = _reload_candidate_modules()
        local.setattr(evaluate, "run_game", fake_run_game)

        evaluate.run_matrix(
            candidates=["current", "learned_v1"], opponents=["pass", "random"],
            seeds=[1, 2], steps=96, seats=[0, 1],
        )

    matrices = {}
    for request in captured:
        matrices.setdefault(request["candidate"], []).append(
            (request["opponent"], request["seed"], request["seat"])
        )
    assert matrices["current"] == matrices["learned_v1"]
    assert matrices["current"] == [
        (opponent, seed, seat)
        for opponent in ("pass", "random") for seed in (1, 2) for seat in (0, 1)
    ]

    _reload_candidate_modules()


def test_cli_preserves_legacy_variants_as_a_separate_selection_mode():
    from scripts.evaluate import parse_args

    args = parse_args(["--variants", "mixed", "animal-heavy"])

    assert args.variants == ["mixed", "animal-heavy"]
    assert args.candidates is None


def test_cli_accepts_stable_route_candidates():
    from scripts.evaluate import parse_args

    args = parse_args(["--candidates", "current", "melon", "premium", "mixed"])

    assert args.candidates == ["current", "melon", "premium", "mixed"]
    assert args.variants is None


def test_cli_rejects_legacy_variant_names_on_stable_candidate_path():
    from scripts.evaluate import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--candidates", "animal-heavy"])


def test_variant_policy_constructs_real_route_candidate_policy():
    from scripts.evaluate import VariantPolicy

    candidate = VariantPolicy("premium")

    assert candidate.policy.strategy_name == "premium"


def test_variant_policy_requires_explicit_route_mode_for_mixed():
    from scripts.evaluate import VariantPolicy

    legacy = VariantPolicy("mixed")
    stable = VariantPolicy("mixed", route_candidate=True)

    assert legacy.is_route_candidate is False
    assert legacy.policy.strategy_name == "current"
    assert stable.is_route_candidate is True
    assert stable.policy.strategy_name == "mixed"


def test_variant_policy_applies_ablations_and_sanitization_to_route_candidates():
    from scripts.evaluate import VariantPolicy

    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}},
        "market": {"prices": {}, "inventory": {}},
    }
    route_policy = lambda _observation: {
        "farmer": ["EAST"], "hands": [], "market": [["BUY_LAND"]],
    }
    candidate = VariantPolicy(
        "mixed",
        ablations={
            "route_scheduling": False, "market_batch_sizing": True,
            "shop_adaptation": True, "land_purchase": False, "animals": True,
        },
        route_candidate=True,
        route_policy=route_policy,
    )

    assert candidate(observation) == {"farmer": ["PASS"], "hands": [], "market": []}


def test_worker_uses_candidate_factory_for_route_candidate(monkeypatch):
    import scripts.evaluation_worker as worker

    calls = []

    class FakeEnvironment:
        configuration = {}

        def run(self, agents):
            calls.append(agents)

        def toJSON(self):
            return {}

    monkeypatch.setitem(
        sys.modules,
        "kaggle_environments",
        type("FakeKaggleEnvironments", (), {"make": lambda *args, **kwargs: FakeEnvironment()})(),
    )
    monkeypatch.setattr(worker, "candidate_policy", lambda name: calls.append(name) or (lambda obs: {}))
    monkeypatch.setattr(
        worker,
        "replay_record",
        lambda *args, **kwargs: {
            "candidate": "melon", "variant": "melon", "opponent": "pass", "seed": 1,
            "seat": 0, "outcome": "tie", "final_bank": 0, "opponent_final_bank": 0,
            "bank_differential": 0, "framework_error": False, "shed_overflow": 0,
            "price_floor_sales": 0, "missed_basic_needs": 0,
        },
    )

    result = worker.run_request({"candidate": "melon", "opponent": "pass", "seed": 1, "steps": 2})

    assert result["candidate"] == "melon"
    assert calls[0] == "melon"


def test_worker_keeps_variant_mixed_on_legacy_policy_path(monkeypatch):
    import scripts.evaluation_worker as worker

    captured = {}

    class FakeEnvironment:
        configuration = {}

        def run(self, agents):
            captured["agents"] = agents

        def toJSON(self):
            return {}

    monkeypatch.setitem(
        sys.modules,
        "kaggle_environments",
        type("FakeKaggleEnvironments", (), {"make": staticmethod(lambda *args, **kwargs: FakeEnvironment())})(),
    )

    class FakeVariantPolicy:
        def __init__(self, variant, ablations, configuration, route_candidate, route_policy):
            captured["variant"] = variant
            captured["route_candidate"] = route_candidate
            captured["route_policy"] = route_policy

        def __call__(self, observation, configuration=None):
            return {"farmer": ["PASS"], "hands": [], "market": []}

    monkeypatch.setattr(worker, "VariantPolicy", FakeVariantPolicy)
    monkeypatch.setattr(
        worker,
        "replay_record",
        lambda *args, **kwargs: {
            "candidate": "mixed", "variant": "mixed", "opponent": "pass", "seed": 1,
            "seat": 0, "outcome": "tie", "final_bank": 0, "opponent_final_bank": 0,
            "bank_differential": 0, "framework_error": False, "shed_overflow": 0,
            "price_floor_sales": 0, "missed_basic_needs": 0,
        },
    )

    result = worker.run_request({"variant": "mixed", "opponent": "pass", "seed": 1, "steps": 2})

    assert result["variant"] == "mixed"
    assert captured["variant"] == "mixed"
    assert captured["route_candidate"] is False
    assert captured["route_policy"] is None


def test_cli_rejects_conflicting_variant_and_candidate_aliases():
    from scripts.evaluate import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--variants", "mixed", "--candidates", "animal-heavy"])


def test_cli_preserves_single_variant_behavior():
    from scripts.evaluate import parse_args

    args = parse_args(["--variant", "animal-heavy"])

    assert args.variants == ["animal-heavy"]
    assert args.candidates is None


@pytest.mark.parametrize("path", [
    ("steps", 0, 0, "observation", "player", True),
    ("rewards", 0, None, None, None, True),
    ("steps", 0, 0, "observation", "farms", None),
])
def test_replay_record_rejects_boolean_numeric_values(path):
    replay = _strict_two_turn_replay()
    if path[0] == "rewards":
        replay["rewards"][path[1]] = path[-1]
    elif path[4] == "farms":
        replay["steps"][path[1]][path[2]][path[3]][path[4]][0]["money"] = True
    else:
        replay[path[0]][path[1]][path[2]][path[3]][path[4]] = path[-1]

    from scripts.evaluate import replay_record

    result = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert result["framework_error"] is True


def test_replay_record_rejects_boolean_quantities():
    replay = _strict_two_turn_replay()
    replay["steps"][0][0]["observation"]["private"]["shed"] = {"WHEAT": True}

    from scripts.evaluate import replay_record

    result = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert result["framework_error"] is True


@pytest.mark.parametrize("mutate", [
    lambda replay: replay["steps"][0][0]["observation"]["farms"][0].__setitem__("money", "100"),
    lambda replay: replay["steps"][0][0]["observation"].__setitem__("step", "0"),
    lambda replay: replay["steps"][0][0]["observation"]["farms"][0]["farmer"].__setitem__(0, "0"),
    lambda replay: replay["steps"][0][0]["observation"]["private"]["shed"].__setitem__("WHEAT", "1"),
    lambda replay: replay["rewards"].__setitem__(0, "100"),
])
def test_nonlegacy_replay_rejects_numeric_strings_in_strict_fields(mutate):
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    mutate(replay)

    assert replay_record(replay, variant="mixed", opponent="pass", seed=1)["framework_error"] is True


def test_run_evaluation_rejects_conflicting_variant_aliases(monkeypatch):
    from scripts.evaluate import run_evaluation

    monkeypatch.setattr("scripts.evaluate.run_matrix", lambda **kwargs: pytest.fail("must not run"))

    with pytest.raises(ValueError, match="variants and candidates are separate modes"):
        run_evaluation(
            variants=["mixed"], candidates=["animal-heavy"],
            opponents=["pass"], seeds=[1], steps=2,
        )


def test_worker_accepts_direct_candidate_request(monkeypatch):
    import scripts.evaluation_worker as worker

    captured = {}

    class FakeEnvironment:
        configuration = {}

        def run(self, agents):
            captured["agents"] = agents

        def toJSON(self):
            return {}

    monkeypatch.setitem(
        sys.modules,
        "kaggle_environments",
        type("FakeKaggleEnvironments", (), {"make": staticmethod(lambda *args, **kwargs: FakeEnvironment())}),
    )
    monkeypatch.setattr(worker, "VariantPolicy", lambda variant, *args: captured.setdefault("variant", variant))
    def fake_replay_record(replay, **kwargs):
        captured["replay_kwargs"] = kwargs
        return {
            "candidate": kwargs["variant"], "variant": kwargs["variant"], "opponent": kwargs["opponent"],
            "seed": kwargs["seed"], "seat": kwargs["seat"], "outcome": "tie",
            "final_bank": 0.0, "opponent_final_bank": 0.0, "bank_differential": 0.0,
            "framework_error": False, "shed_overflow": 0.0, "price_floor_sales": 0.0,
            "missed_basic_needs": 0.0,
        }
    monkeypatch.setattr(worker, "replay_record", fake_replay_record)

    result = worker.run_request({
        "candidate": "mixed", "opponent": "pass", "seed": 1, "steps": 2, "seat": 1,
        "churn_window": 1,
    })

    assert captured["variant"] == "mixed"
    assert captured["replay_kwargs"]["variant"] == "mixed"
    assert captured["replay_kwargs"]["churn_window"] == 1
    assert result["candidate"] == "mixed"


def _metric_record(*, seat, seed, outcome, differential, candidate="melon", opponent="pass",
                   framework_error=False, missed_basic_needs=0):
    return {
        "candidate": candidate, "variant": candidate, "opponent": opponent,
        "seat": seat, "seed": seed, "outcome": outcome,
        "final_bank": 100.0 + differential, "opponent_final_bank": 100.0,
        "bank_differential": differential, "framework_error": framework_error,
        "shed_overflow": 0.0, "price_floor_sales": 0.0,
        "missed_basic_needs": missed_basic_needs,
    }


def test_paired_seed_summary_has_confidence_metrics_and_both_seats():
    from scripts.evaluate import paired_seed_summary

    records = [
        _metric_record(seat=0, seed=1, outcome="win", differential=10),
        _metric_record(seat=1, seed=1, outcome="loss", differential=-4),
        _metric_record(seat=0, seed=2, outcome="tie", differential=2),
    ]

    summary = paired_seed_summary(records)

    assert summary["paired_games"] == 1
    assert summary["missing_seat_pairs"] == 1
    assert summary["seat_balanced_win_rate"] == 0.5
    assert summary["mean_paired_bank_differential"] == 3.0
    assert summary["wilson_win_rate"] is None
    assert 0.0 <= summary["bootstrap_seat_balanced_win_rate"]["lower"] <= summary["bootstrap_seat_balanced_win_rate"]["upper"] <= 1.0
    assert set(summary["bootstrap_bank_differential"].keys()) == {"lower", "upper"}
    json.dumps(summary, allow_nan=False)


def test_paired_seed_summary_exposes_canonical_candidate_diagnostics():
    from scripts.evaluate import paired_seed_summary

    records = [
        {
            **_metric_record(seat=seat, seed=1, outcome="win", differential=10),
            "termination_reason": "resolved" if seat == 0 else "terminal",
            "bootstrap_truncated": seat == 0,
            "no_progress_steps": 4 if seat == 0 else 0,
            "time_limit_ending": False,
            "safety_flags": ["safety_regression"] if seat == 0 else [],
            "safety_regression": seat == 0,
        }
        for seat in (0, 1)
    ]

    diagnostics = paired_seed_summary(records)["diagnostics"]

    assert diagnostics["truncation_count"] == 1
    assert diagnostics["resolved_count"] == 1
    assert diagnostics["max_no_progress_streak"] == 4
    assert diagnostics["safety_regression_count"] == 1


def test_replay_record_reads_top_level_stall_metadata():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    replay.update({
        "termination_reason": "resolved",
        "bootstrap_truncated": True,
        "no_progress_steps": 6,
        "time_limit_ending": True,
        "safety_flags": ["safety_regression"],
    })
    replay["steps"][-1][0].update({
        "termination_reason": "no_progress",
        "bootstrap_truncated": True,
        "no_progress_steps": 8,
        "time_limit_ending": True,
        "safety_flags": ["review_flag"],
    })

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["termination_reason"] == "no_progress"
    assert record["bootstrap_truncated"] is True
    assert record["no_progress_steps"] == 8
    assert record["time_limit_ending"] is True
    assert record["safety_flags"] == ["safety_regression", "review_flag"]
    assert record["safety_regression"] is True


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("replay", "termination_reason", 17),
        ("info", "no_progress_steps", 1.5),
        ("state", "safety_flags", ["ok", 3]),
        ("state", "bootstrap_truncated", "true"),
    ],
)
def test_replay_record_rejects_malformed_diagnostic_metadata(location, field, value):
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    if location == "replay":
        replay[field] = value
    elif location == "info":
        replay["info"][field] = value
    else:
        replay["steps"][-1][0][field] = value

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"
    assert "malformed_replay" in record["framework_error_reasons"]


def test_paired_seed_summary_confidence_interval_uses_paired_seeds():
    from scripts.evaluate import _wilson_interval, paired_seed_summary

    records = [
        _metric_record(seat=seat, seed=seed, outcome=outcome, differential=1)
        for seed, outcome in ((1, "win"), (2, "loss"))
        for seat in (0, 1)
    ]

    summary = paired_seed_summary(records)

    assert summary["wilson_win_rate"] == _wilson_interval(1.0, 2)


def test_bootstrap_confidence_intervals_use_approximately_95_percent_coverage():
    from scripts.evaluate import paired_seed_summary

    records = [
        _metric_record(seat=seat, seed=seed, outcome=outcome, differential=differential)
        for seed, outcome, differential in ((1, "win", -10), (2, "loss", 10), (3, "win", 30))
        for seat in (0, 1)
    ]

    summary = paired_seed_summary(records)

    assert summary["bootstrap_seat_balanced_win_rate"]["lower"] >= 0.0
    assert summary["bootstrap_seat_balanced_win_rate"]["upper"] <= 1.0
    assert summary["bootstrap_bank_differential"]["lower"] <= summary["fifth_percentile_bank_differential"]


def test_nonlegacy_replay_allows_numeric_strings_in_opaque_metadata():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    replay["metadata"]["run_id"] = "2026"

    assert replay_record(replay, variant="mixed", opponent="pass", seed=1)["framework_error"] is False


def test_paired_seed_summary_reports_duplicate_pairs_separately():
    from scripts.evaluate import paired_seed_summary, promotion_decision

    records = [
        _metric_record(seat=0, seed=1, outcome="win", differential=10),
        _metric_record(seat=0, seed=1, outcome="win", differential=11),
        _metric_record(seat=1, seed=1, outcome="win", differential=10),
    ]

    summary = paired_seed_summary(records)

    assert summary["missing_seat_pairs"] == 0
    assert summary["duplicate_seat_pairs"] == 1
    assert promotion_decision(records, [], min_valid_games=1)["reasons"] == [
        "duplicate_seat_pairs"
    ]


def test_promotion_decision_applies_discard_gates_in_order():
    from scripts.evaluate import promotion_decision

    framework = [_metric_record(seat=0, seed=1, outcome="win", differential=10, framework_error=True)]
    assert promotion_decision(framework, [], min_valid_games=1)["reasons"] == ["framework_error"]

    needs = [_metric_record(seat=0, seed=1, outcome="win", differential=10, missed_basic_needs=1)]
    assert promotion_decision(needs, [], min_valid_games=1)["reasons"] == ["missed_basic_needs"]

    one_seat = [_metric_record(seat=0, seed=1, outcome="win", differential=10)]
    assert promotion_decision(one_seat, [], min_valid_games=2)["reasons"] == ["insufficient_valid_games"]

    missing_pair = [
        _metric_record(seat=0, seed=1, outcome="win", differential=10),
        _metric_record(seat=1, seed=1, outcome="loss", differential=-10),
        _metric_record(seat=0, seed=2, outcome="win", differential=10),
    ]
    assert promotion_decision(missing_pair, [], min_valid_games=1)["reasons"] == ["missing_seat_pairs"]

    negative_tail = [
        _metric_record(seat=0, seed=1, outcome="win", differential=-10),
        _metric_record(seat=1, seed=1, outcome="loss", differential=-10),
    ]
    assert promotion_decision(negative_tail, [], min_valid_games=1)["reasons"] == ["negative_tail"]


def test_promotion_decision_compares_baseline_only_after_gates():
    from scripts.evaluate import promotion_decision

    candidate = [
        _metric_record(seat=0, seed=1, outcome="win", differential=10),
        _metric_record(seat=1, seed=1, outcome="win", differential=10),
    ]
    baseline = [
        _metric_record(seat=0, seed=1, outcome="loss", differential=1, candidate="current"),
        _metric_record(seat=1, seed=1, outcome="loss", differential=1, candidate="current"),
    ]

    decision = promotion_decision(candidate, baseline, min_valid_games=1)

    assert decision["status"] == "promote"
    assert decision["reasons"] == []


def test_promotion_decision_rejects_diagnostic_regression_and_reports_deltas():
    from scripts.evaluate import promotion_decision

    candidate = [
        {
            **_metric_record(seat=seat, seed=1, outcome="win", differential=10),
            "termination_reason": "no_progress",
            "bootstrap_truncated": True,
            "no_progress_steps": 8,
            "time_limit_ending": True,
            "safety_regression": True,
            "safety_flags": ["safety_regression"],
        }
        for seat in (0, 1)
    ]
    baseline = [
        {
            **_metric_record(seat=seat, seed=1, candidate="current", outcome="loss", differential=1),
            "termination_reason": "terminal",
            "bootstrap_truncated": False,
            "no_progress_steps": 0,
            "time_limit_ending": False,
            "safety_regression": False,
            "safety_flags": [],
        }
        for seat in (0, 1)
    ]

    decision = promotion_decision(candidate, baseline, min_valid_games=1)

    assert decision["status"] == "discard"
    assert decision["reasons"] == ["diagnostic_regression"]
    assert decision["diagnostic_deltas"]["truncation_count"] == 2
    assert decision["diagnostic_deltas"]["no_progress_count"] == 2
    assert decision["diagnostic_deltas"]["time_limit_endings"] == 2
    assert decision["diagnostic_deltas"]["safety_regression_count"] == 2
    assert set(decision["diagnostic_regressions"]) >= {
        "truncation_count", "no_progress_count", "time_limit_endings",
        "safety_regression_count",
    }


@pytest.mark.parametrize("issue", ["missing", "duplicate", "extra"])
def test_promotion_decision_rejects_non_exact_expected_matrix(issue):
    from scripts.evaluate import promotion_decision

    expected_matrix = [("pass", 1, 0), ("pass", 1, 1)]
    candidate = [
        _metric_record(seat=seat, seed=1, opponent="pass", outcome="win", differential=10)
        for seat in (0, 1)
    ]
    baseline = [
        _metric_record(
            seat=seat, seed=1, opponent="pass", candidate="current",
            outcome="loss", differential=1,
        )
        for seat in (0, 1)
    ]
    if issue == "missing":
        candidate.pop()
    elif issue == "duplicate":
        candidate.append(dict(candidate[0]))
    else:
        candidate.extend([
            _metric_record(seat=seat, seed=1, opponent="starter", outcome="win", differential=100)
            for seat in (0, 1)
        ])

    decision = promotion_decision(
        candidate, baseline, min_valid_games=1, expected_matrix=expected_matrix,
    )

    assert decision["status"] == "discard"
    assert decision["matrix_completeness"][issue]


def test_promotion_decision_uses_baseline_median_and_requires_it():
    from scripts.evaluate import promotion_decision

    candidate = [
        _metric_record(seat=seat, seed=seed, outcome="win", differential=differential)
        for seed, differential in enumerate((0, 0, 100), start=1)
        for seat in (0, 1)
    ]
    baseline = [
        _metric_record(seat=seat, seed=seed, outcome="loss", differential=10, candidate="current")
        for seed in (1, 2, 3)
        for seat in (0, 1)
    ]

    decision = promotion_decision(candidate, baseline, min_valid_games=1)

    assert decision["candidate"]["mean_paired_bank_differential"] > decision["baseline"]["mean_paired_bank_differential"]
    assert decision["candidate"]["median_paired_bank_differential"] < decision["baseline"]["median_paired_bank_differential"]
    assert decision["reasons"] == ["no_paired_improvement"]
    assert promotion_decision(candidate, [], min_valid_games=1)["reasons"] == [
        "no_paired_improvement"
    ]


@pytest.mark.parametrize("kwargs", [
    {"seed": True},
    {"seed": 1.0},
    {"steps": True},
    {"steps": 2.0},
    {"steps": 0},
])
def test_run_game_rejects_non_integer_seed_or_steps(kwargs):
    from scripts.evaluate import run_game

    parameters = {"seed": 1, "steps": 2}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        run_game(variant="mixed", opponent="pass", **parameters)


def test_run_matrix_rejects_invalid_seed_and_steps_before_running(monkeypatch):
    from scripts.evaluate import run_matrix

    monkeypatch.setattr("scripts.evaluate.run_game", lambda **kwargs: pytest.fail("run_game should not run"))

    with pytest.raises(ValueError):
        run_matrix(variants=["mixed"], opponents=["pass"], seeds=[1, True], steps=2)
    with pytest.raises(ValueError):
        run_matrix(variants=["mixed"], opponents=["pass"], seeds=[1], steps=True)


def test_worker_rejects_boolean_seat_in_request_and_failure_record():
    from scripts.evaluation_worker import _request_failure, run_request

    request = {
        "variant": "mixed", "opponent": "pass", "seed": 1,
        "steps": 2, "seat": True,
    }
    result = run_request(request)
    failure = _request_failure(request, "invalid seat")

    assert result["framework_error"] is True
    assert result["seat"] == 0
    assert failure["framework_error"] is True
    assert failure["seat"] == 0


def test_worker_rejects_boolean_seat_directly_in_main(monkeypatch, capsys):
    import scripts.evaluation_worker as worker

    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "variant": "mixed", "opponent": "pass", "seed": 1,
            "steps": 2, "seat": True,
        }) + "\n"),
    )

    assert worker.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["framework_error"] is True


def test_replay_record_rejects_boolean_seat_and_candidate_player():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()

    seat_result = replay_record(replay, variant="mixed", opponent="pass", seed=1, seat=True)
    candidate_result = replay_record(
        replay, variant="mixed", opponent="pass", seed=1, candidate_player=True,
    )

    assert seat_result["framework_error"] is True
    assert candidate_result["framework_error"] is True


def test_replay_record_and_validation_reject_boolean_seed_aliases():
    from scripts.evaluate import _mapping, _player_states, _valid_replay, replay_record

    replay = _strict_two_turn_replay()
    own_states = _player_states(replay, 0)
    other_states = _player_states(replay, 1)
    configuration = _mapping(replay["configuration"])

    record = replay_record(replay, variant="mixed", opponent="pass", seed=True)

    assert record["framework_error"] is True
    assert _valid_replay(
        replay, own_states, other_states, configuration,
        expected_seed=True, candidate_player=0,
    ) is False


@pytest.mark.parametrize("seat", [0, 1])
def test_worker_runs_from_project_root_in_a_fresh_interpreter(seat):
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/evaluation_worker.py"],
        input=json.dumps({
            "variant": "mixed", "opponent": "pass", "seed": 1,
            "steps": 8, "seat": seat,
        }) + "\n",
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["variant"] == "mixed"
    assert response["opponent"] == "pass"
    assert response["seed"] == 1
    assert response["seat"] == seat
    assert response["framework_error"] is False
    assert response["outcome"] in {"win", "loss", "tie"}
    assert response["final_bank"] is not None
    assert response["opponent_final_bank"] is not None
    assert response["termination_reason"] in {"terminal", "time_limit"}
    assert response["bootstrap_truncated"] is False
    assert response["no_progress_steps"] == 0
    assert response["time_limit_ending"] is (response["termination_reason"] == "time_limit")
    assert response["safety_flags"] == []
    assert response["safety_regression"] is False


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
@pytest.mark.parametrize("name", ["current", "melon", "premium", "mixed"])
def test_worker_runs_stable_candidates_through_candidate_request(name):
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/evaluation_worker.py"],
        input=json.dumps({
            "candidate": name, "opponent": "pass", "seed": 1,
            "steps": 8, "seat": 0,
        }) + "\n",
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["candidate"] == name
    assert response["variant"] == name
    assert response["framework_error"] is False


def test_worker_rejects_ambiguous_variant_and_candidate_request():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/evaluation_worker.py"],
        input=json.dumps({
            "variant": "mixed", "candidate": "mixed", "opponent": "pass",
            "seed": 1, "steps": 8, "seat": 0,
        }) + "\n",
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["framework_error"] is True
    assert "separate modes" in response["error"]


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_run_game_real_worker_seat_one_returns_candidate_metrics(monkeypatch):
    from scripts.evaluate import run_game

    monkeypatch.setattr("scripts.evaluate._WORKER_TIMEOUT_SECONDS", 30)
    record = run_game(variant="mixed", opponent="pass", seed=1, steps=8, seat=1)

    assert record["framework_error"] is False
    assert record["seat"] == 1
    assert record["outcome"] in {"win", "loss", "tie"}
    assert record["final_bank"] is not None
    assert record["opponent_final_bank"] is not None
    assert record["bank_differential"] is not None


def test_worker_orders_candidate_and_opponent_by_seat():
    from scripts.evaluation_worker import _ordered_agents

    candidate = object()
    opponent = object()

    assert _ordered_agents(candidate, opponent, 0) == [candidate, opponent]
    assert _ordered_agents(candidate, opponent, 1) == [opponent, candidate]


def test_nonzero_worker_exit_is_normalized_as_framework_failure(monkeypatch):
    import scripts.evaluate as evaluate

    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 7, stdout="", stderr="worker exploded",
        ),
    )

    record = evaluate.run_game(variant="mixed", opponent="pass", seed=4, steps=2)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"
    assert "worker exited with status 7" in record["error"]


def test_worker_timeout_is_normalized_as_framework_failure(monkeypatch):
    import scripts.evaluate as evaluate

    def fail_worker(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(evaluate.subprocess, "run", fail_worker)

    record = evaluate.run_game(variant="mixed", opponent="pass", seed=4, steps=2, seat=1)

    assert record["seat"] == 1
    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"
    assert "timeout" in record["error"].lower()


def _complete_worker_record():
    return {
        "variant": "mixed",
        "opponent": "pass",
        "seed": 4,
        "seat": 1,
        "outcome": "tie",
        "final_bank": 100.0,
        "opponent_final_bank": 100.0,
        "bank_differential": 0.0,
        "framework_error": False,
        "shed_overflow": 0.0,
        "price_floor_sales": 0,
        "missed_basic_needs": 0,
        "submitted_market_order_count": 0,
        "market_transaction_count": 0,
        "same_item_market_churn": 0,
        "same_item_sell_buy_churn": 0,
        "terminal_cash": 100.0,
        "terminal_inventory_value": 0.0,
    }


@pytest.mark.parametrize("field,value", [
    ("framework_error", True),
    ("outcome", "framework_error"),
    ("final_bank", None),
    ("opponent_final_bank", None),
    ("bank_differential", None),
    ("shed_overflow", None),
    ("price_floor_sales", None),
    ("missed_basic_needs", None),
])
def test_worker_result_rejects_contradictory_or_incomplete_success(monkeypatch, field, value):
    import scripts.evaluate as evaluate

    response = _complete_worker_record()
    response[field] = value
    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(response) + "\n", stderr="",
        ),
    )

    record = evaluate.run_game(variant="mixed", opponent="pass", seed=4, steps=2, seat=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"
    assert "invalid worker result" in record["error"]


@pytest.mark.parametrize("field,value", [
    ("variant", "conservative"),
    ("opponent", "starter"),
    ("seed", True),
    ("seat", True),
    ("seat", 2),
    ("outcome", "invalid"),
    ("outcome", []),
    ("framework_error", 1),
    ("final_bank", True),
    ("bank_differential", "not-a-number"),
    ("shed_overflow", float("nan")),
    ("price_floor_sales", float("inf")),
])
def test_worker_result_must_match_request_and_normalized_types(monkeypatch, field, value):
    import scripts.evaluate as evaluate

    response = _complete_worker_record()
    response[field] = value
    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(response) + "\n", stderr="",
        ),
    )

    record = evaluate.run_game(variant="mixed", opponent="pass", seed=4, steps=2, seat=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"
    assert "invalid worker result" in record["error"]


def test_worker_evaluator_exception_is_reported_on_stderr_and_exits_nonzero(monkeypatch, capsys):
    import scripts.evaluation_worker as worker

    def fail_replay(*args, **kwargs):
        raise ValueError("replay validator exploded")

    monkeypatch.setattr(worker, "replay_record", fail_replay)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "variant": "mixed", "opponent": "pass", "seed": 1,
            "steps": 2, "seat": 0,
        }) + "\n"),
    )

    assert worker.main() == 1
    captured = capsys.readouterr()
    assert "replay validator exploded" in captured.err
    assert captured.out == ""


def test_worker_rejects_a_second_json_request(monkeypatch, capsys):
    import scripts.evaluation_worker as worker

    request = json.dumps({
        "variant": "mixed", "opponent": "pass", "seed": 1,
        "steps": 2, "seat": 0,
    })
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{request}\n{{}}\n"))

    assert worker.main() == 1
    captured = capsys.readouterr()
    assert "exactly one JSON request" in captured.err
    assert captured.out == ""


def test_worker_candidate_construction_failure_is_evaluator_failure(monkeypatch, capsys):
    import scripts.evaluation_worker as worker

    def fail_constructor(*args, **kwargs):
        raise RuntimeError("constructor exploded")

    monkeypatch.setattr(worker, "VariantPolicy", fail_constructor)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "variant": "mixed", "opponent": "pass", "seed": 1,
            "steps": 2, "seat": 0,
        }) + "\n"),
    )

    assert worker.main() == 1
    captured = capsys.readouterr()
    assert "candidate policy construction failure" in captured.err
    assert captured.out == ""


def test_worker_candidate_invocation_failure_is_evaluator_failure(monkeypatch, capsys):
    import scripts.evaluation_worker as worker

    class FakeEnvironment:
        configuration = {}

        def run(self, agents):
            agents[0]({}, {})

        def toJSON(self):
            return {}

    def fail_candidate(*args, **kwargs):
        raise RuntimeError("candidate exploded")

    monkeypatch.setattr(worker, "VariantPolicy", lambda *args, **kwargs: fail_candidate)
    monkeypatch.setitem(
        sys.modules,
        "kaggle_environments",
        type("FakeKaggleEnvironments", (), {"make": lambda *args, **kwargs: FakeEnvironment()})(),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "variant": "mixed", "opponent": "pass", "seed": 1,
            "steps": 2, "seat": 0,
        }) + "\n"),
    )

    assert worker.main() == 1
    captured = capsys.readouterr()
    assert "candidate policy failure" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("failure", ["make", "run", "toJSON"])
def test_worker_engine_failures_are_framework_records(monkeypatch, capsys, failure):
    import scripts.evaluation_worker as worker

    class FakeEnvironment:
        configuration = {}

        def run(self, agents):
            if failure == "run":
                raise RuntimeError("engine run exploded")

        def toJSON(self):
            if failure == "toJSON":
                raise RuntimeError("engine serialization exploded")
            return {}

    if failure == "make":
        def make(*args, **kwargs):
            raise RuntimeError("engine make exploded")
    else:
        make = lambda *args, **kwargs: FakeEnvironment()
    monkeypatch.setitem(
        sys.modules,
        "kaggle_environments",
        type("FakeKaggleEnvironments", (), {"make": staticmethod(make)})(),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "variant": "mixed", "opponent": "pass", "seed": 1,
            "steps": 2, "seat": 0,
        }) + "\n"),
    )

    assert worker.main() == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["framework_error"] is True
    assert result["error"]
    assert "engine" in result["error"]


def test_replay_type_error_is_not_normalized_as_framework_failure(monkeypatch, capsys):
    import scripts.evaluate as evaluate
    import scripts.evaluation_worker as worker

    def fail_replay(*args, **kwargs):
        raise TypeError("unexpected evaluator type error")

    monkeypatch.setattr(evaluate, "_replay_record", fail_replay)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({
            "variant": "mixed", "opponent": "pass", "seed": 1,
            "steps": 2, "seat": 0,
        }) + "\n"),
    )

    assert worker.main() == 1
    captured = capsys.readouterr()
    assert "unexpected evaluator type error" in captured.err
    assert captured.out == ""


def test_sidecar_sort_order_includes_seat(tmp_path):
    from scripts.evaluate import write_result_document

    records = [
        {"ablation": "baseline", "variant": "mixed", "opponent": "pass", "seed": 1, "seat": 1},
        {"ablation": "baseline", "variant": "mixed", "opponent": "pass", "seed": 1, "seat": 0},
    ]
    path = tmp_path / "evaluation.json"
    write_result_document(path, {}, records=records)

    ordered = json.loads(path.with_name("evaluation.replays.json").read_text())["records"]

    assert [record["seat"] for record in ordered] == [0, 1]


def test_write_result_document_rejects_non_finite_json_before_publication(tmp_path):
    from scripts.evaluate import write_result_document

    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="Out of range|finite|JSON"):
        write_result_document(output, {"metric": float("nan")}, records=[])

    assert not output.exists()
    assert not output.with_name("evaluation.replays.json").exists()


def test_write_result_document_rejects_protected_and_symlinked_destinations(tmp_path):
    from scripts.evaluate import write_result_document

    with pytest.raises(ValueError, match="production|protected"):
        write_result_document(tmp_path / "model.json", {}, records=[])
    with pytest.raises(ValueError, match="production"):
        write_result_document(tmp_path / "models" / "evaluation.json", {}, records=[])

    target = tmp_path / "target.json"
    target.write_text("keep\n", encoding="utf-8")
    link = tmp_path / "evaluation.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        write_result_document(link, {}, records=[])

    assert target.read_text(encoding="utf-8") == "keep\n"


def test_malformed_successful_worker_output_is_normalized(monkeypatch):
    import scripts.evaluate as evaluate

    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps({"framework_error": False, "outcome": "tie"}) + "\n", stderr="",
        ),
    )

    record = evaluate.run_game(variant="mixed", opponent="pass", seed=4, steps=2, seat=1)

    required = {
        "variant", "opponent", "seed", "seat", "outcome", "final_bank",
        "opponent_final_bank", "bank_differential", "framework_error",
        "shed_overflow", "price_floor_sales", "missed_basic_needs",
    }
    assert required <= record.keys()
    assert record["framework_error"] is True
    assert "missing normalized record fields" in record["error"]


def test_worker_runner_exception_is_normalized_as_framework_failure(monkeypatch):
    import scripts.evaluate as evaluate

    def raise_runner(*args, **kwargs):
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(evaluate.subprocess, "run", raise_runner)

    record = evaluate.run_game(variant="mixed", opponent="pass", seed=4, steps=2)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"
    assert "runner exploded" in record["error"]


def test_run_evaluation_forwards_seats_to_every_matrix(monkeypatch):
    from scripts.evaluate import run_evaluation

    calls = []

    def fake_run_matrix(**kwargs):
        calls.append(kwargs)
        return {"records": []}

    monkeypatch.setattr("scripts.evaluate.run_matrix", fake_run_matrix)
    run_evaluation(
        variants=["mixed"], opponents=["pass"], seeds=[1], steps=2,
        seats=[0, 1], ablations=[("animals", False)],
    )

    assert [call["seats"] for call in calls] == [[0, 1], [0, 1]]


def test_run_matrix_defaults_to_both_seats_without_running_real_games(monkeypatch):
    from scripts.evaluate import run_matrix

    calls = []
    monkeypatch.setattr("scripts.evaluate.run_game", lambda **kwargs: calls.append(kwargs) or kwargs)

    run_matrix(variants=["mixed"], opponents=["pass"], seeds=[1], steps=2)

    assert [call["seat"] for call in calls] == [0, 1]


def test_run_evaluation_defaults_to_both_seats_without_running_real_games(monkeypatch):
    from scripts.evaluate import run_evaluation

    calls = []

    def fake_run_matrix(**kwargs):
        calls.append(kwargs)
        return {"records": []}

    monkeypatch.setattr("scripts.evaluate.run_matrix", fake_run_matrix)

    run_evaluation(variants=["mixed"], opponents=["pass"], seeds=[1], steps=2)

    assert calls == [{
        "variants": ["mixed"], "opponents": ["pass"], "seeds": [1], "steps": 2,
        "ablations": {
            "route_scheduling": True, "market_batch_sizing": True,
            "shop_adaptation": True, "land_purchase": True, "animals": True,
        }, "seats": [0, 1],
    }]


def test_run_evaluation_keeps_holdout_matrix_disjoint_and_separate(monkeypatch):
    from scripts.evaluate import run_evaluation

    calls = []

    def fake_run_matrix(**kwargs):
        calls.append(kwargs)
        return {"records": [{"seed": seed} for seed in kwargs["seeds"]]}

    monkeypatch.setattr("scripts.evaluate.run_matrix", fake_run_matrix)
    result = run_evaluation(
        variants=["mixed"], opponents=["pass"], seeds=[1, 2], steps=2,
        holdout_seeds=[100, 101], min_valid_games=1,
    )

    assert [call["seeds"] for call in calls] == [[1, 2], [100, 101]]
    assert result["records"] == [{"seed": 1}, {"seed": 2}]
    assert result["holdout_records"] == [{"seed": 100}, {"seed": 101}]


def test_run_matrix_forwards_candidate_seats_and_keeps_seed_pairs_ordered(monkeypatch):
    from scripts.evaluate import run_matrix

    calls = []

    def fake_run_game(**kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr("scripts.evaluate.run_game", fake_run_game)
    result = run_matrix(
        variants=["mixed"], opponents=["pass"], seeds=[4, 9], steps=2,
        seats=[0, 1],
    )

    assert result["records"] == calls
    assert [(call["seat"], call["seed"]) for call in calls] == [
        (0, 4), (1, 4), (0, 9), (1, 9),
    ]
    assert all(
        {call["variant"], call["opponent"], call["steps"]} == {"mixed", "pass", 2}
        for call in calls
    )


def test_run_matrix_accepts_candidates_and_forwards_each_seat_and_seed(monkeypatch):
    from scripts.evaluate import run_matrix

    calls = []

    def fake_run_game(**kwargs):
        calls.append(kwargs)
        return {
            **_complete_worker_record(),
            "candidate": kwargs["candidate"],
            "variant": kwargs["candidate"],
            "opponent": kwargs["opponent"],
            "seed": kwargs["seed"],
            "seat": kwargs["seat"],
            "error": "worker diagnostic",
        }

    monkeypatch.setattr("scripts.evaluate.run_game", fake_run_game)
    result = run_matrix(
        candidates=["mixed"], opponents=["pass"], seeds=[4, 9], steps=2,
        seats=[0, 1],
    )

    assert [(call["candidate"], call["opponent"], call["seed"], call["seat"], call["steps"])
            for call in calls] == [
        ("mixed", "pass", 4, 0, 2),
        ("mixed", "pass", 4, 1, 2),
        ("mixed", "pass", 9, 0, 2),
        ("mixed", "pass", 9, 1, 2),
    ]
    assert [set(record) for record in result["records"]] == [
        {"candidate", "variant", "opponent", "seed", "seat", "outcome",
         "final_bank", "opponent_final_bank", "bank_differential", "framework_error",
         "shed_overflow", "price_floor_sales", "missed_basic_needs",
         "submitted_market_order_count",
         "market_transaction_count", "same_item_market_churn",
         "same_item_sell_buy_churn", "terminal_cash", "terminal_inventory_value",
         "error"},
    ] * 4
    assert [record["candidate"] for record in result["records"]] == ["mixed"] * 4
    assert [record["error"] for record in result["records"]] == ["worker diagnostic"] * 4


def test_run_evaluation_accepts_candidates_and_forwards_candidate_identity(monkeypatch):
    from scripts.evaluate import run_evaluation

    calls = []

    def fake_run_matrix(**kwargs):
        calls.append(kwargs)
        return {"records": []}

    monkeypatch.setattr("scripts.evaluate.run_matrix", fake_run_matrix)
    run_evaluation(
        candidates=["mixed"], opponents=["pass"], seeds=[1], steps=2,
        seats=[0, 1], ablations=[("animals", False)],
    )

    assert [call["candidates"] for call in calls] == [["mixed"], ["mixed"]]
    assert [call["seats"] for call in calls] == [[0, 1], [0, 1]]


def test_main_includes_seats_in_evaluation_and_report_config(monkeypatch, tmp_path):
    import scripts.evaluate as evaluate

    captured = {}

    def fake_run_evaluation(**kwargs):
        captured["evaluation"] = kwargs
        return {"records": [], "ablation_records": {}, "ablation_configs": {}}

    def fake_build_result_document(**kwargs):
        captured["config"] = kwargs["config"]
        return {"selected_default": "mixed"}

    monkeypatch.setattr(evaluate, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(evaluate, "build_result_document", fake_build_result_document)
    monkeypatch.setattr(evaluate, "write_result_document", lambda *args, **kwargs: None)

    assert evaluate.main([
        "--seeds", "1", "--steps", "2", "--seats", "0", "1",
        "--output", str(tmp_path / "evaluation.json"),
    ]) == 0
    assert captured["evaluation"]["seats"] == [0, 1]
    assert captured["config"]["seats"] == [0, 1]


def test_main_exposes_candidates_in_evaluation_and_report_config(monkeypatch, tmp_path):
    import scripts.evaluate as evaluate

    captured = {}

    def fake_run_evaluation(**kwargs):
        captured["evaluation"] = kwargs
        return {"records": [], "ablation_records": {}, "ablation_configs": {}}

    def fake_build_result_document(**kwargs):
        captured["config"] = kwargs["config"]
        return {"selected_default": "mixed"}

    monkeypatch.setattr(evaluate, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(evaluate, "build_result_document", fake_build_result_document)
    monkeypatch.setattr(evaluate, "write_result_document", lambda *args, **kwargs: None)

    assert evaluate.main([
        "--seeds", "1", "--steps", "2", "--candidates", "mixed",
        "--output", str(tmp_path / "evaluation.json"),
    ]) == 0
    assert captured["evaluation"]["candidates"] == ["mixed"]
    assert captured["config"]["candidates"] == ["mixed"]


def test_main_forwards_holdout_contract_to_evaluation_and_report(monkeypatch, tmp_path):
    import scripts.evaluate as evaluate

    captured = {}

    def fake_run_evaluation(**kwargs):
        captured["evaluation"] = kwargs
        return {"records": [], "ablation_records": {}, "ablation_configs": {}, "holdout_records": []}

    def fake_build_result_document(**kwargs):
        captured["config"] = kwargs["config"]
        captured["holdout_records"] = kwargs["holdout_records"]
        return {"selected_default": "mixed"}

    monkeypatch.setattr(evaluate, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(evaluate, "build_result_document", fake_build_result_document)
    monkeypatch.setattr(evaluate, "write_result_document", lambda *args, **kwargs: None)

    assert evaluate.main([
        "--seeds", "2", "--start-seed", "10", "--steps", "2",
        "--holdout-seeds", "100", "101", "--min-valid-games", "1",
        "--output", str(tmp_path / "evaluation.json"),
    ]) == 0
    assert captured["evaluation"]["holdout_seeds"] == [100, 101]
    assert captured["evaluation"]["min_valid_games"] == 1
    assert captured["config"]["holdout_seed_values"] == [100, 101]
    assert captured["config"]["min_valid_games"] == 1
    assert captured["holdout_records"] == []


def test_report_groups_records_by_candidate_identity():
    from scripts.evaluate import build_result_document

    document = build_result_document(
        config={"opponents": ["pass"], "candidates": ["animal-heavy"]},
        records=[{
            "candidate": "animal-heavy", "variant": "mixed", "opponent": "pass", "seed": 1,
            "outcome": "win", "final_bank": 100, "opponent_final_bank": 50,
            "bank_differential": 50, "framework_error": False,
            "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 0,
        }],
    )

    assert document["results"]["animal-heavy"]["pass"]["wins"] == 1


def test_report_schema_carries_candidate_diagnostics_from_paired_summary():
    from scripts.evaluate import build_result_document

    records = [
        {
            **_metric_record(
                seat=seat, seed=1, candidate=candidate,
                outcome="win" if candidate == "challenger" else "loss",
                differential=10 if candidate == "challenger" else 1,
            ),
            "termination_reason": "no_progress" if candidate == "challenger" else "terminal",
            "bootstrap_truncated": candidate == "challenger",
            "no_progress_steps": 5 if candidate == "challenger" else 0,
            "time_limit_ending": candidate == "challenger",
            "safety_regression": candidate == "challenger",
            "safety_flags": ["safety_regression"] if candidate == "challenger" else [],
        }
        for candidate in ("baseline", "challenger")
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={
            "candidates": ["baseline", "challenger"],
            "opponents": ["pass"],
            "min_valid_games": 1,
        },
        records=records,
    )

    diagnostics = document["paired_summaries"]["challenger"]["diagnostics"]
    assert diagnostics["truncation_count"] == 2
    assert diagnostics["no_progress_count"] == 2
    assert diagnostics["time_limit_endings"] == 2
    assert diagnostics["safety_regression_count"] == 2
    assert document["promotion_decisions"]["challenger"]["reasons"] == [
        "diagnostic_regression"
    ]
    assert document["promotion_decisions"]["challenger"]["diagnostic_deltas"][
        "no_progress_count"
    ] == 2


def test_report_exposes_per_candidate_pairs_and_promotion_decisions():
    from scripts.evaluate import build_result_document

    records = [
        _metric_record(seat=seat, seed=seed, candidate=candidate,
                       outcome="win" if candidate == "challenger" else "loss",
                       differential=10 if candidate == "challenger" else 1)
        for candidate in ("baseline", "challenger")
        for seed in (1, 2)
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={"candidates": ["baseline", "challenger"], "opponents": ["pass"],
                "min_valid_games": 1},
        records=records,
    )

    assert document["metadata"]["baseline_convention"] == (
        "the first configured candidate is the baseline for promotion decisions"
    )
    assert list(document["paired_summaries"]) == ["baseline", "challenger"]
    assert document["paired_summaries"]["challenger"]["paired_games"] == 2
    assert document["promotion_decisions"]["baseline"]["status"] == "baseline"
    assert document["promotion_decisions"]["challenger"]["status"] == "promote"
    json.dumps(document, allow_nan=False)


def test_report_rejects_conflicting_variant_and_candidate_config_aliases():
    from scripts.evaluate import build_result_document

    with pytest.raises(ValueError, match="variants and candidates are separate modes"):
        build_result_document(
            config={"variants": ["mixed"], "candidates": ["animal-heavy"], "opponents": ["pass"]},
            records=[],
        )


def test_report_does_not_select_candidate_that_fails_safety_gates():
    from scripts.evaluate import build_result_document

    records = [
        _metric_record(seat=seat, seed=1, candidate="baseline", outcome="tie", differential=1)
        for seat in (0, 1)
    ] + [
        _metric_record(seat=seat, seed=1, candidate="challenger", outcome="win", differential=100,
                       missed_basic_needs=1)
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={"candidates": ["baseline", "challenger"], "opponents": ["pass"], "min_valid_games": 1},
        records=records,
    )

    assert document["promotion_decisions"]["challenger"]["status"] == "discard"
    assert document["promotion_decisions"]["challenger"]["reasons"] == ["missed_basic_needs"]
    assert document["selected_default"] == "baseline"


def test_report_has_no_default_when_baseline_fails_safety_gates():
    from scripts.evaluate import build_result_document

    records = [
        _metric_record(seat=seat, seed=1, candidate="baseline", outcome="tie", differential=1,
                       framework_error=(seat == 1))
        for seat in (0, 1)
    ] + [
        _metric_record(seat=seat, seed=1, candidate="challenger", outcome="win", differential=100)
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={"candidates": ["baseline", "challenger"], "opponents": ["pass"], "min_valid_games": 1},
        records=records,
    )

    assert document["promotion_decisions"]["baseline"]["status"] == "discard"
    assert document["promotion_decisions"]["baseline"]["reasons"] == ["framework_error"]
    assert document["selected_default"] is None


def test_holdout_report_controls_selection_and_exposes_holdout_decisions():
    from scripts.evaluate import build_result_document

    development = [
        _metric_record(
            seat=seat, seed=1, candidate=candidate,
            outcome="win" if candidate == "challenger" else "tie",
            differential=2 if candidate == "challenger" else 1,
        )
        for candidate in ("baseline", "challenger")
        for seat in (0, 1)
    ]
    holdout = [
        _metric_record(
            seat=seat, seed=100, candidate=candidate,
            outcome="win" if candidate == "challenger" else "tie",
            differential=100 if candidate == "challenger" else 1,
            missed_basic_needs=1 if candidate == "challenger" else 0,
        )
        for candidate in ("baseline", "challenger")
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={
            "candidates": ["baseline", "challenger"], "opponents": ["pass"],
            "seed_values": [1], "holdout_seed_values": [100],
            "steps": 720, "seats": [0, 1], "min_valid_games": 1,
        },
        records=development, holdout_records=holdout,
        command=["scripts/evaluate.py"],
    )

    assert document["selected_candidate"] == "baseline"
    assert document["selected_default"] == "baseline"
    assert document["holdout"]["selected_candidate"] == "baseline"
    assert document["holdout_promotion_decisions"]["challenger"]["reasons"] == [
        "missed_basic_needs"
    ]
    assert document["promotion_decisions"]["challenger"]["holdout"]["status"] == "discard"
    assert document["metadata"]["holdout"]["selected_candidate"] == "baseline"
    assert document["metadata"]["manifest"]["schema_version"] == 3
    json.dumps(document, allow_nan=False)


def test_partial_holdout_cannot_populate_selected_candidate():
    from scripts.evaluate import build_result_document

    development = [
        _metric_record(
            seat=seat, seed=1, candidate=candidate,
            outcome="win" if candidate == "challenger" else "tie",
            differential=2 if candidate == "challenger" else 1,
        )
        for candidate in ("baseline", "challenger")
        for seat in (0, 1)
    ]
    partial_holdout = [
        _metric_record(
            seat=seat, seed=100, candidate=candidate,
            outcome="win" if candidate == "challenger" else "tie",
            differential=100 if candidate == "challenger" else 1,
        )
        for candidate in ("baseline", "challenger")
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={
            "candidates": ["baseline", "challenger"], "opponents": ["pass"],
            "seed_values": [1], "holdout_seed_values": [100, 101],
            "seats": [0, 1], "min_valid_games": 1,
        },
        records=development, holdout_records=partial_holdout,
    )

    assert document["selected_candidate"] is None
    assert document["holdout"]["selected_candidate"] is None
    assert document["holdout_promotion_decisions"]["challenger"]["status"] == "discard"
    assert document["holdout_promotion_decisions"]["challenger"]["matrix_completeness"]["missing"]


def test_no_holdout_report_labels_development_only_default_selection():
    from scripts.evaluate import build_result_document

    records = [
        _metric_record(
            seat=seat, seed=1, candidate=candidate,
            outcome="win" if candidate == "challenger" else "tie",
            differential=2 if candidate == "challenger" else 1,
        )
        for candidate in ("baseline", "challenger")
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={
            "candidates": ["baseline", "challenger"], "opponents": ["pass"],
            "seed_values": [1], "seats": [0, 1], "min_valid_games": 1,
        },
        records=records,
    )

    assert document["selected_candidate"] is None
    assert document["selected_default"] == "challenger"
    assert document["selected_default_source"] == "development_only"
    assert document["metadata"]["selected_candidate"] is None
    assert document["metadata"]["selected_default_source"] == "development_only"


def test_unavailable_learned_candidate_cannot_win_complete_development_and_holdout():
    from scripts.evaluate import build_result_document

    development = [
        _metric_record(
            seat=seat, seed=1, candidate=candidate,
            outcome="win" if candidate == "learned_v1" else "tie",
            differential=2 if candidate == "learned_v1" else 1,
        )
        for candidate in ("current", "learned_v1")
        for seat in (0, 1)
    ]
    holdout = [
        _metric_record(
            seat=seat, seed=100, candidate=candidate,
            outcome="win" if candidate == "learned_v1" else "tie",
            differential=2 if candidate == "learned_v1" else 1,
        )
        for candidate in ("current", "learned_v1")
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={
            "candidates": ["current", "learned_v1"],
            "opponents": ["pass"],
            "seed_values": [1],
            "holdout_seed_values": [100],
            "seats": [0, 1],
            "min_valid_games": 1,
        },
        records=development,
        holdout_records=holdout,
    )

    assert document["selected_candidate"] is None
    assert document["holdout"]["selected_candidate"] is None
    assert document["promotion_decisions"]["learned_v1"]["reasons"] == [
        "candidate_unavailable"
    ]
    assert document["holdout_promotion_decisions"]["learned_v1"]["reasons"] == [
        "candidate_unavailable"
    ]


def test_sidecar_sort_uses_candidate_and_canonical_record_tiebreaker(tmp_path):
    from scripts.evaluate import write_result_document

    records = [
        {"candidate": "z", "variant": "mixed", "opponent": "pass", "seed": 1, "seat": 0, "payload": {"b": 1, "a": 2}},
        {"candidate": "a", "variant": "mixed", "opponent": "pass", "seed": 1, "seat": 0, "payload": {"z": 1}},
        {"candidate": "a", "variant": "mixed", "opponent": "pass", "seed": 1, "seat": 0, "payload": {"a": 1}},
    ]

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_result_document(first, {}, records=list(reversed(records)))
    write_result_document(second, {}, records=records)

    assert first.with_name("first.replays.json").read_bytes().replace(b"first", b"second") == second.with_name("second.replays.json").read_bytes()


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_quick_starter_replay_does_not_require_full_season_liquidation():
    from scripts.evaluate import run_game

    record = run_game(variant="mixed", opponent="starter", seed=1, steps=96)

    assert record["framework_error"] is False


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_seed4_random_replay_accepts_natural_boundary_decay():
    from scripts.evaluate import run_game

    record = run_game(variant="mixed", opponent="random", seed=4, steps=720)

    assert record["framework_error"] is False
    assert record["missed_basic_needs"] == 0


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
@pytest.mark.parametrize("opponent", ["pass", "random", "starter"])
def test_animal_heavy_seed17_preserves_required_needs(opponent):
    from scripts.evaluate import run_game

    record = run_game(variant="animal-heavy", opponent=opponent, seed=17, steps=720)

    assert record["framework_error"] is False
    assert record["missed_basic_needs"] == 0


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_conservative_seed17_random_replay_has_no_framework_error():
    from scripts.evaluate import run_game

    record = run_game(variant="conservative", opponent="random", seed=17, steps=720)

    assert record["framework_error"] is False


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_conservative_seed17_random_replay_meets_all_basic_need_deadlines():
    from scripts.evaluate import run_game

    record = run_game(variant="conservative", opponent="random", seed=17, steps=720)

    assert record["framework_error"] is False
    assert record["missed_basic_needs"] == 0


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
@pytest.mark.parametrize("variant", ["conservative", "melon-heavy"])
@pytest.mark.parametrize("opponent", ["pass", "random", "starter"])
def test_seed17_required_needs_regression_matrix(variant, opponent):
    from scripts.evaluate import run_game

    record = run_game(variant=variant, opponent=opponent, seed=17, steps=720)

    assert record["framework_error"] is False
    assert record["missed_basic_needs"] == 0


def test_percentile_uses_linear_interpolation():
    from scripts.evaluate import percentile

    assert percentile([10, 20, 30, 40, 50], 5) == 12.0
    assert percentile([10, 20, 30, 40, 50], 50) == 30.0
    assert percentile([], 5) is None


def test_aggregate_counts_outcomes_and_metrics():
    from scripts.evaluate import aggregate_records

    records = [
        {"outcome": "win", "final_bank": 110, "opponent_final_bank": 90, "framework_error": False,
         "shed_overflow": 2, "price_floor_sales": 4, "missed_basic_needs": 1},
        {"outcome": "framework_error", "final_bank": None, "opponent_final_bank": None, "framework_error": True,
         "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 3},
        {"outcome": "tie", "final_bank": 95, "opponent_final_bank": 95, "framework_error": False,
         "shed_overflow": 1, "price_floor_sales": 2, "missed_basic_needs": 0},
    ]

    summary = aggregate_records(records)

    assert summary["count"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["ties"] == 1
    assert summary["framework_failures"] == 1
    assert summary["valid_count"] == 2
    assert summary["win_rate"] == pytest.approx(1 / 2)
    assert summary["mean_final_bank"] == pytest.approx(102.5)
    assert summary["median_final_bank"] == pytest.approx(102.5)
    assert summary["fifth_percentile_final_bank"] == pytest.approx(95.75)
    assert summary["mean_bank_differential"] == pytest.approx(0)
    assert summary["framework_error_rate"] == pytest.approx(1 / 3)
    assert summary["average_shed_overflow"] == pytest.approx(1)
    assert summary["average_price_floor_sales"] == pytest.approx(2)
    assert summary["average_missed_basic_needs_events"] == pytest.approx(4 / 3)


def test_replay_record_and_aggregation_emit_real_stall_diagnostics():
    from scripts.evaluate import aggregate_records, replay_record

    replay = _strict_two_turn_replay()
    replay["info"] = {
        "time_limit_ending": True,
        "safety_flags": ["safety_regression"],
    }
    replay["steps"][-1][0]["info"] = {
        "termination_reason": "no_progress",
        "bootstrap_truncated": True,
        "no_progress_steps": 7,
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)
    summary = aggregate_records([record])

    assert record["termination_reason"] == "no_progress"
    assert record["bootstrap_truncated"] is True
    assert record["no_progress_steps"] == 7
    assert record["time_limit_ending"] is True
    assert record["safety_flags"] == ["safety_regression"]
    assert record["safety_regression"] is True
    assert summary["termination_reasons"] == {"no_progress": 1}
    assert summary["bootstrap_truncated_count"] == 1
    assert summary["truncation_count"] == 1
    assert summary["truncation_rate"] == pytest.approx(1.0)
    assert summary["resolved_count"] == 0
    assert summary["no_progress_count"] == 1
    assert summary["no_progress_rate"] == pytest.approx(1.0)
    assert summary["max_no_progress_steps"] == 7
    assert summary["max_no_progress_streak"] == 7
    assert summary["time_limit_endings"] == 1
    assert summary["time_limit_rate"] == pytest.approx(1.0)
    assert summary["safety_regression_count"] == 1
    assert summary["safety_regression_rate"] == pytest.approx(1.0)


def test_run_matrix_executes_variant_opponent_cartesian_product_with_same_seeds(monkeypatch):
    from scripts.evaluate import run_matrix

    calls = []

    def fake_run_game(*, variant, opponent, seed, steps, seat):
        calls.append((variant, opponent, seed, steps, seat))
        return {
            "variant": variant,
            "opponent": opponent,
            "seed": seed,
            "outcome": "tie",
            "final_bank": 10,
            "opponent_final_bank": 10,
            "framework_error": False,
            "shed_overflow": 0,
            "price_floor_sales": 0,
            "missed_basic_needs": 0,
        }

    monkeypatch.setattr("scripts.evaluate.run_game", fake_run_game)
    result = run_matrix(
        variants=["mixed", "melon-heavy"],
        opponents=["pass", "starter"],
        seeds=[4, 9],
        steps=16,
    )

    assert len(result["records"]) == 16
    assert calls == [
        (variant, opponent, seed, 16, seat)
        for variant in ["mixed", "melon-heavy"]
        for opponent in ["pass", "starter"]
        for seed in [4, 9]
        for seat in [0, 1]
    ]


def test_result_schema_is_json_serializable_and_has_selected_default():
    from scripts.evaluate import build_result_document

    document = build_result_document(
        config={"seeds": 1, "start_seed": 7, "steps": 4, "opponents": ["pass"], "variants": ["mixed"]},
        records=[{
            "variant": "mixed", "opponent": "pass", "seed": 7, "outcome": "win",
            "final_bank": 100, "opponent_final_bank": 50, "framework_error": False,
            "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 0,
        }],
    )

    assert document["schema_version"] == 1
    assert document["metadata"]["config"]["variants"] == ["mixed"]
    assert document["selected_default"] == "mixed"
    assert document["results"]["mixed"]["pass"]["wins"] == 1
    json.dumps(document, sort_keys=True)


def test_default_selection_breaks_ties_by_median_then_failure_rate():
    from scripts.evaluate import build_result_document

    records = [
        {"variant": "mixed", "opponent": "pass", "outcome": "win", "final_bank": 100,
         "opponent_final_bank": 50, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "pass", "outcome": "win", "final_bank": 110,
         "opponent_final_bank": 50, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "mixed", "opponent": "starter", "outcome": "loss", "final_bank": 90,
         "opponent_final_bank": 100, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "starter", "outcome": "loss", "final_bank": 80,
         "opponent_final_bank": 100, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "starter", "outcome": "framework_error", "final_bank": None,
         "opponent_final_bank": None, "framework_error": True, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
    ]

    document = build_result_document(
        config={"opponents": ["pass", "starter"], "variants": ["mixed", "melon-heavy"]},
        records=records,
    )

    assert document["selected_default"] == "mixed"


def test_isolated_ablation_execution_keeps_each_toggle_separate(monkeypatch):
    from scripts.evaluate import run_evaluation

    calls = []

    def fake_run_game(*, variant, opponent, seed, steps, seat, ablations=None):
        calls.append((variant, opponent, seed, seat, dict(ablations or {})))
        return {
            "variant": variant, "opponent": opponent, "seed": seed, "outcome": "tie",
            "final_bank": 10, "opponent_final_bank": 10, "framework_error": False,
            "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 0,
        }

    monkeypatch.setattr("scripts.evaluate.run_game", fake_run_game)
    result = run_evaluation(
        variants=["mixed"], opponents=["pass"], seeds=[1], steps=4,
        ablations=[("animals", False), ("land_purchase", False)],
    )

    assert [call[4] for call in calls] == [
        {"animals": True, "land_purchase": True, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
        {"animals": True, "land_purchase": True, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
        {"animals": False, "land_purchase": True, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
        {"animals": False, "land_purchase": True, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
        {"animals": True, "land_purchase": False, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
        {"animals": True, "land_purchase": False, "market_batch_sizing": True, "route_scheduling": True, "shop_adaptation": True},
    ]
    assert [call[3] for call in calls] == [0, 1, 0, 1, 0, 1]
    assert set(result["ablation_records"]) == {"animals", "land_purchase"}


def test_write_result_document_is_byte_stable(tmp_path):
    from scripts.evaluate import write_result_document

    document = {"schema_version": 1, "metadata": {"config": {"x": 1}}, "selected_default": "mixed", "results": {}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_result_document(first, document, records=[])
    write_result_document(second, document, records=[])

    assert first.read_bytes() == second.read_bytes()
    assert first.with_name("first.replays.json").read_bytes() == second.with_name("second.replays.json").read_bytes()


def test_result_metadata_normalizes_invocation_and_sidecar_paths():
    from scripts.evaluate import build_result_document

    record = [{
        "variant": "mixed", "opponent": "pass", "seed": 7, "outcome": "win",
        "final_bank": 100, "opponent_final_bank": 50, "framework_error": False,
        "shed_overflow": 0, "price_floor_sales": 0, "missed_basic_needs": 0,
    }]
    first = build_result_document(
        config={"variants": ["mixed"], "opponents": ["pass"],
                "replay_summary": "/tmp/one/evaluation.replays.json"},
        records=record,
        command=["/tmp/one/scripts/evaluate.py", "--output", "/tmp/one/evaluation.json"],
    )
    second = build_result_document(
        config={"variants": ["mixed"], "opponents": ["pass"],
                "replay_summary": "/tmp/two/evaluation.replays.json"},
        records=record,
        command=["/tmp/two/scripts/evaluate.py", "--output", "/tmp/two/evaluation.json"],
    )

    assert first == second
    assert first["metadata"]["command"] == ["scripts/evaluate.py", "--output", "<report>"]
    assert first["metadata"]["config"]["replay_summary"] == "evaluation.replays.json"


def test_no_network_dependency_for_import_and_aggregation(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def block_network(name, *args, **kwargs):
        if name in {"requests", "urllib3", "httpx"}:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_network)
    from scripts.evaluate import aggregate_records

    assert aggregate_records([])["count"] == 0


def test_replay_record_extracts_outcome_and_replay_metrics():
    from scripts.evaluate import replay_record

    observation = {
        "player": 0,
        "step": 23,
        "hour": 23,
        "farms": [{"money": 120, "farmer": [0, 0], "hands": [], "tiles": [[{"kind": "PLANT", "watered_today": False}] + [None for _ in range(9)]] + [[None for _ in range(10)] for _ in range(9)]}],
        "private": {"shed": {"WHEAT": 102}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 1}, "inventory": {"WHEAT": 10000}},
    }
    replay = {
        "steps": [[
                {"observation": observation, "action": {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 3]]}, "status": "DONE", "info": {}},
                {"observation": {"player": 1, "farms": [{"money": 120}, {"money": 80, "farmer": [0, 0], "hands": [], "tiles": [[None for _ in range(10)] for _ in range(10)]}], "private": {"inventories": [{}]}, "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}}}, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
        ]],
        "rewards": [120, 80],
        "statuses": ["DONE", "DONE"],
        "info": {},
    }

    record = replay_record(_engine_envelope(replay, seed=5), variant="mixed", opponent="pass", seed=5)

    assert record["outcome"] == "win"
    assert record["bank_differential"] == 40
    assert record["shed_overflow"] == 2
    assert record["price_floor_sales"] == 3
    # A terminal duplicate hour-23 snapshot is not a day boundary.
    assert record["missed_basic_needs"] == 0


def test_price_floor_sales_uses_sequential_per_unit_quotes():
    from scripts.evaluate import _price_floor_sales

    inventory = 1_717_032_651_332
    observation = {
        "market": {"inventory": {"WHEAT": inventory}, "prices": {"WHEAT": 2}},
        "private": {"shed": {"WHEAT": 2}},
    }
    state = {"action": {"market": [["SELL", "WHEAT", 2]]}}

    assert _price_floor_sales(state, observation) == 1


def test_shed_overflow_uses_pre_transition_shed_and_worker_inventory():
    from scripts.evaluate import replay_record

    before = {
        "player": 0, "step": 23, "hour": 23,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                   "unlocked_quadrants": ["NW"]}],
        "private": {"shed": {"WHEAT": 100}, "inventories": [{"WHEAT": 3}], "seeds": {}},
        "market": {"prices": {}, "inventory": {}},
    }
    after = {
        "player": 0, "step": 24, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                   "unlocked_quadrants": ["NW"]}],
        "private": {"shed": {"WHEAT": 100}, "inventories": [{}], "seeds": {}},
        "market": {"prices": {}, "inventory": {}},
    }
    other = {
        "player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                                  "unlocked_quadrants": ["NW"]}],
        "private": {"shed": {}, "inventories": [{}], "seeds": {}},
        "market": {"prices": {}, "inventory": {}},
    }
    replay = {"steps": [
        [{"observation": before, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
        [{"observation": after, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
    ], "statuses": ["DONE", "DONE"], "info": {}}

    record = replay_record(_engine_envelope(replay), variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is False
    assert record["shed_overflow"] == 3


def test_terminal_hour_23_snapshot_is_not_a_missed_needs_boundary():
    from scripts.evaluate import replay_record

    plant = {"kind": "PLANT", "watered_today": False}
    before = {"player": 0, "step": 23, "hour": 23,
              "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[plant]], "unlocked_quadrants": ["NW"]}],
              "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    after = {"player": 0, "step": 24, "hour": 23,
             "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[plant]], "unlocked_quadrants": ["NW"]}],
             "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    other = {"player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]], "unlocked_quadrants": ["NW"]}],
             "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    replay = {"steps": [
        [{"observation": before, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
        [{"observation": after, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
    ], "statuses": ["DONE", "DONE"], "info": {}}

    assert replay_record(replay, variant="mixed", opponent="pass", seed=1)["missed_basic_needs"] == 0


@pytest.mark.parametrize("unlocked_quadrants", [None, {"NW": True}])
def test_malformed_unlocked_quadrants_is_a_framework_failure(unlocked_quadrants):
    from scripts.evaluate import replay_record

    observation = {"player": 0, "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                                                "unlocked_quadrants": unlocked_quadrants}],
                   "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    other = {"player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]],
                                      "unlocked_quadrants": ["NW"]}],
             "private": {"shed": {}, "inventories": [{}], "seeds": {}}, "market": {"prices": {}, "inventory": {}}}
    replay = {"steps": [[
        {"observation": observation, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
        {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
    ]], "statuses": ["DONE", "DONE"], "info": {}}

    record = replay_record(_engine_envelope(replay), variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_actions_are_validated_against_the_preceding_observation():
    from scripts.evaluate import replay_record

    initial = {
        "player": 0, "step": 0, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {"WHEAT": 1}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }
    post_plant = {
        "player": 0, "step": 1, "hour": 1,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[{"kind": "PLANT", "crop": "WHEAT", "watered_today": False, "yield_units": 0}]]}],
        "private": {"seeds": {"WHEAT": 0}, "shed": {}, "inventories": [{}]},
        "market": initial["market"],
    }
    other_initial = {
        "player": 1, "step": 0, "hour": 0,
        "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": initial["market"],
    }
    other_post = {**other_initial, "step": 1, "hour": 1}
    replay = {
        "steps": [
            [{"observation": initial, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
             {"observation": other_initial, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
            [{"observation": post_plant, "action": {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, "status": "DONE", "info": {}},
             {"observation": other_post, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
        ],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    record = replay_record(_engine_envelope(replay), variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is False
    assert record["outcome"] == "win"


def test_replay_accepts_engine_atomic_noop_for_duplicate_plant_requests():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    for turn in replay["steps"]:
        observation = turn[0]["observation"]
        farm = observation["farms"][0]
        farm["farmer"] = [0, 0]
        farm["hands"] = [[1, 0]]
        farm["tiles"] = [[None for _ in range(4)] for _ in range(4)]
        observation["private"] = {
            "seeds": {"MELON": 1}, "shed": {}, "inventories": [{}, {}],
        }
    replay["steps"][0][0]["action"]["hands"] = [["PASS"]]
    replay["steps"][1][0]["action"] = {
        "farmer": ["PLANT", "MELON"],
        "hands": [["PLANT", "MELON"]],
        "market": [],
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is False


def test_replay_rejects_tampered_plant_without_post_state_effect():
    from scripts.evaluate import replay_record

    initial = {
        "player": 0, "step": 0, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {"WHEAT": 1}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }
    tampered_post = {
        **initial,
        "step": 1,
        "hour": 1,
        "private": {"seeds": {"WHEAT": 1}, "shed": {}, "inventories": [{}]},
    }
    other = {
        "player": 1,
        "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": initial["market"],
    }
    replay = {
        "steps": [
            [{"observation": initial, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
             {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
            [{"observation": tampered_post, "action": {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, "status": "DONE", "info": {}},
             {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
        ],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


@pytest.mark.parametrize("bad_quantity", [1.5, -1, float("nan"), float("inf")])
def test_replay_rejects_invalid_inventory_quantity_for_any_player_snapshot(bad_quantity):
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    replay["steps"][0][1]["observation"]["private"]["shed"] = {"WHEAT": bad_quantity}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True


def test_replay_rejects_unknown_inventory_product():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    replay["steps"][1][0]["observation"]["private"]["inventories"] = [{"NOT_A_PRODUCT": 1}]

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True


@pytest.mark.parametrize("field", ["info", "metadata"])
def test_replay_record_rejects_supplied_seed_mismatch(field):
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    if field == "info":
        replay["info"]["seed"] = 2
    else:
        replay["metadata"]["seed"] = 2

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True


def test_replay_rejects_truncated_done_replay_against_episode_steps():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    replay["configuration"]["episodeSteps"] = 720

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_rejects_opponent_final_money_tampering():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    post = replay["steps"][1][1]["observation"]
    replay["steps"][1][1]["observation"] = {
        **post,
        "farms": [post["farms"][0], {**post["farms"][1], "money": 91}],
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_rejects_player_hour_tampering():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    post = replay["steps"][1][0]["observation"]
    replay["steps"][1][0]["observation"] = {**post, "hour": 2}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_rejects_opponent_hour_tampering():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    post = replay["steps"][1][1]["observation"]
    replay["steps"][1][1]["observation"] = {**post, "hour": 2}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_real_replay_rejects_extra_board_row():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    post = replay["steps"][1][0]["observation"]
    farm = post["farms"][0]
    replay["steps"][1][0]["observation"] = {
        **post, "farms": [{**farm, "tiles": [*farm["tiles"], [None, None, None, None]]}],
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_real_replay_rejects_deleted_target_tile_field():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    animal = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 0,
        "yield_units": 0, "consecutive_unfed": 0, "fed_today": False,
        "cared_today": False, "fertilizer_available": False,
        "pending_care_bonus": 0,
    }
    pre = replay["steps"][0][0]["observation"]
    post = replay["steps"][1][0]["observation"]
    replay["steps"][0][0]["action"] = {"farmer": ["PASS"], "hands": [], "market": []}
    replay["steps"][1][0]["action"] = {"farmer": ["CARE"], "hands": [], "market": []}
    pre_farm = {**pre["farms"][0], "tiles": [[animal]]}
    post_animal = {**animal, "cared_today": True}
    post_animal.pop("pending_care_bonus")
    post_farm = {**post["farms"][0], "tiles": [[post_animal]]}
    replay["steps"][0][0]["observation"] = {**pre, "farms": [pre_farm]}
    replay["steps"][1][0]["observation"] = {**post, "farms": [post_farm]}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_midday_accepts_sequential_same_tile_water_then_fertilize():
    from scripts.evaluate import _midday_board_changes_valid

    before = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "watered_today": False, "consecutive_unwatered": 0,
        "yield_units": 1, "max_lifespan_step": 120,
        "fertilized_until_day": -1,
    }
    after = {**before, "watered_today": True, "fertilized_until_day": 2}
    pre_farm = {
        "money": 100, "farmer": [0, 0], "hands": [[0, 0]], "hires_today": 1,
        "tiles": [[before]], "unlocked_quadrants": ["NW"],
    }
    post_farm = {**pre_farm, "tiles": [[after]]}
    pre = {
        "player": 0, "step": 1, "day": 0, "hour": 1, "farms": [pre_farm],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}, {"FERTILIZER": 1}]},
        "market": {"inventory": {}, "prices": {}},
    }
    post = {**pre, "step": 2, "hour": 2, "farms": [post_farm]}
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _midday_board_changes_valid(
        pre, post, {"farmer": ["WATER"], "hands": [["FERTILIZE"]], "market": []},
        {"boardSize": 1, "turnsPerDay": 24}, market_result,
    )


def test_transition_effects_accepts_same_tile_fertilize_then_consuming_harvest():
    from scripts.evaluate import _transition_effects_valid

    plant = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
        "watered_today": True, "consecutive_unwatered": 0,
        "yield_units": 1, "max_lifespan_step": 120,
        "fertilized_until_day": -1,
    }
    pre_farm = {
        "money": 100, "farmer": [0, 0], "hands": [[0, 0]], "hires_today": 1,
        "tiles": [[plant]], "unlocked_quadrants": ["NW"],
    }
    post_farm = {**pre_farm, "tiles": [[None]]}
    pre = {
        "player": 0, "step": 1, "day": 0, "hour": 1, "farms": [pre_farm],
        "private": {"seeds": {}, "shed": {}, "inventories": [{"FERTILIZER": 1}, {}]},
        "market": {"inventory": {}, "prices": {}},
    }
    post = {
        "player": 0, "step": 2, "day": 0, "hour": 2, "farms": [post_farm],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}, {"WHEAT": 1}]},
        "market": {"inventory": {}, "prices": {}},
    }
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["FERTILIZE"], "hands": [["HARVEST"]], "market": []},
        {"boardSize": 1, "turnsPerDay": 24}, market_result,
    )


def test_malformed_unhashable_replay_crop_is_a_framework_failure_not_an_exception():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    malformed_tile = {"kind": "PLANT", "crop": [], "yield_units": 1, "planted_day": 0}
    replay["steps"][0][0]["observation"]["farms"][0]["tiles"][0][0] = malformed_tile
    replay["steps"][1][0]["action"] = {"farmer": ["HARVEST"], "hands": [], "market": []}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_price_floor_sales_uses_both_players_market_queues():
    from scripts.evaluate import _price_floor_sales

    inventory = 1_717_032_651_333
    own_observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"shed": {"WHEAT": 2}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 1}, "inventory": {"WHEAT": inventory}},
    }
    other_observation = {
        "player": 1,
        "farms": [{"money": 100}, {"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"shed": {}, "inventories": [{}]},
        "market": own_observation["market"],
    }
    own_state = {"action": {"market": [["SELL", "WHEAT", 2]]}}
    other_state = {"action": {"market": [["BUY_PRODUCT", "WHEAT", 1]]}}

    assert _price_floor_sales(
        own_state, own_observation, {},
        other_state=other_state, other_observation=other_observation,
    ) == 1


def test_default_selection_prioritizes_zero_framework_failure_rate():
    from scripts.evaluate import build_result_document

    records = [
        {"variant": "mixed", "opponent": "pass", "outcome": "loss", "final_bank": 10,
         "opponent_final_bank": 20, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "pass", "outcome": "win", "final_bank": 100,
         "opponent_final_bank": 20, "framework_error": False, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
        {"variant": "melon-heavy", "opponent": "starter", "outcome": "framework_error", "final_bank": None,
         "opponent_final_bank": None, "framework_error": True, "shed_overflow": 0,
         "price_floor_sales": 0, "missed_basic_needs": 0},
    ]

    document = build_result_document(
        config={"opponents": ["pass", "starter"], "variants": ["mixed", "melon-heavy"]},
        records=records,
    )

    assert document["selected_default"] == "mixed"


@pytest.mark.parametrize("replay", [
    {"steps": None, "statuses": ["DONE", "DONE"], "info": {}},
    {"steps": [], "statuses": ["DONE", "DONE"], "info": None},
    {"steps": [], "statuses": ["DONE", "DONE"], "info": {}, "configuration": []},
    {"steps": [], "statuses": ["DONE", "DONE"], "info": {}, "metadata": []},
])
def test_malformed_replay_shapes_become_framework_failures(replay):
    from scripts.evaluate import replay_record

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_transition_effects_accept_animal_place_inventory_consumption():
    from scripts.evaluate import _transition_effects_valid

    board = [[None for _ in range(10)] for _ in range(10)]
    board[4][4] = {"kind": "COOP"}
    post_board = [[tile for tile in row] for row in board]
    post_board[4][4] = {"kind": "COOP", "animal": "GOOSE", "yield_units": 0}
    pre = {
        "player": 0, "hour": 5,
        "farms": [{"money": 100, "farmer": [4, 4], "hands": [], "tiles": board, "unlocked_quadrants": ["NW"]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{"GOOSE": 1}]},
        "market": {"inventory": {}, "prices": {}},
    }
    post = {**pre, "farms": [{**pre["farms"][0], "tiles": post_board}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}}
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["PLACE", "GOOSE"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_accepts_same_tile_water_blocked_by_harvest():
    from scripts.evaluate import _transition_effects_valid

    plant = {"kind": "PLANT", "crop": "MELON", "watered_today": False,
             "yield_units": 1, "planted_day": 0, "fertilized_until_day": -1}
    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [[0, 0]],
                "hires_today": 1, "tiles": [[plant]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[None]]}
    pre = {"player": 0, "step": 1, "day": 0, "hour": 1, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}, {}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 2, "day": 0, "hour": 2, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{"MELON": 1}, {}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post,
        {"farmer": ["HARVEST"], "hands": [["WATER"]], "market": []},
        {}, market_result,
    )


def test_transition_effects_accepts_water_followed_by_lifespan_decay():
    from scripts.evaluate import _transition_effects_valid

    plant = {"kind": "PLANT", "crop": "MELON", "watered_today": False,
             "yield_units": 1, "planted_day": 0, "fertilized_until_day": -1,
             "max_lifespan_step": 312, "consecutive_unwatered": 0}
    farm = {"money": 100, "farmer": [3, 0], "hands": [], "hires_today": 0,
            "tiles": [[None, None, None, plant]], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 322, "day": 13, "hour": 10, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 323, "day": 13, "hour": 11,
            "farms": [{**farm, "tiles": [[None, None, None, {"kind": "WEED"}]]}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["WATER"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_accept_end_of_day_hire_hand_reset():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [], "hires_today": 0,
            "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "hour": 23, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "hour": 0, "farms": [{**farm, "money": 99, "hands": [], "hires_today": 0}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 99, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}, {}, market_result,
    )


def test_transition_effects_accepts_hired_hand_disappearance_at_day_boundary():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [[2, 2]], "hires_today": 1,
            "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}, {}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0,
            "farms": [{**farm, "farmer": [4, 4], "hands": [], "hires_today": 0}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}, {}, market_result,
    )


def _boundary_feed_replay_states():
    board = [[None for _ in range(10)] for _ in range(10)]
    board[0][0] = {"kind": "PASTURE", "animal": "COW", "fed_today": False,
                   "cared_today": False, "consecutive_unfed": 1, "yield_units": 0,
                   "fertilizer_available": False, "pending_care_bonus": 0, "placed_day": 0}
    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": board, "unlocked_quadrants": ["NW"]}
    post_board = [[tile for tile in row] for row in board]
    post_board[0][0] = {**board[0][0], "consecutive_unfed": 0,
                        "fertilizer_available": True, "fed_today": False, "cared_today": False}
    post_farm = {**pre_farm, "farmer": [4, 4], "tiles": post_board}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {"WHEAT": 99}, "inventories": [{"WHEAT": 2}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {"WHEAT": 100}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {"WHEAT": 99}, "seeds": {},
                                  "hires": 0, "unlocked": ["NW"]}], "market_inventory": {}}
    return pre, post, market_result


def test_transition_effects_accepts_boundary_feed_consumption_and_drop():
    from scripts.evaluate import _transition_effects_valid

    pre, post, market_result = _boundary_feed_replay_states()

    assert _transition_effects_valid(
        pre, post, {"farmer": ["FEED"], "hands": [], "market": []}, {}, market_result,
    )


@pytest.mark.parametrize("tamper", ["shed", "farmer"])
def test_transition_effects_rejects_boundary_inventory_or_farmer_reset_tampering(tamper):
    from scripts.evaluate import _transition_effects_valid

    pre, post, market_result = _boundary_feed_replay_states()
    if tamper == "shed":
        post["private"]["shed"]["WHEAT"] = 99
    else:
        post["farms"][0]["farmer"] = [0, 0]

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["FEED"], "hands": [], "market": []}, {}, market_result,
    )


@pytest.mark.parametrize("worker_field", ["farmer", "hands"])
def test_transition_effects_rejects_boolean_worker_coordinates(worker_field):
    from scripts.evaluate import _transition_effects_valid

    pre, post, market_result = _boundary_feed_replay_states()
    if worker_field == "farmer":
        pre["farms"][0][worker_field] = [True, 0]
        action = {"farmer": ["FEED"], "hands": [], "market": []}
    else:
        pre["farms"][0][worker_field] = [[True, 0]]
        pre["private"]["inventories"].append({})
        action = {"farmer": ["FEED"], "hands": [["PASS"]], "market": []}

    assert not _transition_effects_valid(pre, post, action, {}, market_result)


def test_transition_effects_accepts_midday_hire_spawn_and_new_inventory():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [], "hires_today": 0,
            "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 0, "day": 0, "hour": 0, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 1, "day": 0, "hour": 1,
            "farms": [{**farm, "money": 99, "hands": [[5, 4]], "hires_today": 1}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}, {}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 99, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
        {"boardSize": 10}, market_result,
    )


def test_transition_effects_rejects_invalid_midday_hire_spawn():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [], "hires_today": 0,
            "tiles": [[None for _ in range(10)] for _ in range(10)], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 0, "day": 0, "hour": 0, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 1, "day": 0, "hour": 1,
            "farms": [{**farm, "money": 99, "hands": [[0, 0]], "hires_today": 1}],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}, {}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 99, "shed": {}, "seeds": {}, "hires": 1, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
        {"boardSize": 10}, market_result,
    )


def test_transition_effects_rejects_unrelated_board_mutation():
    from scripts.evaluate import _transition_effects_valid

    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": [[None, None], [None, None]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[None, {"kind": "COOP"}], [None, None]]}
    pre = {"player": 0, "step": 0, "hour": 0, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 1, "hour": 1, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_rejects_unrelated_end_of_day_board_mutation():
    from scripts.evaluate import _transition_effects_valid

    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": [[None, None], [None, None]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[None, {"kind": "COOP"}], [None, None]], "farmer": [0, 0], "hands": []}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_rejects_boundary_water_without_refresh_effect():
    from scripts.evaluate import _transition_effects_valid

    plant = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False,
             "consecutive_unwatered": 1, "yield_units": 1}
    farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
            "tiles": [[plant]], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["WATER"], "hands": [], "market": []}, {}, market_result,
    )


@pytest.mark.parametrize("operation", ["WATER", "FEED", "CARE"])
def test_transition_effects_rejects_tampered_boundary_target_tile(operation):
    from scripts.evaluate import _transition_effects_valid

    if operation == "WATER":
        before_tile = {"kind": "PLANT", "crop": "WHEAT", "watered_today": False,
                       "consecutive_unwatered": 1, "yield_units": 1,
                       "max_lifespan_step": 120, "fertilized_until_day": -1, "planted_day": -2}
        # The action waters WHEAT, then the day refresh clears the flag.  The
        # crop mutation is unrelated and must not be hidden by that reset.
        after_tile = {**before_tile, "crop": "CARROT", "watered_today": False,
                      "consecutive_unwatered": 0, "yield_units": 2}
    else:
        before_tile = {"kind": "PASTURE", "animal": "COW", "fed_today": False,
                       "cared_today": False, "consecutive_unfed": 1, "yield_units": 0,
                       "fertilizer_available": False, "placed_day": 0}
        after_tile = {**before_tile, "animal": "SHEEP", "consecutive_unfed": 0,
                      "fertilizer_available": True, "fed_today": False,
                      "cared_today": False}

    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": [[before_tile]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[after_tile]]}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{"WHEAT": 1}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": [operation], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_rejects_boundary_plant_field_tampering():
    from scripts.evaluate import _transition_effects_valid

    plant_before = {"kind": "PLANT", "crop": "WHEAT", "watered_today": True,
                    "consecutive_unwatered": 1, "yield_units": 1,
                    "max_lifespan_step": 120, "fertilized_until_day": -1, "planted_day": 0}
    plant_after = {**plant_before, "yield_units": 99}
    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": [[plant_before]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[plant_after]]}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_rejects_boundary_animal_removal_without_refresh_rule():
    from scripts.evaluate import _transition_effects_valid

    animal_before = {"kind": "PASTURE", "animal": "COW", "fed_today": True,
                     "cared_today": False, "consecutive_unfed": 0, "yield_units": 0,
                     "fertilizer_available": False, "placed_day": 0}
    animal_after = {"kind": "PASTURE"}
    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": [[animal_before]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[animal_after]]}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_accepts_boundary_refresh_growth_and_escape():
    from scripts.evaluate import _transition_effects_valid

    plant_before = {"kind": "PLANT", "crop": "TOMATO", "watered_today": True,
                    "consecutive_unwatered": 1, "yield_units": 0,
                    "max_lifespan_step": -1, "fertilized_until_day": -1, "planted_day": -7}
    plant_after = {**plant_before, "watered_today": False, "consecutive_unwatered": 0, "yield_units": 1}
    animal_before = {"kind": "PASTURE", "animal": "COW", "fed_today": False,
                     "cared_today": False, "consecutive_unfed": 1, "yield_units": 0,
                     "fertilizer_available": False, "placed_day": 0}
    animal_after = {"kind": "PASTURE"}
    pre_farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                "tiles": [[plant_before, animal_before]], "unlocked_quadrants": ["NW"]}
    post_farm = {**pre_farm, "tiles": [[plant_after, animal_after]]}
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23, "farms": [pre_farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0, "farms": [post_farm],
            "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"inventory": {}, "prices": {}}}
    market_result = {"states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {}}

    assert _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_end_of_day_refresh_accepts_newly_unlocked_locked_tile_becoming_weed():
    from scripts.evaluate import _midday_board_changes_valid

    before_tiles = [[None for _ in range(6)] for _ in range(6)]
    after_tiles = [[None for _ in range(6)] for _ in range(6)]
    before_tiles[0][4] = "LOCKED"
    after_tiles[0][4] = {"kind": "WEED"}
    before_farm = {
        "money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
        "tiles": before_tiles, "unlocked_quadrants": ["NW"],
    }
    after_farm = {
        **before_farm, "tiles": after_tiles,
        "unlocked_quadrants": ["NW", "NE"],
    }
    pre = {"player": 0, "step": 23, "day": 0, "hour": 23,
           "farms": [before_farm], "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 24, "day": 1, "hour": 0,
            "farms": [after_farm], "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0,
                     "unlocked": ["NW", "NE"]}],
        "market_inventory": {},
    }

    assert _midday_board_changes_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_midday_care_rejects_unrelated_target_tile_field_tampering():
    from scripts.evaluate import _midday_board_changes_valid

    before_tile = {
        "kind": "COOP", "animal": "GOOSE", "fed_today": True,
        "cared_today": False, "consecutive_unfed": 0, "yield_units": 1,
        "fertilizer_available": False, "placed_day": 0,
        "pending_care_bonus": 0,
    }
    after_tile = {**before_tile, "cared_today": True, "yield_units": 99}
    before_farm = {
        "money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
        "tiles": [[before_tile]], "unlocked_quadrants": ["NW"],
    }
    after_farm = {**before_farm, "tiles": [[after_tile]]}
    pre = {"player": 0, "step": 1, "day": 0, "hour": 1,
           "farms": [before_farm], "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "step": 2, "day": 0, "hour": 2,
            "farms": [after_farm], "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 100, "shed": {}, "seeds": {}, "hires": 0,
                     "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert not _midday_board_changes_valid(
        pre, post, {"farmer": ["CARE"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_treats_unhashable_unlock_metadata_as_invalid():
    from scripts.evaluate import _midday_board_changes_valid

    farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
            "tiles": [[None]], "unlocked_quadrants": [{}]}
    pre = {"player": 0, "step": 1, "hour": 1, "farms": [farm], "private": {}, "market": {}}
    post = {"player": 0, "step": 2, "hour": 2, "farms": [farm], "private": {}, "market": {}}
    market_result = {"market_inventory": {}, "states": [{"unlocked": [{}]}]}

    assert not _midday_board_changes_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": []}, {}, market_result,
    )


def test_transition_effects_rejects_unexpected_post_market_inventory():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
            "tiles": [[None]], "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "step": 1, "hour": 1, "farms": [farm],
           "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
           "market": {"inventory": {"WHEAT": 10}, "prices": {"WHEAT": 25}}}
    post = {"player": 0, "step": 2, "hour": 2, "farms": [{**farm, "money": 75}],
            "private": {"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{}]},
            "market": {"inventory": {"WHEAT": 8}, "prices": {"WHEAT": 25}}}
    market_result = {"states": [{"money": 75, "shed": {"WHEAT": 1}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
                     "market_inventory": {"WHEAT": 9}}

    assert not _transition_effects_valid(
        pre, post, {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 1]]},
        {"townShopSellInterval": 99, "townCenterSellInterval": 99}, market_result,
    )


def test_replay_requires_configuration_and_engine_provenance():
    from scripts.evaluate import replay_record

    observation = {
        "player": 0, "step": 0, "day": 0, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                   "tiles": [[None]], "unlocked_quadrants": ["NW"]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }
    other = {**observation, "player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [],
                                                        "hires_today": 0, "tiles": [[None]], "unlocked_quadrants": ["NW"]}]}
    replay = {"steps": [[
        {"observation": observation, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
        {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
    ]], "statuses": ["DONE", "DONE"], "info": {}}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_rejects_final_carried_inventory_even_when_transition_is_consistent():
    from scripts.evaluate import replay_record

    pre = {
        "player": 0, "step": 0, "day": 0, "hour": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "hires_today": 0,
                   "tiles": [[None]], "unlocked_quadrants": ["NW"]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{"WHEAT": 1}]},
        "market": {"prices": {}, "inventory": {}},
    }
    post = {**pre, "step": 1, "day": 0, "hour": 1}
    other_pre = {
        "player": 1, "step": 0, "day": 0, "hour": 0,
        "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "hires_today": 0,
                   "tiles": [[None]], "unlocked_quadrants": ["NW"]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }
    other_post = {**other_pre, "step": 1, "day": 0, "hour": 1}
    replay = {
        "steps": [
            [{"observation": pre, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
             {"observation": other_pre, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
            [{"observation": post, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
             {"observation": other_post, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
        ],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    record = replay_record(_engine_envelope(replay), variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


@pytest.mark.parametrize(
    ("item", "framework_error"),
    [("WHEAT", True), ("FERTILIZER", False)],
)
def test_complete_season_replay_validates_terminal_shed_liquidation(item, framework_error):
    from scripts.evaluate import replay_record

    replay = _full_season_replay_with_terminal_shed(item)

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is framework_error


def test_transition_effects_accept_drop_capacity_discard():
    from scripts.evaluate import _transition_effects_valid

    farm = {"money": 100, "farmer": [4, 4], "hands": [], "tiles": [[None for _ in range(10)] for _ in range(10)],
            "unlocked_quadrants": ["NW"]}
    pre = {"player": 0, "hour": 5, "farms": [farm],
           "private": {"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{"CARROT": 3}]},
           "market": {"inventory": {}, "prices": {}}}
    post = {"player": 0, "hour": 6, "farms": [farm],
            "private": {"seeds": {}, "shed": {"WHEAT": 1, "CARROT": 1}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}}}
    market_result = {
        "states": [{"money": 100, "shed": {"WHEAT": 1}, "seeds": {}, "hires": 0, "unlocked": ["NW"]}],
        "market_inventory": {},
    }

    assert _transition_effects_valid(
        pre, post, {"farmer": ["DROP"], "hands": [], "market": []}, {"shedCapacity": 2}, market_result,
    )


def test_invalid_or_missing_replay_states_are_framework_failures():
    from scripts.evaluate import aggregate_records, replay_record

    replay = {
        "steps": [[
            {"observation": {"player": 0, "farms": [{"money": 100}]}, "action": {}, "status": "DONE", "info": {}},
        ]],
        "statuses": ["DONE", "DONE"],
        "info": {},
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)
    summary = aggregate_records([record])

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"
    assert summary["framework_failures"] == 1
    assert summary["wins"] == summary["losses"] == summary["ties"] == 0


def test_replay_missing_a_player_state_is_a_framework_failure():
    from scripts.evaluate import replay_record

    class TruthyEmptySteps(list):
        def __bool__(self):
            return True

    replay = _engine_envelope({
        "steps": TruthyEmptySteps(),
        "statuses": ["DONE", "DONE"],
        "info": {},
    })

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_top_level_failure_status_is_counted_even_with_complete_states():
    from scripts.evaluate import replay_record

    states = [
        {"observation": {"player": player, "farms": [{"money": 100}, {"money": 90}]},
         "action": {}, "status": "DONE", "info": {}}
        for player in [0, 1]
    ]
    record = replay_record({"steps": [states], "statuses": ["ERROR", "DONE"], "info": {}},
                           variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_rejects_invalid_command_and_bad_worker_count():
    from scripts.evaluate import replay_record

    observation = {
        "player": 0, "farms": [{"money": 100, "hands": [], "tiles": [[None]]}, {"money": 90, "hands": [], "tiles": [[None]]}],
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
        "private": {"shed": {}, "seeds": {}},
    }
    bad_action = {"farmer": ["NOT_REAL"], "hands": [], "market": []}
    good_other = {"player": 1, "farms": [{"money": 100}, {"money": 90}], "market": observation["market"]}
    replay = {
        "steps": [[
            {"observation": observation, "action": bad_action, "status": "DONE", "info": {}},
            {"observation": good_other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}},
        ]],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_replay_rejects_market_orders_that_cannot_execute_from_prior_state():
    from scripts.evaluate import _valid_action_schema

    observation = {
        "player": 0,
        "farms": [{"money": 0, "farmer": [0, 0], "hands": [], "hires_today": 0,
                   "unlocked_quadrants": ["NW"], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }
    prefix = {"farmer": ["PASS"], "hands": []}

    assert not _valid_action_schema({**prefix, "market": [["SELL", "WHEAT", 1]]}, observation)
    assert not _valid_action_schema({**prefix, "market": [["BUY_SEED", "WHEAT", 1]]}, observation)


def test_malformed_market_order_limit_is_a_framework_failure_not_an_exception():
    from scripts.evaluate import _valid_action_schema

    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }

    assert not _valid_action_schema(
        {"farmer": ["PASS"], "hands": [], "market": []},
        observation,
        {"maxMarketOrdersPerTurn": "not-a-number"},
    )


def test_boundary_needs_use_post_action_state_when_available():
    from scripts.evaluate import replay_record

    before = {
        "player": 0, "step": 23, "hour": 23,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[{"kind": "PLANT", "watered_today": False}]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }
    after = {
        "player": 0, "step": 24, "hour": 23,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[{"kind": "PLANT", "watered_today": True}]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }
    other = {"player": 1, "farms": [{"money": 90}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
             "private": {"seeds": {}, "shed": {}, "inventories": [{}]}, "market": {"prices": {}, "inventory": {}}}
    replay = {
        "steps": [
            [{"observation": before, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
             {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
            [{"observation": after, "action": {"farmer": ["WATER"], "hands": [], "market": []}, "status": "DONE", "info": {}},
             {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
        ],
        "statuses": ["DONE", "DONE"], "info": {},
    }

    assert replay_record(replay, variant="mixed", opponent="pass", seed=1)["missed_basic_needs"] == 0


def test_variants_and_ablations_change_only_legal_action_shapes():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {"MELON": 1}},
        "market": {"prices": {"MELON": 250, "WHEAT": 25}},
    }
    action = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}

    variant_action = apply_variant(action, observation, "melon-heavy")
    ablated_action = apply_variant(
        {"farmer": ["EAST"], "hands": [], "market": [["BUY_LAND"]]},
        observation,
        "mixed",
        {"route_scheduling": False, "market_batch_sizing": True, "shop_adaptation": True,
         "land_purchase": False, "animals": True},
    )

    assert variant_action["farmer"] == ["PLANT", "MELON"]
    assert variant_action["market"] == [["BUY_SEED", "MELON", 1]]
    assert ablated_action == {"farmer": ["PASS"], "hands": [], "market": []}


@pytest.mark.parametrize("order", [[], ()])
def test_apply_variant_skips_empty_market_order(order):
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
    }

    result = apply_variant(
        {"farmer": ["PASS"], "hands": [], "market": [order]},
        observation,
        "conservative",
    )

    assert result["market"] == []


@pytest.mark.parametrize(
    "order",
    [
        None,
        "BOGUS",
        ["BOGUS"],
        ["BUY_PRODUCT"],
        ["BUY_PRODUCT", "WHEAT", 0],
    ],
)
def test_apply_variant_preserves_nonempty_malformed_market_order_as_framework_error(order):
    from scripts.evaluate import apply_variant, replay_record

    replay = _strict_two_turn_replay()
    observation = replay["steps"][0][0]["observation"]

    result = apply_variant(
        {"farmer": ["PASS"], "hands": [], "market": [order]},
        observation,
        "mixed",
        configuration=replay["configuration"],
    )
    replay["steps"][0][0]["action"] = result
    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert result["market"] == [order]
    assert record["framework_error"] is True
    assert record["outcome"] == "framework_error"


def test_conservative_variant_preserves_mandatory_wheat_and_fertilizer_orders():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 200, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}},
        "market": {
            "prices": {"WHEAT": 25, "FERTILIZER": 100, "CARROT": 35},
            "inventory": {"WHEAT": 10_000, "FERTILIZER": 10_000},
        },
    }
    action = {
        "farmer": ["PASS"], "hands": [],
        "market": [
            ["BUY_PRODUCT", "WHEAT", 1],
            ["BUY_PRODUCT", "FERTILIZER", 1],
            ["BUY_SEED", "CARROT", 1],
            ["BUY_ANIMAL", "GOOSE", 1],
        ],
    }

    result = apply_variant(action, observation, "conservative")

    assert result["market"] == [
        ["BUY_PRODUCT", "WHEAT", 1],
        ["BUY_PRODUCT", "FERTILIZER", 1],
    ]


def test_conservative_variant_preserves_terminal_liquidation_orders():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "day": 29,
        "hour": 22,
        "farms": [{"money": 200, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {
            "seeds": {},
            "shed": {"WHEAT": 20, "STRAWBERRY": 2},
            "inventories": [{}],
        },
        "market": {
            "prices": {"WHEAT": 25, "STRAWBERRY": 100},
            "inventory": {"WHEAT": 10_000, "STRAWBERRY": 10_000},
        },
    }
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "WHEAT", 20], ["SELL", "STRAWBERRY", 2]],
    }

    result = apply_variant(action, observation, "conservative")

    assert result["market"] == action["market"]


def test_conservative_variant_keeps_deadline_hire_from_raw_engine_observation():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "day": 27,
        "hour": 0,
        "farms": [{
            "money": 1_000,
            "farmer": [0, 0],
            "hands": [],
            "hires_today": 0,
            "tiles": [[{"kind": "PLANT", "crop": "STRAWBERRY", "watered_today": False}]],
        }],
        "private": {"seeds": {}, "shed": {}},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10_000}},
    }

    result = apply_variant(
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"], ["HIRE"], ["BUY_PRODUCT", "WHEAT", 1]]},
        observation,
        "conservative",
    )

    assert result["market"][:2] == [["HIRE"], ["BUY_PRODUCT", "WHEAT", 1]]


def test_route_scheduling_ablation_safely_inspects_nested_hand_commands():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [[0, 0]], "tiles": [[None]]}],
        "private": {"seeds": {}, "inventories": [{}, {}]},
        "market": {"prices": {}, "inventory": {}},
    }
    action = {"farmer": ["PASS"], "hands": [["EAST"]], "market": []}

    result = apply_variant(
        action, observation, "mixed",
        {"route_scheduling": False, "market_batch_sizing": True, "shop_adaptation": True,
         "land_purchase": True, "animals": True},
    )

    assert result["hands"] == [["PASS"]]


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_route_scheduling_ablation_produces_a_valid_replay():
    from scripts.evaluate import run_game

    record = run_game(
        variant="mixed", opponent="pass", seed=2, steps=96,
        ablations={"route_scheduling": False, "market_batch_sizing": True,
                   "shop_adaptation": True, "land_purchase": True, "animals": True},
    )

    assert record["framework_error"] is False


def test_demand_reactive_preserves_needs_safe_planner_schedule():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {"WHEAT": 1}},
        "market": {"prices": {"WHEAT": 25, "MELON": 250}, "inventory": {}},
    }
    action = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}

    assert apply_variant(action, observation, "demand-reactive") == apply_variant(action, observation, "mixed")


def test_malformed_unit_args_are_a_framework_failure_not_an_exception():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    replay["steps"][0][0]["action"]["farmer"] = ["PLANT", []]

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True


def test_real_replay_requires_every_intermediate_inventory_snapshot():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    del replay["steps"][1][1]["observation"]["private"]["inventories"]

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True


def test_variant_postprocessing_caps_market_orders_and_preserves_affordability():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 80, "farmer": [0, 0], "hands": [], "tiles": [[None]]}],
        "private": {"seeds": {}},
        "market": {"prices": {"MELON": 250, "WHEAT": 25}},
    }
    action = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]] * 11}

    result = apply_variant(action, observation, "melon-heavy")

    assert len(result["market"]) <= 10
    assert sum({"WHEAT": 10, "MELON": 80}[order[1]] * order[2] for order in result["market"]) <= 80


def test_market_sanitization_obeys_hire_land_and_shed_rules():
    from scripts.evaluate import apply_variant

    base = {
        "player": 0,
        "farms": [{"money": 2003, "farmer": [0, 0], "hands": [], "hires_today": 3, "unlocked_quadrants": ["NW"], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {"WHEAT": 99}},
        "market": {"prices": {"WHEAT": 25}, "inventory": {"WHEAT": 10000}},
    }
    action = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["BUY_LAND"], ["BUY_PRODUCT", "WHEAT", 3]]}

    result = apply_variant(action, base, "mixed", configuration={"shedCapacity": 100, "maxMarketOrdersPerTurn": 10})

    assert result["market"] == [["HIRE"], ["BUY_LAND"], ["BUY_PRODUCT", "WHEAT", 1]]


def test_buy_land_uses_next_unlockable_land_and_cost():
    from scripts.evaluate import apply_variant

    observation = {
        "player": 0,
        "farms": [{"money": 1999, "farmer": [0, 0], "hands": [], "hires_today": 0, "unlocked_quadrants": ["NW", "NE"], "tiles": [[None]]}],
        "private": {"seeds": {}, "shed": {}},
        "market": {"prices": {}, "inventory": {}},
    }

    result = apply_variant({"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]}, observation, "mixed")

    assert result["market"] == []


def test_pickup_and_place_commands_are_set_compatible_and_state_aware():
    from scripts.evaluate import apply_variant

    empty_board = [[None for _ in range(10)] for _ in range(10)]
    observation = {
        "player": 0,
        "farms": [{"money": 100, "farmer": [4, 4], "hands": [], "tiles": empty_board}],
        "private": {"seeds": {}, "shed": {"WHEAT": 1}, "inventories": [{"WHEAT": 1}]},
        "market": {"prices": {}, "inventory": {}},
    }

    pickup = apply_variant({"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}, observation, "mixed")
    place = apply_variant({"farmer": ["PLACE", "WHEAT", 1], "hands": [], "market": []}, observation, "mixed")
    observation["farms"][0]["farmer"] = [0, 0]
    invalid_pickup = apply_variant({"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}, observation, "mixed")
    observation["private"]["inventories"] = [{}]
    invalid_place = apply_variant({"farmer": ["PLACE", "WHEAT", 1], "hands": [], "market": []}, observation, "mixed")

    assert pickup["farmer"] == ["PICKUP", "WHEAT", 1]
    assert place["farmer"] == ["PLACE", "WHEAT", 1]
    assert invalid_pickup["farmer"] == ["PASS"]
    assert invalid_place["farmer"] == ["PASS"]


def test_state_aware_unit_sanitization_rejects_locked_occupied_and_failed_preconditions():
    from scripts.evaluate import apply_variant

    def observation(tile, *, seeds=None, inventory=None, farmer=(0, 0)):
        board = [[None for _ in range(3)] for _ in range(3)]
        board[farmer[1]][farmer[0]] = tile
        return {
            "player": 0,
            "farms": [{"money": 100, "farmer": list(farmer), "hands": [], "tiles": board}],
            "private": {"seeds": seeds or {}, "shed": {}, "inventories": [inventory or {}]},
            "market": {"prices": {}, "inventory": {}},
        }

    assert apply_variant({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, observation("LOCKED", seeds={"WHEAT": 1}), "mixed")["farmer"] == ["PASS"]
    assert apply_variant({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, observation({"kind": "PLANT", "crop": "WHEAT"}, seeds={"WHEAT": 1}), "mixed")["farmer"] == ["PASS"]
    assert apply_variant({"farmer": ["BUILD_COOP"], "hands": [], "market": []}, observation({"kind": "COOP"}), "mixed")["farmer"] == ["PASS"]
    assert apply_variant({"farmer": ["WATER"], "hands": [], "market": []}, observation(None), "mixed")["farmer"] == ["PASS"]
    animal = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": False}
    assert apply_variant({"farmer": ["FEED"], "hands": [], "market": []}, observation(animal), "mixed")["farmer"] == ["PASS"]
    assert apply_variant({"farmer": ["CARE"], "hands": [], "market": []}, observation(animal), "mixed")["farmer"] == ["CARE"]


def test_final_boundary_targeted_feed_without_wheat_is_still_missed():
    from scripts.evaluate import replay_record

    tiles = [[None for _ in range(3)] for _ in range(3)]
    tiles[0][0] = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": True}
    before = {"player": 0, "step": 23, "hour": 23, "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": tiles}],
              "private": {"shed": {}, "seeds": {}, "inventories": [{}]}, "market": {"prices": {}, "inventory": {}}}
    after = {"player": 0, "step": 24, "hour": 0, "farms": [{"money": 100, "farmer": [0, 0], "hands": [], "tiles": tiles}],
             "private": {"shed": {}, "seeds": {}, "inventories": [{}]}, "market": {"prices": {}, "inventory": {}}}
    other = {"player": 1, "farms": [{"money": 100}, {"money": 90, "farmer": [0, 0], "hands": [], "tiles": [[None for _ in range(3)] for _ in range(3)]}], "private": {"inventories": [{}]}, "market": {"prices": {}, "inventory": {}}}
    replay = {"steps": [
        [{"observation": before, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "ACTIVE", "info": {}}],
        [{"observation": after, "action": {"farmer": ["FEED"], "hands": [], "market": []}, "status": "DONE", "info": {}},
         {"observation": other, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "status": "DONE", "info": {}}],
    ], "statuses": ["DONE", "DONE"], "info": {}}

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["missed_basic_needs"] == 1


def test_missed_basic_needs_excludes_optional_care_bonus():
    from scripts.evaluate import _missed_needs_at_boundary

    animal = {"kind": "PASTURE", "animal": "COW", "fed_today": True, "cared_today": False}
    farm = {"farmer": [0, 0], "hands": [], "tiles": [[animal]]}
    observation = {"player": 0, "hour": 23, "farms": [farm]}
    post = {"player": 0, "hour": 0, "farms": [farm]}
    action = {"farmer": ["PASS"], "hands": [], "market": []}

    assert _missed_needs_at_boundary(observation, True, post, {"action": action}, {}) == 0


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_same_seed_and_deterministic_opponent_produce_same_record():
    from scripts.evaluate import run_game

    first = run_game(variant="mixed", opponent="pass", seed=17, steps=8)
    second = run_game(variant="mixed", opponent="pass", seed=17, steps=8)

    assert first == second


@pytest.mark.skipif(make is None, reason="local engine dependency is unavailable")
def test_same_seed_random_style_evaluator_is_reproducible():
    from scripts.evaluate import run_game

    first = run_game(variant="mixed", opponent="random", seed=17, steps=8)
    second = run_game(variant="mixed", opponent="random", seed=17, steps=8)

    assert first == second
    assert first["framework_error"] is False


def _metric_action_state(step, orders):
    return {
        "observation": {"step": step},
        "action": {"farmer": ["PASS"], "hands": [], "market": orders},
    }


def test_replay_market_metrics_count_transactions_and_short_window_churn():
    from scripts.evaluate import _market_metrics

    states = [
        _metric_action_state(0, []),  # bootstrap action is not evaluated
        _metric_action_state(1, [["BUY_PRODUCT", "WHEAT", 2]]),
        _metric_action_state(2, [["SELL", "WHEAT", 1]]),
        _metric_action_state(8, [["SELL", "MELON", 1]]),
    ]

    metrics = _market_metrics(states, churn_window=2)

    assert metrics == {
        "submitted_market_order_count": 3,
        "market_transaction_count": 3,
        "same_item_market_churn": 1,
        "same_item_sell_buy_churn": 1,
    }


def test_replay_record_exposes_terminal_cash_and_inventory_value():
    from scripts.evaluate import replay_record

    replay = _strict_two_turn_replay()
    clean_record = replay_record(_strict_two_turn_replay(), variant="mixed", opponent="pass", seed=1)
    assert clean_record["framework_error"] is False
    assert clean_record["framework_error_reasons"] == []

    terminal = replay["steps"][-1][0]["observation"]
    terminal["farms"][0]["money"] = 125
    terminal["private"]["shed"] = {"WHEAT": 2}
    terminal["market"]["prices"] = {"WHEAT": 25}
    replay["rewards"][0] = 125

    record = replay_record(replay, variant="mixed", opponent="pass", seed=1)

    assert record["terminal_cash"] == 125
    assert record["terminal_inventory_value"] == 50
    assert record["final_bank"] == record["terminal_cash"]


def test_malformed_replay_record_reports_malformed_replay_reason():
    from scripts.evaluate import replay_record

    record = replay_record({}, variant="mixed", opponent="pass", seed=1)

    assert record["framework_error"] is True
    assert record["framework_error_reasons"] == ["malformed_replay"]


def test_framework_error_reasons_are_ordered_and_set_framework_error():
    from scripts.evaluate import framework_error_reasons

    record = {
        "framework_error": False,
        "missed_basic_needs": 1,
        "same_item_market_churn": 1,
    }

    reasons = framework_error_reasons(record)
    record["framework_error_reasons"] = reasons
    record["framework_error"] = bool(reasons)

    assert record["framework_error"] is True
    assert record["framework_error_reasons"] == ["missed_basic_needs", "market_churn"]


def test_clean_legacy_record_has_no_framework_error_reasons():
    from scripts.evaluate import framework_error_reasons

    record = {
        "framework_error": False,
        "missed_basic_needs": 0,
        "same_item_market_churn": 0,
    }

    assert framework_error_reasons(record) == []
    assert record["framework_error"] is False


def test_aggregate_exposes_market_and_terminal_metrics():
    from scripts.evaluate import aggregate_records

    summary = aggregate_records([
        {"outcome": "win", "final_bank": 100, "terminal_cash": 100,
         "terminal_inventory_value": 30, "market_transaction_count": 4,
         "same_item_market_churn": 1, "same_item_sell_buy_churn": 1,
         "framework_error": False},
        {"outcome": "tie", "final_bank": 80, "terminal_cash": 80,
         "terminal_inventory_value": 10, "market_transaction_count": 2,
         "same_item_market_churn": 0, "same_item_sell_buy_churn": 0,
         "framework_error": False},
    ])

    assert summary["total_market_transaction_count"] == 6
    assert summary["mean_market_transaction_count"] == 3
    assert summary["total_same_item_market_churn"] == 1
    assert summary["max_same_item_market_churn"] == 1
    assert summary["mean_terminal_cash"] == 90
    assert summary["mean_terminal_inventory_value"] == 20


def test_promotion_decision_discards_candidate_over_market_activity_caps():
    from scripts.evaluate import promotion_decision

    candidate = [
        {**_metric_record(seat=seat, seed=1, outcome="win", differential=10),
         "market_transaction_count": 1, "same_item_market_churn": 2,
         "same_item_sell_buy_churn": 2, "terminal_cash": 90,
         "terminal_inventory_value": 5}
        for seat in (0, 1)
    ]

    assert promotion_decision(
        candidate, candidate, min_valid_games=1, max_same_item_market_churn=1,
    )["reasons"] == ["same_item_market_churn"]
    assert promotion_decision(
        candidate, candidate, min_valid_games=1, max_same_item_market_churn=10,
        max_market_transactions=1,
    )["reasons"] == ["market_transaction_count"]
    assert promotion_decision(
        candidate, candidate, min_valid_games=1, max_same_item_market_churn=10,
        min_terminal_cash=100,
    )["reasons"] == ["terminal_cash_below_threshold"]
    assert promotion_decision(
        candidate, candidate, min_valid_games=1, max_same_item_market_churn=10,
        min_terminal_inventory_value=10,
    )["reasons"] == ["terminal_inventory_value_below_threshold"]


def test_legacy_records_apply_terminal_gates_to_each_record():
    from scripts.evaluate import promotion_decision

    candidate = [
        {
            "variant": "mixed", "opponent": "pass", "seed": seed,
            "outcome": "win", "final_bank": terminal_cash,
            "opponent_final_bank": 100.0,
            "bank_differential": terminal_cash - 100.0,
            "framework_error": False,
            "terminal_cash": terminal_cash,
            "terminal_inventory_value": terminal_inventory_value,
        }
        for seed, terminal_cash, terminal_inventory_value in (
            (1, 200.0, 200.0), (2, 0.0, 0.0),
        )
    ]

    decision = promotion_decision(
        candidate, candidate, min_valid_games=1,
        min_terminal_cash=100.0, min_terminal_inventory_value=100.0,
    )

    assert decision["reasons"] == ["terminal_cash_below_threshold"]

    inventory_decision = promotion_decision(
        candidate, candidate, min_valid_games=1,
        min_terminal_inventory_value=100.0,
    )

    assert inventory_decision["reasons"] == ["terminal_inventory_value_below_threshold"]


def test_external_metric_deltas_are_suppressed_when_candidate_fails_safety_gates():
    from scripts.evaluate import promotion_decision

    candidate = [
        {**_metric_record(seat=seat, seed=1, outcome="win", differential=10),
         "terminal_cash": 0.0}
        for seat in (0, 1)
    ]
    baseline = [
        {**_metric_record(seat=seat, seed=1, candidate="previous-agent",
                          outcome="loss", differential=1),
         "terminal_cash": 100.0}
        for seat in (0, 1)
    ]

    decision = promotion_decision(
        candidate, baseline, min_valid_games=1, min_terminal_cash=100.0,
        baseline_policy="previous-agent",
    )

    assert decision["reasons"] == ["terminal_cash_below_threshold"]
    assert decision["paired_metric_deltas"] is None


def test_external_baseline_policy_is_run_on_the_same_matrix(monkeypatch):
    from scripts.evaluate import run_evaluation

    calls = []

    def fake_run_matrix(**kwargs):
        calls.append(kwargs)
        return {"records": []}

    monkeypatch.setattr("scripts.evaluate.run_matrix", fake_run_matrix)
    result = run_evaluation(
        candidates=["mixed"], opponents=["pass"], seeds=[4, 9], steps=8,
        seats=[0, 1], baseline_policy="/tmp/previous_agent.py:agent",
        baseline_identity="previous-agent",
    )

    assert calls[0]["policy_path"] == "/tmp/previous_agent.py:agent"
    assert calls[0]["policy_identity"] == "previous-agent"
    assert calls[0]["opponents"] == calls[1]["opponents"]
    assert calls[0]["seeds"] == calls[1]["seeds"]
    assert calls[0]["seats"] == calls[1]["seats"]
    assert result["baseline_policy"] == {
        "identity": "previous-agent", "path": "/tmp/previous_agent.py:agent",
    }


def test_run_matrix_forwards_configured_churn_window(monkeypatch):
    from scripts.evaluate import run_matrix

    calls = []

    def fake_run_game(**kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr("scripts.evaluate.run_game", fake_run_game)
    run_matrix(variants=["mixed"], opponents=["pass"], seeds=[1], steps=2,
                seats=[0], churn_window=1)

    assert calls[0]["churn_window"] == 1


def test_run_evaluation_forwards_configured_churn_window_to_every_matrix(monkeypatch):
    from scripts.evaluate import run_evaluation

    calls = []

    def fake_run_matrix(**kwargs):
        calls.append(kwargs)
        return {"records": []}

    monkeypatch.setattr("scripts.evaluate.run_matrix", fake_run_matrix)
    run_evaluation(variants=["mixed"], opponents=["pass"], seeds=[1], steps=2,
                    seats=[0, 1], churn_window=1)

    assert calls
    assert all(call["churn_window"] == 1 for call in calls)


def test_external_baseline_pairing_failure_is_explicit_in_decision():
    from scripts.evaluate import promotion_decision

    candidate = [
        _metric_record(seat=seat, seed=1, outcome="win", differential=10)
        for seat in (0, 1)
    ]
    incomplete_baseline = [_metric_record(seat=0, seed=1, outcome="loss", differential=1)]

    decision = promotion_decision(
        candidate, incomplete_baseline, min_valid_games=1,
        expected_matrix=[("pass", 1, 0), ("pass", 1, 1)],
        baseline_policy="previous-agent",
    )

    assert decision["status"] == "discard"
    assert "baseline_incomplete_pairing" in decision["reasons"]
    assert decision["baseline_safety_gate_reasons"] == ["missing_expected_matrix_records"]


def test_cli_parses_economic_gates_and_external_baseline():
    from scripts.evaluate import parse_args

    args = parse_args([
        "--candidates", "mixed", "--baseline-policy", "old_agent:agent",
        "--baseline-identity", "previous-agent", "--churn-window", "3",
        "--max-same-item-churn", "1", "--max-market-transactions", "4",
        "--min-terminal-cash", "100", "--min-terminal-inventory-value", "25",
    ])

    assert args.baseline_policy == "old_agent:agent"
    assert args.baseline_identity == "previous-agent"
    assert args.churn_window == 3
    assert args.max_same_item_churn == 1
    assert args.max_market_transactions == 4
    assert args.min_terminal_cash == 100
    assert args.min_terminal_inventory_value == 25


def test_report_exposes_gate_thresholds_and_external_baseline():
    from scripts.evaluate import build_result_document

    records = [
        {**_metric_record(seat=seat, seed=1, candidate="challenger",
                          outcome="win", differential=10),
         "market_transaction_count": 4, "same_item_market_churn": 0,
         "same_item_sell_buy_churn": 0, "terminal_cash": 110,
         "terminal_inventory_value": 30}
        for seat in (0, 1)
    ]
    baseline = [
        {**_metric_record(seat=seat, seed=1, candidate="previous-agent",
                          outcome="loss", differential=1),
         "market_transaction_count": 4, "same_item_market_churn": 0,
         "same_item_sell_buy_churn": 0, "terminal_cash": 101,
         "terminal_inventory_value": 20}
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={"candidates": ["challenger"], "opponents": ["pass"],
                "seed_values": [1], "seats": [0, 1], "min_valid_games": 1,
                "baseline_policy": "old_agent:agent",
                "baseline_identity": "previous-agent", "churn_window": 2,
                "max_same_item_churn": 0, "max_market_transactions": 10,
                "min_terminal_cash": 100, "min_terminal_inventory_value": 0},
        records=records, baseline_records=baseline,
    )

    assert document["metadata"]["baseline_policy"] == {
        "identity": "previous-agent", "path": "old_agent:agent",
    }
    assert document["metadata"]["gate_thresholds"]["max_same_item_churn"] == 0
    assert document["promotion_decisions"]["challenger"]["status"] == "promote"
    assert document["promotion_decisions"]["challenger"]["paired_metric_deltas"] == {
        "market_transaction_count": 0.0,
        "same_item_market_churn": 0.0,
        "terminal_cash": 9.0,
        "terminal_inventory_value": 10.0,
    }


def test_paired_summary_contains_economic_metrics_for_seed_matched_pairs():
    from scripts.evaluate import paired_seed_summary

    records = [
        {**_metric_record(seat=seat, seed=1, outcome="win", differential=10),
         "market_transaction_count": 4 + seat,
         "same_item_market_churn": seat,
         "same_item_sell_buy_churn": seat,
         "terminal_cash": 100 + 10 * seat,
         "terminal_inventory_value": 20 + 5 * seat}
        for seat in (0, 1)
    ]

    summary = paired_seed_summary(records)

    assert summary["total_market_transaction_count"] == 9
    assert summary["total_submitted_market_order_count"] == 9
    assert summary["mean_paired_market_transaction_count"] == 4.5
    assert summary["median_paired_market_transaction_count"] == 4.5
    assert summary["mean_paired_same_item_market_churn"] == 0.5
    assert summary["mean_paired_terminal_cash"] == 105
    assert summary["mean_paired_terminal_inventory_value"] == 22.5


def test_write_result_document_includes_external_baseline_records_in_sidecar(tmp_path):
    from scripts.evaluate import write_result_document

    baseline = [{"candidate": "previous-agent", "seed": 1, "seat": 0}]
    sidecar = write_result_document(
        tmp_path / "report.json", {"schema_version": 1}, records=[],
        baseline_records=baseline,
    )

    sidecar_document = json.loads(sidecar.read_text())
    assert sidecar_document["records"] == [
        {"ablation": "baseline-policy", "evaluation_split": "development", **baseline[0]}
    ]


def test_guarded_external_callable_supports_one_and_two_argument_policies_without_retrying_internal_typeerror():
    from scripts.evaluation_worker import EvaluatorFailure, _GuardedCandidate

    one_argument_calls = []

    def one_argument(observation):
        one_argument_calls.append(observation)
        return "one"

    assert _GuardedCandidate(one_argument)("obs", {"episodeSteps": 2}) == "one"
    assert one_argument_calls == ["obs"]

    two_argument_calls = []

    def two_arguments(observation, configuration):
        two_argument_calls.append((observation, configuration))
        return "two"

    configuration = {"episodeSteps": 2}
    assert _GuardedCandidate(two_arguments)("obs", configuration) == "two"
    assert two_argument_calls == [("obs", configuration)]

    internal_typeerror_calls = []

    def internal_typeerror(observation, configuration):
        internal_typeerror_calls.append((observation, configuration))
        raise TypeError("policy body failed")

    with pytest.raises(EvaluatorFailure, match="policy body failed"):
        _GuardedCandidate(internal_typeerror)("obs", configuration)
    assert len(internal_typeerror_calls) == 1


def test_main_agent_reference_is_accepted_as_one_argument_external_policy():
    from scripts.evaluate import _load_policy_reference
    from scripts.evaluation_worker import _GuardedCandidate

    policy = _load_policy_reference("main.py:agent")

    assert callable(policy)
    assert _GuardedCandidate(policy).accepts_configuration is False


def test_terminal_inventory_values_products_seeds_and_animals_separately():
    from scripts.evaluate import _terminal_inventory_value

    observation = {
        "private": {
            "shed": {"WHEAT": 2},
            "inventories": [{"GOOSE": 1}],
            "seeds": {"WHEAT": 3},
        },
        "market": {"prices": {"WHEAT": 25}, "inventory": {}},
    }

    # Harvested WHEAT is quoted at 25, private WHEAT seed at CROPS[WHEAT].seed,
    # and the animal at its acquisition cost; seed quantity must not be merged
    # into harvested product quantity.
    assert _terminal_inventory_value(observation) == 2 * 25 + 3 * 10 + 300


def test_worker_requires_and_type_checks_new_economic_metrics():
    from scripts.evaluate import _worker_record_error

    record = {
        **_complete_worker_record(),
        "submitted_market_order_count": 2,
        "market_transaction_count": 2,
        "same_item_market_churn": 1,
        "same_item_sell_buy_churn": 1,
        "terminal_cash": 100.0,
        "terminal_inventory_value": 20.0,
    }
    for field in (
        "submitted_market_order_count", "market_transaction_count", "same_item_market_churn",
        "same_item_sell_buy_churn", "terminal_cash", "terminal_inventory_value",
    ):
        missing = dict(record)
        del missing[field]
        assert _worker_record_error(missing, variant="mixed", opponent="pass", seed=4, seat=1)

    for field in ("submitted_market_order_count", "market_transaction_count",
                  "same_item_market_churn", "same_item_sell_buy_churn"):
        for value in (-1, True, 1.5):
            malformed = {**record, field: value}
            assert _worker_record_error(malformed, variant="mixed", opponent="pass", seed=4, seat=1)
    for field in ("terminal_cash", "terminal_inventory_value"):
        malformed = {**record, field: float("nan")}
        assert _worker_record_error(malformed, variant="mixed", opponent="pass", seed=4, seat=1)

    framework = {
        **record,
        "outcome": "framework_error",
        "framework_error": True,
        "final_bank": None,
        "opponent_final_bank": None,
        "bank_differential": None,
        "terminal_cash": None,
        "terminal_inventory_value": None,
    }
    assert _worker_record_error(framework, variant="mixed", opponent="pass", seed=4, seat=1) is None


def test_legacy_no_seat_records_still_apply_economic_gates():
    from scripts.evaluate import promotion_decision

    record = {
        "variant": "mixed", "opponent": "pass", "seed": 1,
        "outcome": "win", "final_bank": 100.0,
        "opponent_final_bank": 90.0, "bank_differential": 10.0,
        "framework_error": False, "market_transaction_count": 2,
        "terminal_cash": 100.0, "terminal_inventory_value": 10.0,
    }

    decision = promotion_decision(
        [record], [record], min_valid_games=1, max_market_transactions=1,
    )

    assert decision["reasons"] == ["market_transaction_count"]


def test_legacy_missing_terminal_inventory_value_counts_as_zero_for_gate():
    from scripts.evaluate import promotion_decision

    complete = {
        "variant": "mixed", "opponent": "pass", "seed": 1,
        "outcome": "win", "final_bank": 100.0,
        "opponent_final_bank": 90.0, "bank_differential": 10.0,
        "framework_error": False, "terminal_inventory_value": 200.0,
    }
    missing = {**complete, "seed": 2}
    del missing["terminal_inventory_value"]

    decision = promotion_decision(
        [complete, missing], [complete, missing], min_valid_games=1,
        min_terminal_inventory_value=50.0,
    )

    assert decision["reasons"] == ["terminal_inventory_value_below_threshold"]


def test_run_evaluation_accepts_and_reports_economic_thresholds(monkeypatch):
    from scripts.evaluate import run_evaluation

    monkeypatch.setattr("scripts.evaluate.run_matrix", lambda **kwargs: {"records": []})
    result = run_evaluation(
        variants=["mixed"], opponents=["pass"], seeds=[1], steps=2,
        seats=[0, 1], max_same_item_market_churn=2,
        max_market_transactions=7, min_terminal_cash=10,
        min_terminal_inventory_value=3,
    )

    assert result["gate_thresholds"] == {
        "max_same_item_market_churn": 2,
        "max_market_transactions": 7,
        "min_terminal_cash": 10.0,
        "min_terminal_inventory_value": 3.0,
    }
    with pytest.raises(ValueError, match="max_market_transactions"):
        run_evaluation(
            variants=["mixed"], opponents=["pass"], seeds=[1], steps=2,
            max_market_transactions=-1,
        )


def test_main_forwards_economic_thresholds_to_run_evaluation(monkeypatch, tmp_path):
    import scripts.evaluate as evaluate

    captured = {}

    def fake_run_evaluation(**kwargs):
        captured["evaluation"] = kwargs
        return {"records": [], "ablation_records": {}, "ablation_configs": {}}

    monkeypatch.setattr(evaluate, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(evaluate, "build_result_document", lambda **kwargs: {"selected_default": "mixed"})
    monkeypatch.setattr(evaluate, "write_result_document", lambda *args, **kwargs: None)

    assert evaluate.main([
        "--seeds", "1", "--steps", "2", "--max-same-item-churn", "2",
        "--max-market-transactions", "7", "--min-terminal-cash", "10",
        "--min-terminal-inventory-value", "3", "--output",
        str(tmp_path / "evaluation.json"),
    ]) == 0
    assert captured["evaluation"]["max_same_item_market_churn"] == 2
    assert captured["evaluation"]["max_market_transactions"] == 7
    assert captured["evaluation"]["min_terminal_cash"] == 10
    assert captured["evaluation"]["min_terminal_inventory_value"] == 3


@pytest.mark.parametrize("kwargs", [
    {"opponents": ["pass", "pass"], "seats": [0]},
    {"opponents": ["pass"], "seats": [0, 0]},
])
def test_run_matrix_rejects_duplicate_opponents_or_seats_before_games(monkeypatch, kwargs):
    from scripts.evaluate import run_matrix

    monkeypatch.setattr("scripts.evaluate.run_game", lambda **kwargs: pytest.fail("run_game should not run"))

    with pytest.raises(ValueError, match="unique"):
        run_matrix(variants=["mixed"], seeds=[1], steps=2, **kwargs)


def test_market_metrics_exposes_submitted_order_event_semantics():
    from scripts.evaluate import _market_metrics

    metrics = _market_metrics([
        _metric_action_state(0, []),
        _metric_action_state(1, [["BUY_PRODUCT", "WHEAT", 9], ["SELL", "MELON", 4]]),
    ])

    assert metrics["submitted_market_order_count"] == 2
    assert metrics["market_transaction_count"] == 2


def test_build_result_document_does_not_fallback_when_external_holdout_baseline_is_missing():
    from scripts.evaluate import build_result_document

    development = [
        _metric_record(seat=seat, seed=1, candidate=candidate,
                       outcome="win" if candidate == "challenger" else "tie",
                       differential=10 if candidate == "challenger" else 1)
        for candidate in ("baseline", "challenger")
        for seat in (0, 1)
    ]
    holdout = [
        _metric_record(seat=seat, seed=100, candidate=candidate,
                       outcome="win" if candidate == "challenger" else "tie",
                       differential=100 if candidate == "challenger" else 1)
        for candidate in ("baseline", "challenger")
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={
            "candidates": ["baseline", "challenger"], "opponents": ["pass"],
            "seed_values": [1], "holdout_seed_values": [100],
            "seats": [0, 1], "min_valid_games": 1,
            "baseline_policy": "old_agent:agent",
        },
        records=development, holdout_records=holdout,
    )

    assert document["selected_candidate"] is None
    assert all(
        decision["status"] == "discard"
        and "baseline_incomplete_pairing" in decision["reasons"]
        for decision in document["holdout_promotion_decisions"].values()
    )


def test_build_result_document_defaults_external_baseline_identity_for_paired_deltas():
    from scripts.evaluate import build_result_document

    records = [
        _metric_record(seat=seat, seed=1, candidate="challenger",
                       outcome="win", differential=10)
        for seat in (0, 1)
    ]
    baseline = [
        _metric_record(seat=seat, seed=1, candidate="previous-agent",
                       outcome="loss", differential=1)
        for seat in (0, 1)
    ]

    document = build_result_document(
        config={
            "candidates": ["challenger"], "opponents": ["pass"],
            "seed_values": [1], "seats": [0, 1], "min_valid_games": 1,
            "baseline_policy": "old_agent:agent",
        },
        records=records, baseline_records=baseline,
    )

    assert document["metadata"]["baseline_policy"] == {
        "identity": "previous-agent", "path": "old_agent:agent",
    }
    assert document["promotion_decisions"]["challenger"]["paired_metric_deltas"] is not None


def test_run_matrix_rejects_duplicate_seeds_before_games(monkeypatch):
    from scripts.evaluate import run_matrix

    monkeypatch.setattr("scripts.evaluate.run_game", lambda **kwargs: pytest.fail("run_game should not run"))

    with pytest.raises(ValueError, match="seeds must be unique"):
        run_matrix(variants=["mixed"], opponents=["pass"], seeds=[1, 1], steps=2, seats=[0])


def test_ablation_decisions_receive_economic_gate_thresholds():
    from scripts.evaluate import build_result_document

    def records_for(candidate):
        return [
            {**_metric_record(seat=seat, seed=1, candidate=candidate,
                              outcome="tie", differential=0),
             "market_transaction_count": 2}
            for seat in (0, 1)
        ]

    document = build_result_document(
        config={
            "candidates": ["baseline", "challenger"], "opponents": ["pass"],
            "seed_values": [1], "seats": [0, 1], "min_valid_games": 1,
            "max_market_transactions": 1,
        },
        records=records_for("baseline") + records_for("challenger"),
        ablation_records={"animals": records_for("baseline") + records_for("challenger")},
    )

    assert document["ablations"]["animals"]["promotion_decisions"]["baseline"]["reasons"] == [
        "market_transaction_count"
    ]


def test_run_evaluation_rejects_unknown_ablation_component_before_games(monkeypatch):
    from scripts.evaluate import run_evaluation

    monkeypatch.setattr("scripts.evaluate.run_matrix", lambda **kwargs: pytest.fail("run_matrix should not run"))

    with pytest.raises(ValueError, match="unsupported ablation component"):
        run_evaluation(
            variants=["mixed"], opponents=["pass"], seeds=[1], steps=2,
            ablations=[("unknown", False)],
        )


def test_external_metric_deltas_returns_none_when_no_valid_records():
    from scripts.evaluate import _external_metric_deltas

    assert _external_metric_deltas([], []) is None


def test_build_result_document_accepts_canonical_and_legacy_churn_threshold_names():
    from scripts.evaluate import build_result_document

    records = [
        {**_metric_record(seat=seat, seed=1, candidate="baseline",
                          outcome="tie", differential=0),
         "same_item_market_churn": 2}
        for seat in (0, 1)
    ]

    canonical = build_result_document(
        config={
            "candidates": ["baseline"], "opponents": ["pass"],
            "seed_values": [1], "seats": [0, 1], "min_valid_games": 1,
            "max_same_item_market_churn": 1,
        },
        records=records,
    )
    legacy = build_result_document(
        config={
            "candidates": ["baseline"], "opponents": ["pass"],
            "seed_values": [1], "seats": [0, 1], "min_valid_games": 1,
            "max_same_item_churn": 1,
        },
        records=records,
    )

    assert canonical["metadata"]["gate_thresholds"]["max_same_item_market_churn"] == 1
    assert canonical["metadata"]["gate_thresholds"]["max_same_item_churn"] == 1
    assert canonical["promotion_decisions"]["baseline"]["reasons"] == ["same_item_market_churn"]
    assert legacy["promotion_decisions"]["baseline"]["reasons"] == ["same_item_market_churn"]


def test_cli_game_count_includes_external_baseline_records(monkeypatch, tmp_path, capsys):
    import scripts.evaluate as evaluate

    baseline = [{"seed": 1, "seat": 0}]
    monkeypatch.setattr(
        evaluate, "run_evaluation",
        lambda **kwargs: {
            "records": [{"seed": 1}], "baseline_records": baseline,
            "ablation_records": {"animals": [{"seed": 1}]},
            "ablation_configs": {},
        },
    )
    monkeypatch.setattr(evaluate, "build_result_document", lambda **kwargs: {"selected_default": "mixed"})
    monkeypatch.setattr(evaluate, "write_result_document", lambda *args, **kwargs: None)

    assert evaluate.main(["--seeds", "1", "--steps", "2", "--output", str(tmp_path / "report.json")]) == 0
    assert json.loads(capsys.readouterr().out)["games"] == 3


def test_worker_requires_submitted_order_count_and_matches_legacy_alias():
    from scripts.evaluate import _worker_record_error

    record = _complete_worker_record()
    assert _worker_record_error(record, variant="mixed", opponent="pass", seed=4, seat=1) is None

    missing = dict(record)
    del missing["submitted_market_order_count"]
    assert _worker_record_error(missing, variant="mixed", opponent="pass", seed=4, seat=1)

    mismatched = {**record, "submitted_market_order_count": 1}
    assert _worker_record_error(mismatched, variant="mixed", opponent="pass", seed=4, seat=1)


def test_worker_accepts_legacy_record_and_normalizes_missing_framework_reasons():
    from scripts.evaluate import _worker_record_error

    record = _complete_worker_record()

    assert _worker_record_error(record, variant="mixed", opponent="pass", seed=4, seat=1) is None
    assert record["framework_error_reasons"] == []


def test_run_game_adds_empty_framework_reasons_to_legacy_worker_result(monkeypatch):
    import scripts.evaluate as evaluate

    response = _complete_worker_record()
    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout=json.dumps(response) + "\n", stderr="",
        ),
    )

    record = evaluate.run_game(variant="mixed", opponent="pass", seed=4, steps=2, seat=1)

    assert record["framework_error_reasons"] == []
    assert record["framework_error"] is False


@pytest.mark.parametrize("reasons", [
    ["market_churn", "market_churn"],
    ["not_a_reason"],
    ["market_churn", "missed_basic_needs"],
])
def test_worker_rejects_invalid_framework_reason_lists(reasons):
    from scripts.evaluate import _worker_record_error

    record = {**_complete_worker_record(), "framework_error_reasons": reasons}
    if reasons == ["market_churn", "market_churn"]:
        record["framework_error"] = True
        record["outcome"] = "framework_error"

    error = _worker_record_error(record, variant="mixed", opponent="pass", seed=4, seat=1)

    assert error and "framework_error_reasons" in error


def test_worker_accepts_all_framework_reasons_in_canonical_order():
    from scripts.evaluate import FRAMEWORK_REASON_ORDER, _worker_record_error

    record = {
        **_complete_worker_record(),
        "framework_error": True,
        "outcome": "framework_error",
        "final_bank": None,
        "opponent_final_bank": None,
        "bank_differential": None,
        "terminal_cash": None,
        "terminal_inventory_value": None,
        "framework_error_reasons": list(FRAMEWORK_REASON_ORDER),
    }

    assert _worker_record_error(record, variant="mixed", opponent="pass", seed=4, seat=1) is None


def test_worker_rejects_framework_error_inconsistent_with_reasons():
    from scripts.evaluate import _worker_record_error

    record = {**_complete_worker_record(), "framework_error_reasons": ["market_churn"]}

    error = _worker_record_error(record, variant="mixed", opponent="pass", seed=4, seat=1)

    assert error and "framework_error" in error


def test_result_document_exposes_per_opponent_metrics_and_promotion_evidence():
    from scripts.evaluate import build_result_document

    records = []
    for opponent, differential in (("pass", 10), ("random", 2)):
        for seat in (0, 1):
            records.append(_metric_record(
                seat=seat, seed=1, opponent=opponent, outcome="win", differential=differential,
            ))

    document = build_result_document(
        config={
            "candidates": ["melon"], "opponents": ["pass", "random"],
            "seed_values": [1], "seats": [0, 1], "min_valid_games": 1,
        },
        records=records,
    )

    metrics = document["metrics_by_opponent"]["melon"]
    assert set(metrics) == {"pass", "random"}
    assert metrics["pass"]["wins"] == 2
    assert metrics["pass"]["valid"] == 2
    assert metrics["pass"]["seat_balanced_win_rate"] == 1.0
    assert metrics["pass"]["mean_bank_differential"] == 10.0
    assert metrics["pass"]["framework_errors"] == 0
    assert "elo_uncertainty" in metrics["pass"]
    evidence = document["promotion_evidence"]["melon"]
    assert evidence["status"] == "baseline"
    assert evidence["matrix_complete"] is True
    assert evidence["matrix_completeness"]["observed"] == [
        ["pass", 1, 0], ["pass", 1, 1], ["random", 1, 0], ["random", 1, 1],
    ]
    assert evidence["metrics_by_opponent"] == metrics


def test_incomplete_matrix_is_recorded_as_non_promotable_evidence():
    from scripts.evaluate import build_result_document

    records = [
        _metric_record(seat=0, seed=1, opponent="pass", outcome="win", differential=10),
    ]
    document = build_result_document(
        config={
            "candidates": ["melon"], "opponents": ["pass"],
            "seed_values": [1], "seats": [0, 1], "min_valid_games": 1,
        },
        records=records,
    )

    evidence = document["promotion_evidence"]["melon"]
    assert evidence["status"] == "discard"
    assert "missing_expected_matrix_records" in evidence["reasons"]
    assert evidence["matrix_complete"] is False
