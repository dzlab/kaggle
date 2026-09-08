import json


def test_orbit_controller_retries_after_rejection_and_retains_only_winner(tmp_path):
    from scripts.train_orbit import OrbitConfig, OrbitController

    calls = []
    candidates = []

    def rollout_fn(**kwargs):
        calls.append(("rollout", kwargs["round_index"]))
        return [kwargs["round_index"]]

    def train_fn(**kwargs):
        candidate = tmp_path / f"candidate-{kwargs['round_index']}.pt"
        candidate.write_text(str(kwargs["rollout"]))
        candidates.append(candidate)
        return candidate

    def evaluate_fn(**kwargs):
        calls.append(("evaluate", kwargs["round_index"]))
        return {"promoted": kwargs["round_index"] == 1}

    result = OrbitController(
        OrbitConfig(tmp_path, max_rounds=3), rollout_fn=rollout_fn,
        train_fn=train_fn, evaluate_fn=evaluate_fn,
    ).run()

    assert result["round"] == 3
    assert result["best"] == str(tmp_path / "best.pt")
    assert (tmp_path / "best.pt").read_text() == "[1]"
    assert [entry["stage"] for entry in result["history"]] == ["rejected", "retained", "rejected"]


def test_orbit_controller_resumes_completed_round_state(tmp_path):
    from scripts.train_orbit import OrbitConfig, OrbitController

    calls = []
    (tmp_path / "orbit-state.json").write_text(json.dumps({
        "round": 1, "failures": 0, "best": None,
        "history": [{"round": 0, "stage": "rejected"}],
    }))

    controller = OrbitController(
        OrbitConfig(tmp_path, max_rounds=2),
        rollout_fn=lambda **kwargs: calls.append(("rollout", kwargs["round_index"])) or [],
        train_fn=lambda **kwargs: tmp_path / "unused.pt",
        evaluate_fn=lambda **kwargs: {"promoted": False},
    )
    controller.run()
    assert calls == [("rollout", 1)]
