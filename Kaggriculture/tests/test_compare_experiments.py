import json

import pytest


def _report(*, seeds=(1,), win_rate=0.5, elo=1000.0, bank=0.0, safety=0.0):
    expected = [["hard", seed, seat] for seed in seeds for seat in (0, 1)]
    matrix = {
        "expected": expected,
        "expected_count": len(expected),
        "observed_count": len(expected),
        "missing": [], "duplicate": [], "extra": [], "invalid_records": 0,
    }
    metrics = {
        "record_count": len(expected), "valid": len(expected), "valid_games": len(expected),
        "wins": int(win_rate * len(expected)), "losses": 0, "ties": 0,
        "seat_balanced_win_rate": win_rate,
        "elo_rating": elo, "elo_uncertainty": 20.0,
        "mean_bank_differential": bank,
        "framework_errors": 0, "invalid": 0, "timeouts": 0, "no_progress": 0,
        "safety_failure_rate": safety,
    }
    return {
        "schema_version": 1,
        "manifest": {
            "seeds": list(seeds), "opponents": ["hard"], "seats": [0, 1],
        },
        "metrics_by_opponent": {"candidate": {"hard": metrics}},
        "promotion_evidence": {
            "candidate": {"status": "promote", "matrix_complete": True, "matrix_completeness": matrix}
        },
    }


def test_compare_reports_emits_deltas_against_first_report(tmp_path):
    from scripts.compare_experiments import compare_reports, render_markdown

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    third = tmp_path / "third.json"
    first.write_text(json.dumps(_report()), encoding="utf-8")
    second.write_text(json.dumps(_report(win_rate=0.65, elo=1050.0, bank=8.0, safety=0.1)), encoding="utf-8")
    third.write_text(json.dumps(_report(win_rate=0.75, elo=1100.0, bank=12.0, safety=0.0)), encoding="utf-8")

    comparison = compare_reports([first, second, third])

    assert len(comparison["deltas"]) == 2
    assert comparison["deltas"][0] == {
        "candidate": "candidate", "opponent": "hard", "report": "second",
        "win_rate_delta": pytest.approx(0.15), "elo_delta": pytest.approx(50.0),
        "bank_delta": pytest.approx(8.0), "safety_delta": pytest.approx(0.1),
        "decision": "promote",
    }
    markdown = render_markdown(comparison)
    assert "candidate | opponent | win_rate_delta | elo_delta | bank_delta | safety_delta | decision" in markdown
    assert "| candidate | hard | 0.15 | 50.0 | 8.0 | 0.1 | promote |" in markdown


def test_compare_reports_rejects_incompatible_matrix(tmp_path):
    from scripts.compare_experiments import compare_reports

    first = tmp_path / "first.json"
    incompatible = tmp_path / "incompatible.json"
    first.write_text(json.dumps(_report(seeds=(1,))), encoding="utf-8")
    incompatible.write_text(json.dumps(_report(seeds=(2,))), encoding="utf-8")

    with pytest.raises(ValueError, match="matrix"):
        compare_reports([first, incompatible])


def test_compare_cli_writes_json_and_markdown(tmp_path):
    from scripts.compare_experiments import main

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output_json = tmp_path / "comparison.json"
    output_markdown = tmp_path / "comparison.md"
    first.write_text(json.dumps(_report()), encoding="utf-8")
    second.write_text(json.dumps(_report(win_rate=0.75)), encoding="utf-8")

    assert main([
        str(first), str(second), "--json-output", str(output_json),
        "--markdown-output", str(output_markdown),
    ]) == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["deltas"][0]["win_rate_delta"] == pytest.approx(0.25)
    assert output_markdown.read_text(encoding="utf-8").startswith("# Experiment comparison")
