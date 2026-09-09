import json
from pathlib import Path


REPORT_DIR = Path(__file__).parents[1] / "reports"
REPORT_JSON = REPORT_DIR / "orbit-policy-experiment-summary.json"
REPORT_MARKDOWN = REPORT_DIR / "orbit-policy-experiment-summary.md"

EXPECTED_ENTRIES = {
    "baseline_bc_ppo",
    "longer_bc_ppo",
    "extended_bc_ppo",
    "league_bc_ppo",
    "pure_ppo",
    "reduced_bc_ppo",
    "experimental_context_league",
}


def _report():
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


def test_pending_report_has_explicit_schema_and_execution_status():
    report = _report()

    assert report["schema_version"] == 1
    assert report["report_type"] == "orbit_policy_experiment_summary"
    assert report["status"] == "pending"
    assert report["execution_status"] == "not_executed"
    assert report["promotion"]["status"] == "not_promoted"
    assert report["promotion"]["candidate"] is None
    assert report["required_artifacts"]
    assert report["required_metrics"]
    assert report["gate_criteria"]


def test_pending_report_describes_the_configured_matrix_without_results():
    report = _report()
    entries = report["matrix"]["entries"]
    shared = report["matrix"]["shared"]

    assert {entry["experiment_id"] for entry in entries} == EXPECTED_ENTRIES
    assert len(entries) == len(EXPECTED_ENTRIES)
    assert shared["training_seed_candidates"] == [7, 11, 19]
    for entry in entries:
        assert entry["status"] == "not_executed"
        assert entry["results"] is None
    assert shared["development_seeds"] == list(range(50))
    assert shared["holdout_seeds"] == list(range(100, 150))
    assert shared["development_opponents"] == ["pass", "random", "starter"]
    assert shared["holdout_opponents"] == ["pass", "random", "starter"]
    assert shared["development_seats"] == [0, 1]
    assert shared["holdout_seats"] == [0, 1]


def test_absent_results_cannot_claim_promotion():
    report = _report()

    assert all(entry["results"] is None for entry in report["matrix"]["entries"])
    assert report["promotion"] == {
        "status": "not_promoted",
        "candidate": None,
        "reason": "not_executed",
    }
    assert report["status"] == "pending"


def test_markdown_report_matches_pending_boundary():
    markdown = REPORT_MARKDOWN.read_text(encoding="utf-8")

    assert "Status: `pending`" in markdown
    assert "Execution status: `not_executed`" in markdown
    assert "Promotion: `not_promoted`" in markdown
    for experiment_id in EXPECTED_ENTRIES:
        assert f"`{experiment_id}`" in markdown
    assert "No Colab/GPU results were available" in markdown
