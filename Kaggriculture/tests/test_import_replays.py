import hashlib
import json
import os

import pytest

import scripts.import_replays as import_replays_module
from scripts.import_replays import import_replays
from scripts.run_local import run_episode


def _write_replay(directory, *, seed=17, steps=4, name="episode.json"):
    path = directory / name
    run_episode(opponent="pass", seed=seed, steps=steps, replay_path=path)
    return path


def test_import_replays_writes_valid_transitions_and_provenance_manifest(tmp_path):
    source = tmp_path / "public"
    source.mkdir()
    first = _write_replay(source, seed=17, name="a.json")

    output = tmp_path / "corpus.jsonl"
    manifest = import_replays(source, output, candidate_player=1)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert all(row["observation"]["player"] == 1 for row in rows)
    assert manifest["engine_version"] == "1.32.7"
    assert manifest["source_replay_count"] == 1
    assert manifest["transition_count"] == 3
    assert manifest["seeds"] == [17]
    assert manifest["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["sources"] == [{
        "path": "a.json",
        "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        "seed": 17,
        "transition_count": 3,
    }]
    assert json.loads(output.with_suffix(".manifest.json").read_text()) == manifest


def test_import_replays_fails_closed_on_wrong_engine_version(tmp_path):
    source = tmp_path / "public"
    source.mkdir()
    replay = _write_replay(source)
    payload = json.loads(replay.read_text())
    payload["module_version"] = "1.32.6"
    replay.write_text(json.dumps(payload))

    output = tmp_path / "corpus.jsonl"
    with pytest.raises(ValueError, match="engine version"):
        import_replays(source, output)

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()


def test_import_replays_rejects_empty_source_directory(tmp_path):
    with pytest.raises(ValueError, match="no replay JSON files"):
        import_replays(tmp_path / "missing", tmp_path / "corpus.jsonl")


def test_import_replays_can_repeat_when_output_is_inside_source_directory(tmp_path):
    source = tmp_path / "public"
    source.mkdir()
    _write_replay(source, seed=21)
    output = source / "corpus.jsonl"

    first_manifest = import_replays(source, output)
    second_manifest = import_replays(source, output)

    assert second_manifest == first_manifest
    assert second_manifest["source_replay_count"] == 1
    assert output.exists()
    assert output.with_suffix(".manifest.json").exists()


def test_import_replays_restores_both_files_if_manifest_install_fails(tmp_path, monkeypatch):
    source = tmp_path / "public"
    source.mkdir()
    _write_replay(source, seed=21)
    output = tmp_path / "corpus.jsonl"
    import_replays(source, output)
    old_output = output.read_bytes()
    manifest_path = output.with_suffix(".manifest.json")
    old_manifest = manifest_path.read_bytes()

    real_replace = os.replace

    def fail_manifest_install(source_path, destination_path):
        if destination_path == manifest_path and str(source_path).endswith(".tmp"):
            raise OSError("injected manifest install failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(import_replays_module.os, "replace", fail_manifest_install)
    with pytest.raises(OSError, match="manifest install"):
        import_replays(source, output)

    assert output.read_bytes() == old_output
    assert manifest_path.read_bytes() == old_manifest
    assert not list(tmp_path.glob(".*.bak"))
