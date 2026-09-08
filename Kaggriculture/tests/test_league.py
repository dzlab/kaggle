from pathlib import Path

import pytest

from kagriculture_agent.league import (
    HistoricalCheckpoint,
    LeagueSampler,
    OpponentMatch,
    SkillBand,
    sample_match,
    sample_schedule,
)


def _checkpoint(tmp_path: Path, name: str, band: str) -> HistoricalCheckpoint:
    path = tmp_path / name
    path.touch()
    return HistoricalCheckpoint(path=path, skill_band=band)


def test_sampling_is_deterministic_for_seed_and_index(tmp_path):
    checkpoints = (
        _checkpoint(tmp_path, "early.pt", "early"),
        _checkpoint(tmp_path, "late.pt", "late"),
    )
    sampler = LeagueSampler(
        skill_bands={
            "early": SkillBand("early", (checkpoints[0],)),
            "late": SkillBand("late", (checkpoints[1],)),
        },
    )

    first = [sampler.sample(index, seed=91) for index in range(40)]
    second = [sampler.sample(index, seed=91) for index in range(40)]

    assert first == second
    assert all(isinstance(match, OpponentMatch) for match in first)


def test_schedule_keeps_total_size_and_balances_seats(tmp_path):
    checkpoint = _checkpoint(tmp_path, "historical.pt", "middle")
    matches = sample_schedule(
        17,
        seed=7,
        skill_bands={"middle": SkillBand("middle", (checkpoint,))},
    )

    assert len(matches) == 17
    assert sum(match.seat == 0 for match in matches) == 9
    assert sum(match.seat == 1 for match in matches) == 8
    assert [match.seat for match in matches] == [index % 2 for index in range(17)]


def test_checkpoint_sampling_is_uniform_within_selected_band(tmp_path):
    checkpoints = tuple(
        _checkpoint(tmp_path, f"checkpoint-{index}.pt", "hard")
        for index in range(3)
    )
    sampler = LeagueSampler(
        probabilities={"checkpoint": 1.0},
        skill_bands={"hard": SkillBand("hard", checkpoints)},
    )

    selected = {sampler.sample(index, seed=123).checkpoint for index in range(300)}

    assert selected == {str(item.path) for item in checkpoints}
    assert all(sampler.sample(index, seed=123).skill_band == "hard" for index in range(300))


def test_empty_selected_band_falls_back_to_current(tmp_path):
    available = _checkpoint(tmp_path, "available.pt", "available")
    sampler = LeagueSampler(
        probabilities={"checkpoint": 1.0},
        skill_bands={
            "empty": SkillBand("empty", ()),
            "available": SkillBand("available", (available,)),
        },
        band_probabilities={"empty": 1.0, "available": 0.0},
    )

    match = sampler.sample(0, seed=4)

    assert match == OpponentMatch(opponent="current", seat=0)


def test_missing_checkpoint_path_is_not_selected(tmp_path):
    missing = HistoricalCheckpoint(path=tmp_path / "gone.pt", skill_band="hard")
    sampler = LeagueSampler(
        probabilities={"checkpoint": 1.0},
        skill_bands={"hard": SkillBand("hard", (missing,))},
    )

    assert sampler.sample(0, seed=1).opponent == "current"


def test_explicit_hard_band_weight_changes_selection_without_changing_count(tmp_path):
    easy = _checkpoint(tmp_path, "easy.pt", "easy")
    hard = _checkpoint(tmp_path, "hard.pt", "hard")
    sampler = LeagueSampler(
        probabilities={"checkpoint": 1.0},
        skill_bands={
            "easy": SkillBand("easy", (easy,)),
            "hard": SkillBand("hard", (hard,)),
        },
        band_probabilities={"easy": 1.0, "hard": 1.0},
        hard_opponent_weights={"hard": 9.0},
    )

    matches = sampler.schedule(100, seed=8)
    hard_count = sum(match.skill_band == "hard" for match in matches)

    assert len(matches) == 100
    assert hard_count > 70


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probabilities": {}},
        {"probabilities": {"unknown": 1.0}},
        {"probabilities": {"current": -1.0}},
        {"probabilities": {"current": 0.0}},
        {"skill_bands": {"hard": SkillBand("hard", ())}, "band_probabilities": {"missing": 1.0}},
    ],
)
def test_sampler_rejects_malformed_configuration(kwargs):
    with pytest.raises(ValueError):
        LeagueSampler(**kwargs)


def test_match_rejects_invalid_seat():
    with pytest.raises(ValueError):
        OpponentMatch(opponent="current", seat=2)


@pytest.mark.parametrize("skill_bands", [None, (), [], "hard", b"hard", {"hard": "not-a-band"}])
def test_sampler_requires_skill_bands_mapping(skill_bands):
    with pytest.raises(ValueError):
        LeagueSampler(skill_bands=skill_bands)


def test_skill_band_mapping_accepts_path_entries_and_normalizes_them(tmp_path):
    path = tmp_path / "hard.pt"
    path.touch()

    sampler = LeagueSampler(
        probabilities={"checkpoint": 1.0},
        skill_bands={"hard": SkillBand("hard", (path,))},
    )

    assert sampler.skill_bands[0].checkpoints[0].path == str(path)
    assert sampler.sample(0).checkpoint == str(path)


@pytest.mark.parametrize("opponent", [["current"], {"name": "current"}, None])
def test_match_rejects_non_string_opponent_as_value_error(opponent):
    with pytest.raises(ValueError):
        OpponentMatch(opponent=opponent, seat=0)


@pytest.mark.parametrize("checkpoints", ["checkpoint.pt", b"checkpoint.pt"])
def test_skill_band_rejects_string_checkpoint_sequences(checkpoints):
    with pytest.raises(ValueError):
        SkillBand("hard", checkpoints)


@pytest.mark.parametrize("checkpoints", [("",), (None,), (123,)])
def test_skill_band_rejects_empty_or_non_string_checkpoint_entries(checkpoints):
    with pytest.raises(ValueError):
        SkillBand("hard", checkpoints)


def test_sampler_rejects_string_checkpoint_candidates():
    with pytest.raises(ValueError):
        LeagueSampler(checkpoint_candidates="checkpoint.pt")


@pytest.mark.parametrize("opponents", ["current", b"current"])
def test_sampler_rejects_string_mixed_opponent_sequences(opponents):
    with pytest.raises(ValueError):
        LeagueSampler(mixed_opponents=opponents)


@pytest.mark.parametrize("sampler", [[], {}, False, 0, ["not-a-sampler"], {"sampler": True}, object()])
def test_sample_schedule_rejects_falsy_and_truthy_non_sampler_objects(sampler):
    with pytest.raises(ValueError):
        sample_schedule(1, sampler=sampler)


@pytest.mark.parametrize("sampler", [[], {}, False, 0, ["not-a-sampler"], {"sampler": True}, object()])
def test_sample_match_rejects_falsy_and_truthy_non_sampler_objects(sampler):
    with pytest.raises(ValueError):
        sample_match(0, sampler=sampler)
