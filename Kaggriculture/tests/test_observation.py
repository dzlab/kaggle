from kagriculture_agent.observation import (
    is_episode_start,
    is_shed_adjacent,
    is_tile_actionable,
    is_tile_passable,
    iter_tiles,
    observation_turn,
    parse_observation,
    shed_access_tiles,
    shed_total,
)
from kagriculture_agent.types import EpisodeMemory, Position


def make_observation(*, board_size=10, player=0, day=0, hour=0, private=None):
    farms = [
        {
            "tiles": [[None if x < board_size // 2 and y < board_size // 2 else "LOCKED"
                       for x in range(board_size)] for y in range(board_size)],
            "hands": [],
            "farmer": [4, 4],
        },
        {
            "tiles": [["SECOND_PLAYER" for _ in range(board_size)] for _ in range(board_size)],
            "hands": [],
            "farmer": [0, 0],
        },
    ]
    return {
        "player": player,
        "day": day,
        "hour": hour,
        "farms": farms,
        "private": private if private is not None else {},
        "market": {},
        "town": {},
    }


def test_parse_observation_selects_active_player_and_preserves_nw_only_board():
    parsed = parse_observation(make_observation(player=1))

    assert parsed["player"] == 1
    assert parsed["farm"]["tiles"][0][0] == "SECOND_PLAYER"
    assert parsed["farm"]["tiles"][9][9] == "SECOND_PLAYER"


def test_parse_observation_preserves_configuration_mapping():
    obs = make_observation()
    obs["configuration"] = {"shedCapacity": 2}

    parsed = parse_observation(obs)

    assert parsed["configuration"] == {"shedCapacity": 2}


def test_locked_tiles_are_retained_by_tile_iteration():
    farm = make_observation()["farms"][0]

    tiles = list(iter_tiles(farm))

    assert len(tiles) == 100
    assert any(tile == "LOCKED" for _, tile in tiles)
    assert any(position == Position(0, 0) and tile is None for position, tile in tiles)


def test_locked_tiles_are_passable_but_not_actionable():
    assert is_tile_passable("LOCKED")
    assert not is_tile_actionable("LOCKED")
    assert is_tile_passable(None)
    assert is_tile_actionable(None)


def test_shed_access_tiles_for_ten_by_ten_board():
    assert shed_access_tiles(10) == (
        Position(4, 4), Position(5, 4), Position(4, 5), Position(5, 5)
    )
    assert is_shed_adjacent(Position(5, 4), 10)
    assert not is_shed_adjacent(Position(3, 4), 10)


def test_observation_helpers_tolerate_absent_hands_and_count_full_shed():
    private = {"shed": {"WHEAT": 100, "CARROT": 0}}
    obs = make_observation(private=private)
    del obs["farms"][0]["hands"]

    parsed = parse_observation(obs)

    assert parsed["farm"]["hands"] == []
    assert shed_total(private) == 100


def test_parse_observation_normalizes_invalid_player_and_shared_mappings():
    parsed = parse_observation({"player": "not-an-index", "farms": "bad", "private": None,
                                "market": [], "town": object()})

    assert parsed["player"] == 0
    assert parsed["farm"] == {"tiles": [], "hands": []}
    assert parsed["private"] == {}
    assert parsed["market"] == {}
    assert parsed["town"] == {}


def test_parse_observation_handles_none_and_malformed_tiles_and_rows():
    parsed = parse_observation({"farms": [{"tiles": [None, "bad", [["WHEAT"]]], "hands": "bad"}]})

    assert parsed["farm"]["tiles"] == [[], [], [["WHEAT"]]]
    assert parsed["farm"]["hands"] == []
    assert list(iter_tiles(parsed["farm"])) == [(Position(0, 2), ["WHEAT"])]


def test_shed_access_tiles_for_odd_board_size():
    assert shed_access_tiles(9) == (
        Position(3, 3), Position(4, 3), Position(3, 4), Position(4, 4)
    )


def test_episode_start_detects_initial_time_and_time_reset():
    memory = EpisodeMemory(last_day=2, last_hour=23)

    assert is_episode_start({"day": 0, "hour": 0}, memory)
    assert is_episode_start({"day": 1, "hour": 0}, EpisodeMemory(last_day=1, last_hour=3))
    assert not is_episode_start({"day": 2, "hour": 4}, EpisodeMemory(last_day=2, last_hour=4))
    assert is_episode_start({"day": 0, "hour": 0}, None)
    assert not is_episode_start({"day": "bad", "hour": None}, None)


def test_observation_turn_uses_day_and_hour_when_step_is_missing():
    assert observation_turn({"step": None, "day": 7, "hour": 3}) == 171


def test_observation_turn_prefers_valid_step_but_rejects_invalid_values():
    assert observation_turn({"step": 42, "day": 7, "hour": 3}) == 42
    assert observation_turn({"step": -1, "day": 7, "hour": 3}) == -1
    assert observation_turn({"step": "42", "day": 7, "hour": 3}) == 171
    assert observation_turn({"step": True, "day": 7, "hour": 3}) == 171


def test_parse_observation_exposes_seat_safe_turn():
    parsed = parse_observation(make_observation(player=1, day=4, hour=6))

    assert parsed["turn"] == 102
