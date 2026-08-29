import pytest

from kagriculture_agent.constants import (
    ANIMALS,
    CROPS,
    ENGINE_VERSION,
    LAND_ORDER,
    LAND_PRICES,
    PRODUCTS,
    SHOP_DEMANDS,
)


def test_published_constants_import_and_keep_list_value_semantics():
    assert set(CROPS) == {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
    assert set(ANIMALS) == {"GOOSE", "COW", "SHEEP"}
    assert isinstance(SHOP_DEMANDS["BAKERY"], list)
    assert SHOP_DEMANDS["BAKERY"] == ["EGG", "WHEAT"]
    assert isinstance(PRODUCTS, list)
    assert PRODUCTS == ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
    assert isinstance(LAND_ORDER, list)
    assert LAND_ORDER == ["NE", "SW", "SE"]
    assert isinstance(LAND_PRICES, list)
    assert LAND_PRICES == [1000, 2000, 4000]
    assert ENGINE_VERSION == "1.32.7"


def test_nested_rule_tables_are_immutable():
    with pytest.raises(TypeError):
        CROPS["WHEAT"]["seed"] = 999
    with pytest.raises(TypeError):
        SHOP_DEMANDS["BAKERY"].append("MILK")
    with pytest.raises(TypeError):
        PRODUCTS[0] = "MELON"
    with pytest.raises(TypeError):
        LAND_ORDER.__iadd__(["NW"])
    with pytest.raises(TypeError):
        CROPS["WHEAT"] = {}
