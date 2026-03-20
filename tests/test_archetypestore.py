import pytest
import pandas as pd
import numpy as np
from pd_ecs._archetype_store import ArchetypeStore
from pd_ecs import Component


def test_add_components():
    archs = ArchetypeStore(dtype=np.int64)
    comp1 = Component('a')
    comp2 = Component('b')
    archs.add_entities(0)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series(0, index=[0])
    )
    archs.add_component(0, comp1)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series(1, index=[0])
    )
    archs.add_entities(1)
    archs.add_component(1, comp2)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([1, 2], index=[0, 1])
    )
    archs.add_component(0, comp2)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([3, 2], index=[0, 1])
    )


def test_remove_components():
    archs = ArchetypeStore(dtype=np.int64)
    comp1 = Component('a')
    comp2 = Component('b')
    archs.add_entities(0)
    archs.add_component(0, comp1)
    archs.remove_component(0, comp1)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series(0, index=[0])
    )
    archs.add_component(0, comp1)
    archs.add_component(0, comp2)
    archs.remove_component(0, comp2)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series(1, index=[0])
    )
    archs.add_component(0, comp2)
    archs.remove_component(0, comp1)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series(2, index=[0])
    )


# --- edge cases ---

def test_add_duplicate_entity_raises():
    archs = ArchetypeStore()
    archs.add_entities(0)
    with pytest.raises(ValueError):
        archs.add_entities(0)


def test_add_component_already_present_is_idempotent():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities(0)
    archs.add_component(0, comp)
    archs.add_component(0, comp)  # should not corrupt the bitmask
    assert archs.series[0] == 1


def test_remove_component_not_present_raises():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities(0)
    archs.add_entities(1)
    archs.add_component(1, comp)  # register comp in the store, but not on entity 0
    with pytest.raises(ValueError):
        archs.remove_component(0, comp)


def test_add_component_invalid_eid_raises():
    archs = ArchetypeStore()
    comp = Component('a')
    with pytest.raises(KeyError):
        archs.add_component(99, comp)


def test_remove_component_invalid_eid_raises():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities(0)
    archs.add_component(0, comp)
    with pytest.raises(KeyError):
        archs.remove_component(99, comp)


def test_remove_unknown_component_raises():
    archs = ArchetypeStore()
    archs.add_entities(0)
    unknown = Component('z')
    with pytest.raises(KeyError):
        archs.remove_component(0, unknown)


def test_dtype_overflow_raises():
    archs = ArchetypeStore(dtype=np.uint8)  # only 8 bits
    archs.add_entities(0)
    for i in range(8):
        archs.add_component(0, Component(f'c{i}'))
    with pytest.raises(OverflowError):
        archs.add_component(0, Component('overflow'))
