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
