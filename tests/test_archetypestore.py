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


def test_add_or_remove_empty():
    archs = ArchetypeStore(dtype=np.int32)
    comp1 = Component('a')
    comp2 = Component('b')
    archs.add_entities(0)
    archs.add_component([], comp1)
    assert archs._test_ranges == {}
    archs.remove_component([], comp2)
    assert archs._test_ranges == {}


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


def test_remove_component_not_present_is_noop():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities(0)
    archs.add_entities(1)
    archs.add_component(1, comp)  # register comp in the store, but not on entity 0
    archs.remove_component(0, comp)  # should do nothing
    assert archs.series[0] == 0
    assert archs.series[1] == 1


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


# --- vectorized operations ---

def test_add_multiple_entities():
    archs = ArchetypeStore(dtype=np.uint32)
    archs.add_entities([0, 1, 2])
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([0, 0, 0], index=[0, 1, 2], dtype=np.uint32)
    )


def test_add_multiple_entities_existing_raises():
    archs = ArchetypeStore()
    archs.add_entities(0)
    with pytest.raises(ValueError):
        archs.add_entities([1, 0])  # 0 already exists


def test_add_multiple_entities_internal_duplicate_raises():
    archs = ArchetypeStore()
    with pytest.raises(ValueError):
        archs.add_entities([0, 0])


def test_add_component_to_multiple_entities():
    archs = ArchetypeStore(dtype=np.uint32)
    comp = Component('a')
    archs.add_entities([0, 1, 2])
    archs.add_component([0, 1, 2], comp)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([1, 1, 1], index=[0, 1, 2], dtype=np.uint32)
    )


def test_add_component_to_multiple_entities_idempotent():
    archs = ArchetypeStore(dtype=np.uint32)
    comp = Component('a')
    archs.add_entities([0, 1, 2])
    archs.add_component([0, 1], comp)
    archs.add_component([1, 2], comp)  # entity 1 already has comp
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([1, 1, 1], index=[0, 1, 2], dtype=np.uint32)
    )


def test_add_component_multiple_entities_invalid_eid_raises():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities([0, 1])
    with pytest.raises(KeyError):
        archs.add_component([0, 99], comp)


def test_remove_component_from_multiple_entities():
    archs = ArchetypeStore(dtype=np.uint32)
    comp = Component('a')
    archs.add_entities([0, 1, 2])
    archs.add_component([0, 1, 2], comp)
    archs.remove_component([0, 1], comp)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([0, 0, 1], index=[0, 1, 2], dtype=np.uint32)
    )


def test_remove_component_multiple_entities_partial_noop():
    archs = ArchetypeStore(dtype=np.uint32)
    comp = Component('a')
    archs.add_entities([0, 1, 2])
    archs.add_component([0, 2], comp)  # entity 1 does not have comp
    archs.remove_component([0, 1], comp)  # entity 1 is silently skipped
    assert archs.series[0] == 0  # cleared
    assert archs.series[1] == 0  # unchanged (already 0)
    assert archs.series[2] == 1  # untouched


def test_remove_component_multiple_entities_invalid_eid_raises():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities([0, 1])
    archs.add_component([0, 1], comp)
    with pytest.raises(KeyError):
        archs.remove_component([0, 99], comp)


# --- remove_component no-op ---

def test_remove_component_missing_is_noop():
    archs = ArchetypeStore(dtype=np.uint32)
    comp = Component('a')
    archs.add_entities(0)
    archs.add_component(0, comp)  # register comp
    archs.add_entities(1)
    archs.remove_component(1, comp)  # entity 1 never had comp
    assert archs.series[0] == 1  # entity 0 unchanged
    assert archs.series[1] == 0  # entity 1 unchanged


def test_remove_component_missing_vectorized_noop():
    archs = ArchetypeStore(dtype=np.uint32)
    comp = Component('a')
    archs.add_entities([0, 1, 2])
    archs.add_component([0, 2], comp)
    archs.remove_component([0, 1, 2], comp)  # entity 1 never had comp
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([0, 0, 0], index=[0, 1, 2], dtype=np.uint32)
    )


# --- entity removal ---

def test_remove_single_entity():
    archs = ArchetypeStore(dtype=np.uint32)
    archs.add_entities([0, 1, 2])
    archs.remove_entities(1)
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([0, 0], index=[0, 2], dtype=np.uint32)
    )


def test_remove_all_entities():
    archs = ArchetypeStore(dtype=np.uint32)
    archs.add_entities([0, 1])
    archs.remove_entities([0, 1])
    assert archs.series.empty


def test_remove_entity_preserves_components_on_others():
    archs = ArchetypeStore(dtype=np.uint32)
    comp = Component('a')
    archs.add_entities([0, 1])
    archs.add_component([0, 1], comp)
    archs.remove_entities(0)
    assert archs.series[1] == 1


def test_remove_entity_nonexistent_raises():
    archs = ArchetypeStore()
    archs.add_entities(0)
    with pytest.raises(KeyError):
        archs.remove_entities(99)


def test_remove_entities_partial_nonexistent_raises():
    archs = ArchetypeStore()
    archs.add_entities([0, 1])
    with pytest.raises(KeyError):
        archs.remove_entities([0, 99])


def test_remove_entities_duplicate_raises():
    archs = ArchetypeStore()
    archs.add_entities([0, 1])
    with pytest.raises(ValueError):
        archs.remove_entities([0, 0])


def test_remove_multiple_entities():
    archs = ArchetypeStore(dtype=np.uint32)
    comp = Component('a')
    archs.add_entities([0, 1, 2, 3])
    archs.add_component([1, 2], comp)
    archs.remove_entities([1, 3])
    pd.testing.assert_series_equal(
        archs.series,
        pd.Series([0, 1], index=[0, 2], dtype=np.uint32)
    )

# --- basic ranges ---

def test_range_on_component_added():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities([0, 1, 2, 3, 4])
    archs.add_component([0, 1, 2], comp)
    assert archs._test_ranges[comp] == {1: (0, 3)}


def test_range_on_component_removed():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities([0, 1, 2, 3, 4])
    archs.add_component([0, 1, 2], comp)
    archs.remove_component(1, comp)
    assert archs._test_ranges[comp] == {1: (0, 2)}


def test_range_on_non_first_component_removed_from_shared_archetype():
    # Regression test: removing a component that isn't the first one
    # registered (i.e. not bit 0) used to corrupt that component's own
    # _ranges, because remove_component shrank the range at the entity's
    # *new* archetype instead of its *old* one. That silently dropped
    # unrelated entities from the range and left a stale/incorrect entry
    # behind, which downstream could surface as entities appearing twice
    # (or not at all) when reading multiple components, eventually causing
    # `ValueError: duplicate eids in input` out of remove_entities.
    archs = ArchetypeStore()
    z = Component('z')  # registered first -> bit 0
    x = Component('x')  # registered second -> bit 1
    archs.add_entities([0, 1, 2])
    archs.add_component([0, 2], z)  # entity 0: z only, entity 2: z
    archs.add_component([1, 2], x)  # entity 1: x only, entity 2: z+x
    assert archs._test_ranges[x] == {2: (0, 1), 3: (1, 2)}

    archs.remove_component(2, x)

    # entity 1 (mask 2, x only) must still be tracked in x's ranges
    assert archs._test_ranges[x] == {2: (0, 1)}


def test_range_on_entity_removed():
    archs = ArchetypeStore()
    comp = Component('a')
    archs.add_entities([0, 1, 2, 3, 4])
    archs.add_component([0, 1, 2], comp)
    archs.remove_entities([2])
    assert archs._test_ranges[comp] == {1: (0, 2)}


# --- multi-archetype ranges ---

def test_ranges_component_spans_two_archetypes():
    # comp1 alone (mask=1) and comp1+comp2 (mask=3)
    archs = ArchetypeStore()
    comp1 = Component('a')
    comp2 = Component('b')
    archs.add_entities([0, 1, 2, 3, 4])
    archs.add_component([0, 1, 2, 3, 4], comp1)  # all: mask=1
    archs.add_component([3, 4], comp2)             # 3,4: mask=3
    # archetype_counts sorted: {1:3, 3:2}
    # comp1 appears in both archetypes: cumsum {1:3, 3:5}
    assert archs._test_ranges[comp1] == {1: (0, 3), 3: (3, 5)}
    # comp2 only in archetype 3
    assert archs._test_ranges[comp2] == {3: (0, 2)}


def test_ranges_two_components_disjoint_archetypes():
    # comp1-only, comp2-only, and comp1+comp2 entities
    archs = ArchetypeStore()
    comp1 = Component('a')
    comp2 = Component('b')
    archs.add_entities([0, 1, 2, 3, 4])
    archs.add_component([0, 1], comp1)    # mask=1
    archs.add_component([2, 3], comp2)    # mask=2
    archs.add_component([4], comp1)
    archs.add_component([4], comp2)       # mask=3
    # archetype_counts sorted: {1:2, 2:2, 3:1}
    # comp1 (bit 0): archetypes 1, 3 → {1:2, 3:1} → cumsum {1:2, 3:3}
    assert archs._test_ranges[comp1] == {1: (0, 2), 3: (2, 3)}
    # comp2 (bit 1): archetypes 2, 3 → {2:2, 3:1} → cumsum {2:2, 3:3}
    assert archs._test_ranges[comp2] == {2: (0, 2), 3: (2, 3)}


def test_choose_archetypes():
    archs = ArchetypeStore()
    comp1 = Component('a')
    comp2 = Component('b')
    archs.add_entities([0, 1, 2, 3, 4])
    # will have ats 0, 1, 3, 2, 0
    archs.add_component([1, 2], comp1)
    archs.add_component([2, 3], comp2)
    assert all(archs.choose_archetypes([comp1]) == [1, 3])
    assert all(archs.choose_archetypes([comp1, comp2]) == [3])
    assert all(archs.choose_archetypes([~comp1]) == [0, 2])
    assert all(archs.choose_archetypes([comp1, ~comp2]) == 1)
    comp3 = Component('c')
    assert len(archs.choose_archetypes([comp3])) == 0
    archs.add_component([2], comp3)  # gives at 3+2**2 = 7
    # check excludes absent archetypes
    assert all(archs.choose_archetypes([comp3]) == [7])


def test_choose_archetypes_unregistered_include_returns_empty():
    archs = ArchetypeStore()
    comp1 = Component('a')
    comp_unseen = Component('z')
    archs.add_entities([0, 1])
    archs.add_component([0, 1], comp1)
    # comp_unseen has never been given to any entity — should return empty, not crash
    assert len(archs.choose_archetypes([comp_unseen])) == 0
    # and it should have been registered so a subsequent add_component uses the same bit
    assert comp_unseen in archs._component_powers


def test_choose_archetypes_unregistered_exclude_does_not_crash():
    archs = ArchetypeStore()
    comp1 = Component('a')
    comp_unseen = Component('z')
    archs.add_entities([0, 1, 2])
    archs.add_component([0, 1, 2], comp1)
    # ~comp_unseen: no entity has it, so all archetypes pass the exclude
    result = archs.choose_archetypes([comp1, ~comp_unseen])
    assert all(result == archs.choose_archetypes([comp1]))
