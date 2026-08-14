import pandas as pd
import numpy as np
import pytest as pyt
from pd_ecs import Component, World
from pd_ecs.exceptions import ComponentError


def test_world_index_filters():
    component1 = Component('some')
    component2 = Component('field')

    world = World()
    has_2 = world.add_entities({
        component1: [1, 2],
        component2: [4, 5]})
    _ = world.add_entities({
        component1: [4, 5, 6]})
    expdict = {
        component1: [1, 2],
        component2: [4, 5]
    }
    exp = pd.DataFrame(
        expdict,
        index=has_2
    )

    result = world[[component1, component2]]
    assert list(result.index) == list(exp.index)
    pd.testing.assert_series_equal(result[component1], exp[component1])
    pd.testing.assert_series_equal(result[component2], exp[component2])

  
def test_informative_error_accidental_int_index():
    world = World()
    with pyt.raises(ComponentError):
        world.loc[1, 2, 'comp']


def test_world_with_loc():
    # TODO: lotsa code duplicaiton here, can we create a setup func, or else combine into one test?
    component1 = Component('some')
    component2 = Component('field')

    world = World()
    has_2 = world.add_entities({
        component1: [1, 2],
        component2: [4, 5]
    })
    _ = world.add_entities({
        component1:[4, 5, 6]})
    expdict = {
            component1: [1, 2],
            component2: [4, 5]}
    exp = pd.DataFrame(
        expdict,
        index=has_2)

    pd.testing.assert_series_equal(
        world.loc[0, [component1, component2]],
        exp.loc[0], check_dtype=False)
    pd.testing.assert_series_equal(
        world.loc[[0, 1], component1], exp[component1]
    )
    with pyt.raises(ValueError):
        world.loc[[0, 1]]


def test_world_index_negation():
    component1 = Component('some')
    component2 = Component('field')

    world = World()
    has_2 = world.add_entities({
        component1: [1, 2],
        component2: [4, 5]})
    has1 = world.add_entities({
        component1: [4, 5, 6]})
    expdict = {
            component1: [4, 5, 6]}
    exp = pd.DataFrame(
        expdict,
        index=has1)

    result = world[[component1, ~component2]]
    assert list(result.index) == list(exp.index)
    pd.testing.assert_series_equal(result[component1], exp[component1])

    empty = world[[component2, ~component1]]
    assert list(empty.index) == []
    assert component2 in empty.columns


def test_world_add_entities():
    component1 = Component('some')
    component2 = Component('field')

    world = World()

    world.add_entities({component1: [1, 2, 3, 4]})

    pd.testing.assert_series_equal(
        world[component1],
        pd.Series([1, 2, 3, 4], name=component1)
    )
    # check they have unique ids

    new = world.add_entities({component2: [1, 8, 9]})

    newentities_index = world[component2].index
    assert not any(np.isin(newentities_index, world[component1].index))
    assert list(new) == [4, 5, 6]

    # check that this also works with dataframes
    new = world.add_entities(
        pd.DataFrame({component2:  [1, 8, 9]}))

    assert all(world[component2].loc[new] == [1, 8, 9])
    assert list(new) == [7, 8, 9]


def test_world_add_entities_array():
    # TODO: parameterize to avoid duplication
    component1 = Component('some')
    component2 = Component('field')

    world = World()

    # check that it works with tuples and arrays
    world.add_entities(
        {component1:  np.array([1, 2, 3, 4])})

    pd.testing.assert_series_equal(
        world[component1],
        pd.Series([1, 2, 3, 4], name=component1))




def test_world_add_single_entity():
    component1 = Component('some')
    component2 = Component('field')

    world = World()

    world.add_entities(
        {component1: [1]})


def test_world_enties_single_value_extrapolated():
    component1 = Component('some')
    component2 = Component('field')

    world = World()

    world.add_entities(
        {component1: 1,
         component2: ['a', 'b', 'c']})


def test_world_add_invalid_entities():
    component1 = Component('some')
    component2 = Component('field')

    world = World()

    # invalid component field
    with pyt.raises(ComponentError):
        world.add_entities(
            {'fibble': [0, 0, 0]})


def test_world_add_empty():
    component1 = Component('some')
    component2 = Component('field')

    world = World()
    # mismatched numbers of entities
    world.add_entities({component1: []})
    assert(world[component1].shape[0] == 0)


def test_world_give():
    component1 = Component('some')
    component2 = Component('field')

    world = World()

    world.add_entities({component2: [1, 2, 3, 4, 5]})
    world.add_entities({component1: ['d']})
    world.give(
        [1, 4, 2, 5],
        {component1: ['a', 'b', 'c', 'a']})

    # bitmask sort: entity 5 (comp1 only) < entities 1,2,4 (comp1+comp2)
    assert set(world[component1].index) == set([5, 1, 2, 4])
    assert list(world[component1].loc[[5, 1, 2, 4]]) == ['a', 'a', 'c', 'b']

def test_world_give_nonexistent_entity():
    component1 = Component('some')

    world = World()

    world.add_entities({component1: ['d']})
    with pyt.raises(KeyError):
        world.give([1], {component1: 'b'})

def test_world_take():
    component1 = Component('some')
    component2 = Component('field')

    world = World()
    world.add_entities(
        {component1: ['b', 'c', 'd'],
         component2: [1, 2, 3]})
    world.take([1], component1)
    assert list(world[component1].index) == [0, 2]



def test_world_remove_entities():
    component1 = Component('some')
    component2 = Component('field')
    world = World()

    world.add_entities(
        {component2: [1, 2, 3, 4, 5]})
    world.add_entities(
        {component1: ['d']})

    world.remove_entities([3, 4, 5])
    assert list(world[component2].index) == [0, 1, 2]
    assert list(world[component1].index) == []


def test_world_query_after_remove_entities_empties_an_archetype():
    # Regression: removing every entity of a given archetype used to leave a
    # stale, zero-count archetype behind (see
    # test_remove_entities_deletes_emptied_archetype_from_counts in
    # test_archetypestore.py). A multi-component query touching that
    # archetype's bits would then read a neighbouring archetype's range by
    # mistake instead of coming back empty.
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1, 2], comp2: [1, 2]})  # 0, 1: archetype with both
    world.add_entities({comp1: [3]})                    # 2: archetype with comp1 only

    world.remove_entities([0, 1])  # empties the "both" archetype entirely

    assert list(world[[comp1, comp2]].index) == []


def test_world_set_state():
    comp1 = Component('a')
    comp2 = Component('b')
    comp3 = Component('c')
    world = World()
    state = {
        comp1: pd.Series([0, 1, 2, 3, 4, 5, 6, 7],),
        comp2: pd.Series([1, 2, 3, 10], index=[1, 2, 5, 10]),
        comp3: pd.Series([1, 2, 3, 4, 5, 6], index=range(2, 8))}
    world.set_state(state)
    for comp in state.keys():
        exp = pd.Series(state[comp], name=comp)
        pd.testing.assert_series_equal(world[comp].sort_index(), exp.sort_index())


def test_world_set_state_invalid_fields():
    comp1 = Component('a')
    world = World()
    state = {
        'blokle': [0, 1, 2, 3, 4, 5, 6, 7]}
    with pyt.raises(ComponentError):
        world.set_state(state)


def test_world_setting():
    comp1 = Component('a')
    comp2 = Component('c')
    world = World()
    world.add_entities({comp1: [1, 2, 3, 4, 5]})
    world.add_entities({
        comp1: [1, 2],
        comp2: [3, 3]
    })
    world.loc[0, comp1] = 4
    assert world.loc[0, comp1] == 4
    world.loc[[1, 2], comp1] = 5
    assert all(world.loc[[1, 2], comp1] == 5)
    world.loc[[3, 4], [comp1]] = [1, 2]
    assert all(world[comp1].loc[[3, 4]] == [1, 2])

    world.loc[5, [comp1, comp2]] = [1, 3]
    assert all(world.loc[5, [comp1, comp2]].values == [1, 3])

    world.loc[[1, 2], comp1] = [1, 3]
    assert all(world.loc[[1, 2], comp1] == [1, 3])
    
    world.loc[[5, 6], comp2] /= 2
    assert all(world.loc[[5, 6], comp2].values == [1.5, 1.5])
    world.loc[[5, 6], [comp2]] += world[[comp2]].loc[[5, 6]]
    assert all(world.loc[[5, 6], comp2].values == [3, 3])
    
    with pyt.raises(ValueError):
        world.loc[3] = 1


def test_world_loc_del():
    comp1 = Component('a')
    comp2 = Component('c')
    world = World()
    world.add_entities({comp1: [1, 2, 3, 4, 5]})
    world.add_entities({
        comp1: [1, 2],
        comp2: [3, 3]})
    del world.loc[[1, 2]]
    #import pdb; pdb.set_trace()
    assert all(world[comp1].values == [1, 4, 5, 1, 2])
    assert all(world[comp1].index == [0, 3, 4, 5, 6])
    del world.loc[5, comp2]
    assert all(world[comp2].index == 6)
    assert all(world[comp1].values == [1, 4, 5, 1, 2])
    del world.loc[[6], [comp1, comp2]]
    assert all(world[comp1].values == [1, 4, 5, 1])
    assert all(world[comp1].index == [0, 3, 4, 5])


def test_world_index():
    comp1 = Component('a')
    comp2 = Component('c')
    world = World()
    new = world.add_entities({comp1: [1, 2, 3, 4, 5]})
    assert all(world.index == new)


def test_world_give_overwrites_existing_component():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [10, 20, 30], comp2: [1, 2, 3]})  # entities 0, 1, 2

    # Give comp1 to entities 0 and 1, which already have comp1
    world.give([0, 1], {comp1: [99, 88]})

    # New values should overwrite old; entity 2 is unchanged
    qry = world[[comp1, comp2]]
    assert np.allclose(qry[comp1], [99, 88, 30])
    assert np.allclose(qry[comp2], [1, 2, 3])


def test_world_take_component_not_present_is_noop():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1, 2, 3]})   # entities 0,1,2 — no comp2
    world.take([0], comp2)                    # entity 0 never had comp2
    pd.testing.assert_series_equal(
        world[comp1],
        pd.Series([1, 2, 3], name=comp1)
    )

def test_world_take_multi():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1, 2, 3], comp2: [1, 2, 3]})
    world.take([0], comp1, comp2)
    pd.testing.assert_frame_equal(
        world[[comp1, comp2]].to_frame(),
        pd.DataFrame(
            {comp1: [2, 3], comp2: [2, 3]},
            index=[1, 2]
        )
    )


def test_world_take_component_partially_not_present_is_noop():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1, 2, 3], comp2: [4, 5, 6]})  # entities 0,1,2
    world.take([0], comp2)                                      # removes comp2 from 0
    world.take([0], comp2)                                      # entity 0 no longer has comp2 — no-op
    pd.testing.assert_series_equal(
        world[comp2],
        pd.Series([5, 6], index=[1, 2], name=comp2)
    )
    pd.testing.assert_series_equal(
        world[comp1].sort_index(),
        pd.Series([1, 2, 3], name=comp1)
    )


def test_loc_set_empty_index():
    # Assigning to an empty index should be a no-op, not raise IndexError.
    comp1 = Component('a')
    world = World()
    world.add_entities({comp1: [10, 20, 30]})
    empty = pd.Series([], dtype=float)
    world.loc[empty.index, comp1] = empty  # must not raise
    pd.testing.assert_series_equal(
        world[comp1],
        pd.Series([10, 20, 30], name=comp1),
    )


def test_entity_view_boolean_indexing():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1.0, 2.0, 3.0], comp2: [10.0, 20.0, 30.0]})
    view = world[[comp1, comp2]]

    result = view[view[comp1] > 1.0]

    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == [1, 2]
    assert list(result[comp1]) == [2.0, 3.0]
    assert list(result[comp2]) == [20.0, 30.0]


def test_entity_view_boolean_indexing_numpy():
    comp1 = Component('a')
    world = World()
    world.add_entities({comp1: [10.0, 20.0, 30.0]})
    view = world[[comp1]]
    mask = np.array([True, False, True])
    result = view[mask]
    assert isinstance(result, pd.DataFrame)
    assert list(result[comp1]) == [10.0, 30.0]


def test_entity_view_empty_column_access():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1, 2, 3]})
    view = world[[comp1, comp2]]
    assert len(view) == 0
    result = view[comp1]
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_entity_view_dataframe_delegation():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1.0, 2.0, 3.0], comp2: [10.0, 20.0, 30.0]})
    view = world[[comp1, comp2]]

    # property: .values
    assert isinstance(view.values, np.ndarray)
    assert view.values.shape == (3, 2)

    # reduction methods
    assert view.min()[comp1] == 1.0
    assert view.max()[comp2] == 30.0
    assert view.sum()[comp1] == 6.0

    # .iterrows yields (index, Series) pairs
    rows = list(view.iterrows())
    assert len(rows) == 3
    assert rows[0][1][comp1] == 1.0

    # explicit to_frame()
    df = view.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [comp1, comp2]


def test_entity_view_multi_column_getitem_returns_dataframe():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [10, 20, 30], comp2: [1, 2, 3]})

    view = world[[comp1, comp2]]
    result = view[[comp1, comp2]]

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_series_equal(result[comp1], view[comp1], check_names=False)
    pd.testing.assert_series_equal(result[comp2], view[comp2], check_names=False)


def test_entity_view_setitem_single_component():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1, 2, 3], comp2: [10, 20, 30]})

    view = world[[comp1, comp2]]
    view[comp1] = np.array([100, 200, 300])

    pd.testing.assert_series_equal(
        world[comp1],
        pd.Series([100, 200, 300], name=comp1),
    )
    pd.testing.assert_series_equal(
        world[comp2],
        pd.Series([10, 20, 30], name=comp2),
    )


def test_entity_view_setitem_augmented_assignment():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1.0, 2.0, 3.0], comp2: [10.0, 20.0, 30.0]})

    view = world[[comp1, comp2]]
    view[comp1] += view[comp2]

    pd.testing.assert_series_equal(
        world[comp1],
        pd.Series([11.0, 22.0, 33.0], name=comp1),
    )


def test_entity_view_setitem_list_key():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1.0, 2.0, 3.0], comp2: [0.0, 0.0, 0.0]})

    view = world[[comp1, comp2]]
    view[[comp1, comp2]] = pd.DataFrame(
        {comp1: [10.0, 20.0, 30.0], comp2: [4.0, 5.0, 6.0]},
        index=view.index,
    )

    pd.testing.assert_series_equal(world[comp1], pd.Series([10.0, 20.0, 30.0], name=comp1))
    pd.testing.assert_series_equal(world[comp2], pd.Series([4.0, 5.0, 6.0], name=comp2))


# --- EntityView.loc retrieval ---

def test_entity_view_loc_scalar_row_single_col():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [10, 20, 30], comp2: [1, 2, 3]})
    view = world[[comp1, comp2]]
    assert view.loc[1, comp1] == 20
    assert view.loc[0, comp2] == 1


def test_entity_view_loc_list_rows_single_col():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [10, 20, 30], comp2: [1, 2, 3]})
    view = world[[comp1, comp2]]
    pd.testing.assert_series_equal(
        view.loc[[0, 2], comp1],
        pd.Series([10, 30], index=[0, 2], name=comp1),
    )


def test_entity_view_loc_scalar_row_list_cols():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [10, 20, 30], comp2: [1, 2, 3]})
    view = world[[comp1, comp2]]
    row = view.loc[1, [comp1, comp2]]
    assert isinstance(row, pd.Series)
    assert row[comp1] == 20
    assert row[comp2] == 2


def test_entity_view_loc_list_rows_list_cols():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [10, 20, 30], comp2: [1, 2, 3]})
    view = world[[comp1, comp2]]
    sub = view.loc[[0, 2], [comp1, comp2]]
    assert isinstance(sub, pd.DataFrame)
    assert list(sub.index) == [0, 2]
    assert list(sub[comp1]) == [10, 30]
    assert list(sub[comp2]) == [1, 3]


# --- EntityView.loc assignment ---

def test_entity_view_loc_setitem_scalar_row_single_col():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1, 2, 3], comp2: [10, 20, 30]})
    view = world[[comp1, comp2]]
    view.loc[1, comp1] = 99
    assert world[comp1].loc[1] == 99
    assert world[comp1].loc[0] == 1   # untouched


def test_entity_view_loc_setitem_list_rows_single_col():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1.0, 2.0, 3.0], comp2: [10.0, 20.0, 30.0]})
    view = world[[comp1, comp2]]
    view.loc[[0, 2], comp1] = np.array([100.0, 300.0])
    pd.testing.assert_series_equal(
        world[comp1],
        pd.Series([100.0, 2.0, 300.0], name=comp1),
    )


def test_entity_view_loc_setitem_scalar_row_list_cols():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1.0, 2.0, 3.0], comp2: [10.0, 20.0, 30.0]})
    view = world[[comp1, comp2]]
    view.loc[1, [comp1, comp2]] = [99.0, 88.0]
    assert world[comp1].loc[1] == 99.0
    assert world[comp2].loc[1] == 88.0
    assert world[comp1].loc[0] == 1.0   # untouched


def test_entity_view_loc_setitem_list_rows_list_cols():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1.0, 2.0, 3.0], comp2: [10.0, 20.0, 30.0]})
    view = world[[comp1, comp2]]
    view.loc[[0, 2], [comp1, comp2]] = pd.DataFrame(
        {comp1: [100.0, 300.0], comp2: [40.0, 60.0]}, index=[0, 2]
    )
    pd.testing.assert_series_equal(
        world[comp1], pd.Series([100.0, 2.0, 300.0], name=comp1)
    )
    pd.testing.assert_series_equal(
        world[comp2], pd.Series([40.0, 20.0, 60.0], name=comp2)
    )


def test_world_update():
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1.0, 2.0, 3.0], comp2: [10.0, 20.0, 30.0]})
    world.update(pd.DataFrame({comp1: [2.0, 1.0], comp2: [1.0, 1.0]}))
    pd.testing.assert_frame_equal(
        world[[comp1, comp2]].to_frame(),
        pd.DataFrame({
            comp1: [2.0, 1.0, 3.0],
            comp2: [1.0, 1.0, 30.0]})
    )


def test_world_take_does_not_stale():
    # in a previous version, take left stale versions of the taken data
    # in the new dataframe, leading to crashes
    comp1 = Component('a')
    comp2 = Component('b')
    world = World()
    world.add_entities({comp1: [1, 2, 3], comp2: [2, 3, 4]})
    world.take([0], comp2)
    world.give([0], {comp2: [99]})
