import pandas as pd
import numpy as np
import pytest as pyt
from mock import MagicMock
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

    pd.testing.assert_frame_equal(
        world[[component1, component2]],
        exp)

  
def test_informative_error_accidental_int_index():
    world = World()
    with pyt.raises(KeyError):
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
        exp.loc[0])
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

    pd.testing.assert_frame_equal(
        world[[component1, ~component2]],
        exp)


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
    assert new == [4, 5, 6]

    # check that this also works with dataframes
    new = world.add_entities(
        pd.DataFrame({component2:  [1, 8, 9]}))

    assert all(world[component2].loc[new] == [1, 8, 9])
    assert new == [7, 8, 9]


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

    assert list(world[component1].index) == [1, 4, 2, 5]
    assert list(world[component1]) == ['a', 'b', 'c', 'a']


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
        pd.testing.assert_series_equal(world[comp], exp)


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

