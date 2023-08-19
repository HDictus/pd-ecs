import pandas as pd
import numpy as np
import pytest as pyt
from mock import MagicMock
from pd_ecs import Component, World
from pd_ecs.exceptions import ComponentError


def test_world_index_filters():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()
    has_2 = world.add_entities({
        component1: {'some': [1, 2], 'columns': [2, 3]},
        component2: {'field': [4, 5]}})
    _ = world.add_entities({
        component1: {'some': [4, 5, 6], 'columns': [5, 6, 7]}})
    expdict = {
            (component1, 'some'): [1, 2],
            (component1, 'columns'): [2, 3],
            (component2, 'field'): [4, 5]}
    exp = pd.DataFrame(
        expdict,
        index=has_2)

    pd.testing.assert_frame_equal(
        world[[component1, component2]],
        exp)
    

def test_world_with_loc():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()
    has_2 = world.add_entities({
        component1: {'some': [1, 2], 'columns': [2, 3]},
        component2: {'field': [4, 5]}})
    _ = world.add_entities({
        component1: {'some': [4, 5, 6], 'columns': [5, 6, 7]}})
    expdict = {
            (component1, 'some'): [1, 2],
            (component1, 'columns'): [2, 3],
            (component2, 'field'): [4, 5]}
    exp = pd.DataFrame(
        expdict,
        index=has_2)

    pd.testing.assert_series_equal(
        world.loc[0, [component1, component2]],
        exp.loc[0])
    pd.testing.assert_frame_equal(
        world.loc[[0, 1], component1], exp[component1]
    )
    with pyt.raises(ValueError):
        world.loc[[0, 1]]


def test_world_index_component_field():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()
    has_2 = world.add_entities({
        component1: {'some': [1, 2], 'columns': [2, 3]},
        component2: {'field': [4, 5]}})
    _ = world.add_entities({
        component1: {'some': [4, 5, 6], 'columns': [5, 6, 7]}})
    expdict = {
            (component1, 'some'): [1, 2],
            (component2, 'field'): [4, 5]}
    exp = pd.DataFrame(
        expdict,
        index=has_2)

    pd.testing.assert_frame_equal(
        world[[(component1, 'some'), component2]],
        exp)

def test_world_index_negation():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()
    has_2 = world.add_entities({
        component1: {'some': [1, 2], 'columns': [2, 3]},
        component2: {'field': [4, 5]}})
    has1 = world.add_entities({
        component1: {'some': [4, 5, 6], 'columns': [5, 6, 7]}})
    expdict = {
            (component1, 'some'): [4, 5, 6],
            (component1, 'columns'): [5, 6, 7]}
    exp = pd.DataFrame(
        expdict,
        index=has1)

    pd.testing.assert_frame_equal(
        world[[component1, ~component2]],
        exp)


def test_world_add_entities():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()

    world.add_entities(
        {component1: {'some': [1, 2, 3, 4],
                      'columns': [5, 4, 2, 1]}})

    pd.testing.assert_frame_equal(
        world[component1],
        pd.DataFrame({'some': [1, 2, 3, 4],
                      'columns': [5, 4, 2, 1]}))

    # check they have unique ids

    new = world.add_entities(
        {component2: {'field': [1, 8, 9]}})

    newentities_index = world[component2].index
    assert not any(np.isin(newentities_index, world[component1].index))
    assert new == [4, 5, 6]


def test_world_add_entities_array():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()

    # check that it works with tuples and arrays
    world.add_entities(
        {component1: {'some': np.array([1, 2, 3, 4]),
                      'columns': np.array([5, 4, 2, 1])}})

    pd.testing.assert_frame_equal(
        world[component1],
        pd.DataFrame({'some': [1, 2, 3, 4],
                      'columns': [5, 4, 2, 1]}))

def test_world_add_entities_tuple():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()

    world.add_entities(
        {component1: {'some': (1, 2, 3, 4),
                      'columns': (5, 4, 2, 1)}})

    pd.testing.assert_frame_equal(
        world[component1],
        pd.DataFrame({'some': [1, 2, 3, 4],
                      'columns': [5, 4, 2, 1]}))


def test_world_add_single_entity():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()

    world.add_entities(
        {component1: {'some': 1,
                      'columns': 5}})


def test_world_enties_single_value_extrapolated():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()

    world.add_entities(
        {component1: {'some': 1,
                      'columns': 5},
         component2: {'field': ['a', 'b', 'c']}})


def test_world_add_invalid_entities():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()
    # mismatched numbers of entities
    with pyt.raises(ComponentError):
        world.add_entities(
            {component1: {'some': ['a', 'b'],
                          'columns': ['d', 'e']},
             component2: {'field': ['a']}})
    # invalid component field
    with pyt.raises(ComponentError):
        world.add_entities(
            {component2: {'fuld': [0, 0, 0]}})

    # missing field?
    # in the future, replace with type-dependent defaults
    world.add_entities(
            {component1: {'some': ['b', 'c']}})
    assert np.isnan(world[component1]['columns']).all()

def test_world_add_empty():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()
    # mismatched numbers of entities
    world.add_entities(
        {component1: {'some': [], 'columns': []}})
    assert(world[component1].shape[0] == 0)

def test_world_give():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()

    world.add_entities(
        {component2: {'field': [1, 2, 3, 4, 5]}})
    world.add_entities(
        {component1: {'some': ['d'],
                      'columns': ['b']}})
    world.give(
        [1, 4, 2, 5],
        {component1: {'some': ['a', 'b', 'c', 'a'],
                      'columns': ['g', 'e', 'f', 'c']}})

    assert list(world[component1].index) == [1, 4, 2, 5]
    assert list(world[component1]['some']) == ['a', 'b', 'c', 'a']
    assert list(world[component1]['columns']) == ['g', 'e', 'f', 'c']


def test_world_take():
    component1 = Component('some', 'columns')
    component2 = Component('field')

    world = World()
    world.add_entities(
        {component1: {'some': ['b', 'c', 'd'],
                      'columns': ['g', 'e', 'f']},
         component2: {'field': [1, 2, 3]}})
    world.take([1], component1)
    assert list(world[component1].index) == [0, 2]


def test_world_remove_entities():
    component1 = Component('some', 'columns')
    component2 = Component('field')
    world = World()

    world.add_entities(
        {component2: {'field': [1, 2, 3, 4, 5]}})
    world.add_entities(
        {component1: {'some': ['d'],
                      'columns': ['b']}})

    world.remove_entities([3, 4, 5])
    assert list(world[component2].index) == [0, 1, 2]
    assert list(world[component1].index) == []
    return



def test_world_set_state():
    comp1 = Component('a')
    comp2 = Component('b')
    comp3 = Component('c')
    world = World()
    state = {
        comp1: pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7]}),
        comp2: pd.DataFrame({'b': [1, 2, 3, 10]}),
        comp3: pd.DataFrame({'c': [1, 2, 3, 4, 5, 6]})}
    world.set_state(state)
    for comp in state.keys():
        pd.testing.assert_frame_equal(world[comp], state[comp])


def test_world_set_state_invalid_fields():
    comp1 = Component('a')
    world = World()
    state = {
        comp1: pd.DataFrame({'c': [0, 1, 2, 3, 4, 5, 6, 7]})}
    with pyt.raises(ComponentError):
        world.set_state(state)


def test_world_update():
    comp1 = Component('a')
    comp2 = Component('b')
    comp3 = Component('c')
    world = World()
    state = {
        comp1: pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7]}),
        comp2: pd.DataFrame({'b': [1, 2, 3, 10]}),
        comp3: pd.DataFrame({'c': [1, 2, 3, 4, 5, 6]})}
    world.set_state(state)
    world.update({
        comp1: pd.DataFrame({'a': [0, 3, 2, 4]}, index=[3, 2, 5, 1]),
        comp3: pd.DataFrame({'c': [1, 6, 3, 4]}, index=[1, 4, 2, 0])
    })


# add loc indexing
# add ability to set/remove via list indexing too
# add benchmarking tools
# remove filters
# if performance issues: explore multiple implementations with benchmarking