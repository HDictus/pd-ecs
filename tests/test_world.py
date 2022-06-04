import pandas as pd
import numpy as np
import pytest as pyt
from mock import MagicMock
from pd_ecs import Component, World, System
from pd_ecs.exceptions import ComponentError


def test_world_has_components():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)

    print(world[component1])
    pd.testing.assert_frame_equal(
        world[component1],
        pd.DataFrame(columns=['some', 'fields', 'id'], dtype=int).set_index('id'))
    pd.testing.assert_frame_equal(
        world[component2],
        pd.DataFrame(columns=['field', 'id'], dtype=int).set_index('id'))


def test_world_has_systems():

    world = World()

    class Asys(System):
        pass

    inst = Asys(world)

    assert world.systems[Asys] == inst


def test_world_add_entities():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)

    world.add_entities(
        {component1: {'some': [1, 2, 3, 4],
                      'fields': [5, 4, 2, 1]}})

    pd.testing.assert_frame_equal(
        world[component1],
        pd.DataFrame({'some': [1, 2, 3, 4],
                      'fields': [5, 4, 2, 1],
                      'id': [0, 1, 2, 3]}).set_index('id'))

    # check they have unique ids

    new = world.add_entities(
        {component2: {'field': [1, 8, 9]}})

    newentities_index = world[component2].index
    assert not any(np.isin(newentities_index, world[component1].index))
    assert new == [4, 5, 6]

def test_world_add_entities_array():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)

    # check that it works with tuples and arrays
    world.add_entities(
        {component1: {'some': np.array([1, 2, 3, 4]),
                      'fields': np.array([5, 4, 2, 1])}})

    pd.testing.assert_frame_equal(
        world[component1],
        pd.DataFrame({'some': [1, 2, 3, 4],
                      'fields': [5, 4, 2, 1],
                      'id': [0, 1, 2, 3]}).set_index('id'))

def world_add_entities_tuple():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)

    world.add_entities(
        {component1: {'some': (1, 2, 3, 4),
                      'fields': (5, 4, 2, 1)}})

    pd.testing.assert_frame_equal(
        world[component1],
        pd.DataFrame({'some': [1, 2, 3, 4],
                      'fields': [5, 4, 2, 1]}))


def test_world_add_single_entity():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)

    world.add_entities(
        {component1: {'some': 1,
                      'fields': 5}})


def test_world_enties_single_value_extrapolated():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)

    world.add_entities(
        {component1: {'some': 1,
                      'fields': 5},
         component2: {'field': ['a', 'b', 'c']}})


def test_world_add_invalid_entities():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)
    # mismatched numbers of entities
    with pyt.raises(ComponentError):
        world.add_entities(
            {component1: {'some': ['a', 'b'],
                          'fields': ['d', 'e']},
             component2: {'field': ['a']}})
    # invalid component field
    with pyt.raises(ComponentError):
        world.add_entities(
            {component2: {'fuld': [0, 0, 0]}})

    # missing field?
    # in the future, replace with type-dependent defaults
    world.add_entities(
            {component1: {'some': ['b', 'c']}})
    assert np.isnan(world[component1]['fields']).all()

def test_world_add_empty():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)
    # mismatched numbers of entities
    world.add_entities(
        {component1: {'some': [], 'fields': []}})
    assert(world[component1].shape[0] == 0)

def test_world_give():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)

    world.add_entities(
        {component2: {'field': [1, 2, 3, 4, 5]}})
    world.add_entities(
        {component1: {'some': ['d'],
                      'fields': ['b']}})
    world.give(
        [1, 4, 2],
        {component1: {'some': ['a', 'b', 'c'],
                      'fields': ['g', 'e', 'f']}})

    assert list(world[component1].index) == [5, 1, 4, 2]
    assert list(world[component1]['some']) == ['d', 'a', 'b', 'c']
    assert list(world[component1]['fields']) == ['b', 'g', 'e', 'f']


def test_world_take():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)
    world.add_entities(
        {component1: {'some': ['b', 'c', 'd'],
                      'fields': ['g', 'e', 'f']},
         component2: {'field': [1, 2, 3]}})
    world.take([1], component1)
    assert list(world[component1].index) == [0, 2]

def test_world_remove_entities():
    component1 = Component('some', 'fields')
    component2 = Component('field')
    world = World(component1, component2)

    world.add_entities(
        {component2: {'field': [1, 2, 3, 4, 5]}})
    world.add_entities(
        {component1: {'some': ['d'],
                      'fields': ['b']}})

    world.remove_entities([3, 4, 5])
    assert list(world[component2].index) == [0, 1, 2]
    assert list(world[component1].index) == []
    return


def test_world_calls_system_events():
    world = World()

    class ASystem(System):

        something_happens = MagicMock()

    sys = ASystem(world)

    world.events.something_happens('banana', 'fork')

    sys.something_happens.assert_called_with('banana', 'fork')


def test_world_set_state():
    comp1 = Component('a')
    comp2 = Component('b')
    comp3 = Component('c')
    world = World(comp1, comp2, comp3)
    state = {
        comp1: pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7],
                             'a': [0, 1, 2, 3, 4, 5, 6, 7]}).set_index('id'),
        comp2: pd.DataFrame({'id': [1, 2, 3, 10],
                             'b': [1, 2, 3, 10]}).set_index('id'),
        comp3: pd.DataFrame({'id': [0, 1, 2, 4, 5, 6],
                             'c': [1, 2, 3, 4, 5, 6]}).set_index('id')}
    world.set_state(state)
    for comp in state:
        pd.testing.assert_frame_equal(world[comp], state[comp])


def test_world_set_state_invalid_components():
    comp1 = Component('a')
    comp2 = Component('b')
    comp3 = Component('c')
    world = World(comp1, comp2)
    state = {
        comp1: pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7],
                             'a': [0, 1, 2, 3, 4, 5, 6, 7]}).set_index('id'),
        comp2: pd.DataFrame({'id': [1, 2, 3, 10],
                             'b': [1, 2, 3, 10]}).set_index('id'),
        comp3: pd.DataFrame({'id': [0, 1, 2, 4, 5, 6],
                             'c': [1, 2, 3, 4, 5, 6]}).set_index('id')}
    with pyt.raises(ComponentError):
        world.set_state(state)


def test_world_set_state_invalid_fields():
    comp1 = Component('a')
    world = World(comp1)
    state = {
        comp1: pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7],
                             'c': [0, 1, 2, 3, 4, 5, 6, 7]}).set_index('id')}
    with pyt.raises(ComponentError):
        world.set_state(state)
