import pandas as pd
import numpy as np
import pytest as pyt
from pd_ecs import Component, World, System
from pd_ecs.exceptions import ComponentError



def test_world_has_components():
    component1 = Component('some', 'fields')
    component2 = Component('field')

    world = World(component1, component2)

    print(world[component1])
    pd.testing.assert_frame_equal(
        world[component1],
        pd.DataFrame(columns=['some', 'fields'], dtype=int))
    pd.testing.assert_frame_equal(
        world[component2],
        pd.DataFrame(columns=['field'], dtype=int))


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
                      'fields': [5, 4, 2, 1]}))


    # check they have unique ids

    world.add_entities(
        {component2: {'field': [1, 8, 9]}})

    newentities_index = world[component2].index
    assert not any(np.isin(newentities_index, world[component1].index))


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
