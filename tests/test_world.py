import pandas as pd
import numpy as np
import pytest as pyt
from pd_ecs import Component, World, System



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

    # wrong fields
    with pyt.raises(KeyError):
        world.add_entities(
            {component2: {'fuld': [0, 0, 0]}})
