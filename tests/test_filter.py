from pd_ecs import World, Component
from pd_ecs.filter import Filter
import pytest


def test_filter_duplicate_added():
    world = World()
    comp1 = Component('a')
    comp2 = Component('b')
    with pytest.warns(DeprecationWarning):
        filt = world[(comp1, comp2)]
    entity = world.add_entities({comp1: {'a': 1}})
    assert len(filt.ids) == 0
    world.give(entity, {comp2: {'b': 4}})
    assert all(filt.ids == [0])
    world.give(entity, {comp2: {'b': 6}})
    assert len(filt.ids) == 1
    assert all(filt.ids == [0])

    assert (filt.index == filt.ids).all()


def test_filter_excludes_component():
    world = World()
    comp1 = Component('a')
    comp2 = Component('b')
    entity1 = world.add_entities({comp1: {'a': 1}})
    entities2 = world.add_entities({comp1: {'a': [1, 2]},
                                   comp2: {'b': [1, 2]}})
    with pytest.warns(DeprecationWarning):
        filt = world[(comp1, ~comp2)]
    assert list(filt.ids) == [entity1[0]]
    world.take(entities2[1], comp2)
    assert list(filt.ids) == [entity1[0], entities2[1]]
    world.give(entity1, {comp2: {'b': [0]}})
    assert list(filt.ids) == [entities2[1]]
