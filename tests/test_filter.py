from pd_ecs import World, Component
from pd_ecs.filter import Filter


def test_filter_duplicate_added():
    world = World()
    comp1 = Component('a')
    comp2 = Component('b')
    filt = Filter(comp1, comp2, world=world)
    entity = world.add_entities({comp1: {'a': 1}})
    assert len(filt.ids) == 0
    world.give(entity, {comp2: {'b': 4}})
    assert all(filt.ids == [0])
    world.give(entity, {comp2: {'b': 6}})
    assert len(filt.ids) == 1
    assert all(filt.ids == [0])

    assert (filt.index == filt.ids).all()
