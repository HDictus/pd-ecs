import pandas as pd
from pd_ecs import System, Component, World
import numpy as np
import pytest as pyt
from mock import MagicMock


def test_system_initialized_adds_to_world():
    mockworld = MagicMock()
    sys = System(mockworld)
    assert mockworld.add_system.called_with(sys)
    assert sys.world == mockworld


def test_system_has_filters():
    comp1 = Component('a')
    comp2 = Component('b')
    comp3 = Component('c')

    mockwld = World(comp1, comp2, comp3)
    mockwld.set_state({
        comp1: pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7],
                             'a': [0, 1, 2, 3, 4, 5, 6, 7]}).set_index('id'),
        comp2: pd.DataFrame({'id': [1, 2, 3, 10],
                             'b': [1, 2, 3, 10]}).set_index('id'),
        comp3: pd.DataFrame({'id': [0, 1, 2, 4, 5, 6],
                             'c': [1, 2, 3, 4, 5, 6]}).set_index('id')})

    class MySys(System):

        filters = dict(banana=[comp1, comp2])

    sys = MySys(mockwld)
    c1, c2 = sys.banana.data()
    pd.testing.assert_frame_equal(c1, pd.DataFrame({'a': [1, 2, 3]},
                                                   index=[1, 2, 3]))
    pd.testing.assert_frame_equal(c2, pd.DataFrame({'b': [1, 2, 3]},
                                                   index=[1, 2, 3]))




def test_system_filter_can_update():

    comp1 = Component('a', name='1')
    comp2 = Component('a', name='2')
    comp3 = Component('c', name='3')

    mockwld = World(comp1, comp2, comp3)
    mockwld.set_state({
        comp1: pd.DataFrame({'id': [],
                             'a': []}).set_index('id'),
        comp2: pd.DataFrame({'id': [1, 2, 3, 10],
                             'a': [1, 2, 3, 10]}).set_index('id'),
        comp3: pd.DataFrame({'id': [0, 1, 2, 4, 5, 6],
                             'c': [1, 2, 3, 4, 5, 6]}).set_index('id')})

    class MySys(System):

        filters = dict(banana=[comp1, comp2])

    sys = MySys(mockwld)
    sys.banana.data()
    mockwld.give([1], {comp1: {'a': [1]}})
    c1, c2 = sys.banana.data()
    pd.testing.assert_frame_equal(c1, pd.DataFrame({'a': [1.]}, index=[1]))
    pd.testing.assert_frame_equal(c2, pd.DataFrame({'a': [1]}, index=[1]))

    mockwld.take([1], comp1)
    c1, c2 = sys.banana.data()
    pd.testing.assert_frame_equal(c1, pd.DataFrame({'a': []}, dtype=float))
    pd.testing.assert_frame_equal(c2, pd.DataFrame({'a': []}, dtype=int))
