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
    exp = pd.DataFrame({
        (comp1, 'a'): [1, 2, 3],
        (comp2, 'b'): [1, 2, 3]})
    exp.index = [1, 2, 3]

    pd.testing.assert_frame_equal(sys.banana[:], exp)


def test_system_filter_modifies_world():
    comp1 = Component('a')
    comp2 = Component('a')
    comp3 = Component('c')

    mockwld = World(comp1, comp2, comp3)
    mockwld.set_state({
        comp1: pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7],
                             'a': [0, 1, 2, 3, 4, 5, 6, 7]}).set_index('id'),
        comp2: pd.DataFrame({'id': [1, 2, 3, 10],
                             'a': [1, 2, 3, 10]}).set_index('id'),
        comp3: pd.DataFrame({'id': [0, 1, 2, 4, 5, 6],
                             'c': [1, 2, 3, 4, 5, 6]}).set_index('id')})

    class MySys(System):

        filters = dict(banana=[comp1, comp2])

    sys = MySys(mockwld)

    sys.banana[(comp1, 'a')] = 30
    assert np.allclose(mockwld[comp1].loc[[1, 2, 3], 'a'].values, 30)
    sys.banana[comp1] += sys.banana[comp2]
    assert np.allclose(mockwld[comp1].loc[[1, 2, 3], 'a'],
                       [31, 32, 33])
    mockwld[comp1].loc[2] = 400
    assert sys.banana.loc[2, (comp1, 'a')] == 400

    sys.banana.loc[[2, 3]] = 44
    assert np.allclose(mockwld[comp2].loc[[2, 3]].values, 44)

    sys.banana.loc[[2, 3], [comp1]] = 432
    assert np.allclose(mockwld[comp1].loc[[2, 3]].values, 432)

    sys.banana.loc[[2, 3], [(comp1, 'a'), (comp2, 'a')]] = 110
    assert np.allclose(mockwld[comp1].loc[[2, 3]].values, 110)

    sys.banana[[comp1, comp2]] = 3000
    assert np.allclose(mockwld[comp2].loc[[1, 2, 3]].values, 3000)
    assert np.allclose(mockwld[comp1].loc[[1, 2, 3]].values, 3000)


@pyt.mark.xfail
def test_system_filter_copy_breaks_stuff():
    comp1 = Component('a')
    comp2 = Component('a')
    comp3 = Component('c')

    mockwld = World(comp1, comp2, comp3)
    mockwld.set_state({
        comp1: pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7],
                             'a': [0, 1, 2, 3, 4, 5, 6, 7]}).set_index('id'),
        comp2: pd.DataFrame({'id': [1, 2, 3, 10],
                             'a': [1, 2, 3, 10]}).set_index('id'),
        comp3: pd.DataFrame({'id': [0, 1, 2, 4, 5, 6],
                             'c': [1, 2, 3, 4, 5, 6]}).set_index('id')})

    class MySys(System):

        filters = dict(banana=[comp1, comp2])

    sys = MySys(mockwld)
    with pyt.raises(IndexError):
        sys.banana.loc[4, (comp1, 'a')] = 132


def test_system_filter_can_update():

    comp1 = Component('a')
    comp2 = Component('a')
    comp3 = Component('c')

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
    sys.banana
    mockwld.give([1], {comp1: {'a': [1]}})

    pd.testing.assert_frame_equal(
        sys.banana,
        pd.DataFrame({(comp1, 'a'): [1.], (comp2, 'a'): [1]}, index=[1]))
