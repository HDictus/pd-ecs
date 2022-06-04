import pandas as pd
from pd_ecs import System, Component, World
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
    pd.testing.assert_frame_equal(
        sys.banana,
        pd.concat([mockwld[comp1].loc[[1, 2, 3]],
                   mockwld[comp2].loc[[1, 2, 3]]],
                   keys={comp1: mockwld[comp1],
                         comp2: mockwld[comp2]}))
