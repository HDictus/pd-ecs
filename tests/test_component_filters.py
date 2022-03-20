import pandas as pd
import numpy as np
from pd_ecs import Component
from pd_ecs.filter import Filter


def test_fiter_filters_out_ones_that_dont_have():

    comp1 = Component('a')
    comp2 = Component('b')
    comp3 = Component('c')

    mockwld = {
        comp1: pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7],
                             'a': [0, 1, 2, 3, 4, 5, 6, 7]}).set_index('id'),
        comp2: pd.DataFrame({'id': [1, 2, 3, 10],
                             'b': [1, 2, 3, 10]}).set_index('id'),
        comp3: pd.DataFrame({'id': [0, 1, 2, 4, 5, 6],
                             'c': [1, 2, 3, 4, 5, 6]}).set_index('id')}

    filt = Filter(comp1, comp2, world=mockwld)

    assert np.allclose(filt.ids, [1, 2, 3])
    return


# def test_filter_lets
