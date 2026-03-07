import pandas as pd
import numpy as np
from pd_ecs import entity_collection as test_module


def test_ec_mutates_dataframes():

    df1 = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [1, 2, 3, 4]})
    df2 = pd.DataFrame({
        'c': [4, 5, 6],
        })
    collection = test_module.EntityCollection(df1, df2)
    pd.testing.assert_frame_equal(
        collection,
        pd.concat([df1, df2], axis=1, join='inner')
    )
    collection['a'] = 0
    assert np.allclose(df1.loc[collection.index, 'a'], 0)
    assert 'a' not in df2

    pd.testing.assert_frame_equal(
        collection,
        pd.concat([df1, df2], axis=1, join='inner')
    )

    collection.loc[[1, 2], 'c'] = 6
    assert np.allclose(df2.loc[[1, 2], 'c'], 6)

    pd.testing.assert_frame_equal(
        collection,
        pd.concat([df1, df2], axis=1, join='inner')
    )
