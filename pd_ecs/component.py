"""Declaring components."""
import pandas as pd
import numpy as np
from typing import Union
from pd_ecs._filter_ops import Exclude

unnamed_components: list = []


class Component:
    """
    A component represents a variable to a given entity.
    In a world dataframe each component is a column and each row an entity.
    """

    name: str
    dtype: Union[str, np.dtype, pd.api.extensions.ExtensionDtype]

    def __init__(self, name, dtype=object):
        """
        Arguments:
            fields: names of component variables (str)
        """
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        return f'{self.name} Component'

    def __invert__(self):
        return Exclude(self)

    def init_series(self):
        return pd.Series([], dtype=self.dtype, name=self)