"""Declaring components."""

from typing import Union

import numpy as np
import pandas as pd

from pd_ecs._filter_ops import Exclude


class Component:
    """Components of entities in the world.

    A component represents a variable corresponding to a given entity.
    Think of each component as a column, and each entity as a row.
    """

    name: str
    dtype: Union[str, np.dtype, pd.api.extensions.ExtensionDtype]

    def __init__(self, name, dtype=None):
        self.name = name
        self.dtype = dtype

    def __repr__(self):
        """Convert to string for representation."""
        return f"{self.name} Component"

    def __invert__(self):
        """Specify entities without this component.

        When indexing the world, you can do:
            ```
            entities_wo_2 = world[[component1, ~component2]]
            ```
        To get component1 of entities that do not also have component 2.
        """
        return Exclude(self)

    def init_series(self):
        """Initialize a series for this component."""
        return pd.Series([], dtype=self.dtype or np.int64, name=self)

    def __lt__(self, other):
        if isinstance(other, Component):
            return self.name < other.name
