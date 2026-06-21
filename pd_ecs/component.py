"""Declaring components."""

from typing import Union

import numpy as np
import pandas as pd

from pd_ecs._filter_ops import Exclude

unnamed_components: list = []


class Component:
    """Components of entities in the world.

    A component represents a variable corresponding to a given entity.
    Think of each component as a column, and each entity as a row.
    """

    name: str
    dtype: Union[str, np.dtype, pd.api.extensions.ExtensionDtype]
    is_compound = False

    def __init__(self, name, dtype=object, **subcomponents):
        """Initialize a component type.

        Arguments:
            fields: names of component variables (str)
        """
        self.name = name
        self.dtype = dtype
        # TODO: find an elegant way to have this be reflected in
        #   static code analysis
        # each component instance should be a type object?
        # a named tuple or dataclass?
        self.subcomponents = {}
        for component_name, component in subcomponents.items():
            combination = (self, component)
            setattr(self, component_name, combination)
            self.subcomponents[component_name] = combination
            self.is_compound = True

    # pylint: disable=no-member,useless-parent-delegation
    def __getattr__(self, attr):
        """To satisfy linters."""
        return super().__getattr__(attr)

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
        return pd.Series([], dtype=self.dtype, name=self)
