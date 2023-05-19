"""Declaring components."""
import pandas as pd
from pd_ecs._filter_ops import Exclude

unnamed_components: list = []


# pylint: disable=too-few-public-methods
class Component:
    """
    A component represents a set of variables belonging to a given entity.
    In a component dataframe each variable is a column and each row an entity.
    """

    def __init__(self, *fields, name=None):
        """
        Arguments:
            fields: names of component variables (str)
        """
        self.fields = fields
        if name is None:
            name = f'unnamed{len(unnamed_components)}'
            unnamed_components.append(self)
        self.name = name

    def init_dataframe(self):
        """
        Returns an empty dataframe with the field columns.
        """
        return pd.DataFrame(columns=self.fields, dtype=int)

    def __repr__(self):
        return f'{self.name} Component'

    def __invert__(self):
        return Exclude(self)
