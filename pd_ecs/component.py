"""Declaring components."""
import pandas as pd


# pylint: disable=too-few-public-methods
class Component:
    """
    A component represents a set of variables belonging to a given entity.
    In a component dataframe each variable is a column and each row an entity.
    """

    def __init__(self, *fields):
        """
        Arguments:
            fields: names of component variables (str)
        """
        self.fields = fields

    def init_dataframe(self):
        """
        Returns an empty dataframe with the field columns.
        """
        return pd.DataFrame(columns=self.fields, dtype=int)
