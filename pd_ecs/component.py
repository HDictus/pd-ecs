"""Declaring components."""
import pandas as pd
from pd_ecs._filter_ops import Exclude

unnamed_components: list = []


class Component:
    """
    A component represents a set of variables belonging to a given entity.
    In a component dataframe each variable is a column and each row an entity.
    """

    def __init__(self, *fields, name=None, **kw_fields):
        """
        Arguments:
            fields: names of component variables (str)
        """
        self.fields = list(fields)
        if name is None:
            name = f'unnamed{len(unnamed_components)}'
            unnamed_components.append(self)
        self.name = name
        for field in fields:
            if hasattr(self, field):
                raise ValueError(
                    f"Protected attribute: {field}."
                    "if you wish to use this field name, "
                    "pass it as a keyword argument. "
                    "The key will become the attribute name")
            setattr(self, field, (self, field))
        for attrname, field in kw_fields.items():
            self.fields.append(field)
            if hasattr(self, attrname):
                raise ValueError(
                    f"Attribute: {attrname} is already in use"
                    " for this Component.")
            setattr(self, attrname, (self, field))

    def init_dataframe(self):
        """
        Returns an empty dataframe with the field columns.
        """
        return pd.DataFrame(columns=self.fields, dtype=int)

    def __repr__(self):
        return f'{self.name} Component'



    def __invert__(self):
        return Exclude(self)
