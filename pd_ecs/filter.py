"""
The Filter object filters entities and components by specified criteria.
"""
import numpy as np
import pandas as pd
from pd_ecs._filter_ops import Exclude


class Filter:
    """Filter entities which have the specified components"""

    def __init__(self, *components, world):
        """
        Arguments:
            components: the components required to be part of this filter
            world: the world the filter belongs to
        """
        self.components = components
        self.world = world
        self.ids = np.array([], dtype=np.int32)
        self._components_added(
            components, self.world[self.components[0]].index)

    def _components_added(self, component, ids):
        """
        Entities ids have had <component> added, check if they belong in
        the filter now.
        """
        if Exclude(component) in self.components:
            just_added = np.isin(self.ids, ids)
            self.ids = self.ids[~just_added]
            return
        self._add_belonging_ids(ids)

    def _add_belonging_ids(self, ids):
        for comp in self.components:
            if isinstance(comp, Exclude):
                ids = ids[~np.isin(ids, self.world[comp.component].index)]
            else:
                ids = np.intersect1d(self.world[comp].index, ids)
            if len(ids) == 0:
                return
        self.ids = np.unique(np.concatenate([self.ids, ids]))

    def _components_removed(self, component, ids):
        """
        entities have had a component removed.
        check whether or not they belong in the filter
        """
        if Exclude(component) in self.components:
            self._add_belonging_ids(ids)
            return
        toremove = np.isin(self.ids, ids)
        self.ids = self.ids[~toremove]

    @property
    def index(self):
        """The index of the filtered data."""
        return self.ids

    def data(self):
        """Return the dataframes for the filtered components"""
        return tuple(self[comp] for comp in self.components)

    def __getitem__(self, comp):
        return self.world[comp].loc[self.ids]

    def multi_frame(self):
        """Get all the filtered components as a single dataframe.

        Warning: this method is rather slow, use it sparingly.

        Returns:
           a dataframe of the form:
           | component1       | component2 |  ....
           | field1 | field2  | field3     |  ....
           | value1 | value2  | value3
             ...       ...      ....

           The columns are a multiindex with first level corresponding to
           component types, and second level to the fields of those components
        """
        return pd.DataFrame(
            {
                (component, field): df[field]
                for component, df in zip(self.components, self.data())
                for field in df.columns
            },
            index=self.ids)
