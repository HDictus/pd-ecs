"""
The Filter object filters entities and components by specified criteria.
"""
import numpy as np
import pandas as pd


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
        self.add_components(
            components, self.world[self.components[0]].index)

        self.world.add_filter(self, components)

    def add_components(self, component, ids):
        """
        entities ids have had <component> added, check if they belong in
        the filter now
        """
        for comp in self.components:
            if comp == component:
                continue
            ids = np.intersect1d(self.world[comp].index, ids)
            if len(ids) == 0:
                return

        self.ids = np.unique(np.concatenate([self.ids, ids]))

    def remove_components(self, _, ids):
        """
        entities have had ids removed, they no longer belong in this list
        """
        toremove = np.isin(self.ids, ids)
        self.ids = self.ids[~toremove]

    @property
    def index(self):
        return self.ids

    def data(self):
        """Return the dataframes for the filtered components"""
        return tuple(self[comp] for comp in self.components)

    def __getitem__(self, comp):
        return self.world[comp].loc[self.ids]

    def multi_frame(self):
        return pd.DataFrame({(component, field): df[field]
                             for component, df in zip(self.components, self.data())
                             for field in df.columns},
                            index=self.ids)
