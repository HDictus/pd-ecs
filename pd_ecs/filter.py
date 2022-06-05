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
        self._comps = components
        self.world = world

    @property
    def ids(self):
        """the ids of entities in this filter"""
        ids = self.world[self._comps[0]].index
        for comp in self._comps[1:]:
            ids = ids[np.isin(ids, self.world[comp].index)]
        return ids

    def dataframe(self):
        """
        The data for all components in the filter,
        for the entities which have all these components
        """
        ids = self.ids
        data = [
            self.world[comp].loc[ids]
            for comp in self._comps]
        return pd.concat(data, keys=self._comps)
