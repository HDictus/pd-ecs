import numpy as np
import pandas as pd


class Filter:

    def __init__(self, *components, world):
        self._comps = components
        self.world = world
        return

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
