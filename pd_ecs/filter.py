import numpy as np


class Filter:

    def __init__(self, *components, world):
        self._comps = components
        self.world = world
        return

    @property
    def ids(self):
        ids = self.world[self._comps[0]].index
        for comp in self._comps[1:]:
            ids = ids[np.isin(ids, self.world[comp].index)]
        return ids

    def __getitem__(self, key):
        if key in self._comps:
            c = self.world[key]
            return c.loc[self.ids]
        return KeyError

    def __contains__(self, key):
        return key in self._comps
