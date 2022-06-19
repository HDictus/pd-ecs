"""
The Filter object filters entities and components by specified criteria.
"""
import numpy as np
import pandas as pd


# class _MocLoc:

#     def __init__(self, _filter):
#         self._filter = _filter

#     def __getitem__(self, key):
#         return self._filter.dataframe.loc[key]

#     def __setitem__(self, key, value):
#         for c, frame in self._filter.tracked_data.items():
#             frame.loc[key] = value


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
        self.tracked_data = {
            comp: world[comp]
            for comp in components}

    @property
    def loc(self):
        return pd.core.indexing._LocIndexer('loc', self)

    @property
    def iloc(self):
        return pd.core.indexing._iLocIndexer('iloc', self)

    @property
    def ids(self):
        """the ids of entities in this filter"""
        ids = self.world[self._comps[0]].index
        for comp in self._comps[1:]:
            ids = ids[np.isin(ids, self.world[comp].index)]
        return ids

    @property
    def dataframe(self):
        """
        The data for all components in the filter,
        for the entities which have all these components
        """
        ids = self.ids
        data = pd.DataFrame({
            (comp, field): values[ids]
            for comp, frame in self.tracked_data.items()
            for field, values in frame.items()})
        return data

    def __getattr__(self, attr):
        return getattr(self.dataframe, attr)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.tracked_data[key[0]].loc[self.ids, key[1:]] = value
        else:
            self.tracked_data[key].loc[self.ids] = value

    def __getitem__(self, key):
        return self.dataframe.loc[self.ids, key]

    def __repr__(self):
        return repr(self.dataframe)
