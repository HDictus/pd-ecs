"""
The Filter object filters entities and components by specified criteria.
"""
import numpy as np
import pandas as pd
from lazy import lazy

# class _MocLoc:

#     def __init__(self, _filter):
#         self._filter = _filter

#     def __getitem__(self, key):
#         return self._filter.dataframe.loc[key]

#     def __setitem__(self, key, value):
#         for c, frame in self._filter.tracked_data.items():
#             frame.loc[key] = value


class _FilteredFrame(pd.DataFrame):

    def __init__(self, data, filt):
        self.filt = filt
        self._update = True
        super().__init__(data)

    def __setattr__(self, key, value):
        if key not in ('filt', '_update'):
            super().__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    def enable_update(self):
        self._update = True

    def disable_update(self):
        self._update = False

    def __setitem__(self, key, value):
        update = self._update
        self.disable_update()
        super().__setitem__(key, value)
        self.enable_update()
        if update:
            self.filt.update_world(self)

    def __getitem__(self, key):
        if self.update:
            self.filt.update_filteredframe(self)
        return super().__getitem__(key)


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

    @property
    def tracked_data(self):
        return {comp: self.world[comp] for comp in self._comps}

    def update_world(self, filtframe):
        filtframe.disable_update()
        for col in filtframe:
            self.world[col[0]].loc[self.ids, col[1:]] = filtframe[col]
        filtframe.enable_update()

    def update_filteredframe(self, filtframe):
        ids = self.ids
        filtframe.disable_update()
        for comp, frame in self.tracked_data.items():
            for col in frame:
                filtframe[(comp, col)] = frame.loc[ids, col]
        filtframe.enable_update()

    @lazy
    def dataframe(self):
        """
        The data for all components in the filter,
        for the entities which have all these components
        """
        ff = _FilteredFrame({
            (c, f): [] for c, frame in self.tracked_data.items() for f in frame},
                            self)
        self.update_filteredframe(ff)
        return ff
