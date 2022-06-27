"""
The Filter object filters entities and components by specified criteria.
"""
import numpy as np
import pandas as pd
from lazy import lazy


class _LocWrapper:

    def __init__(self, _filter, loc, frame):
        self._filter = _filter
        self._frame = frame
        self._loc = loc

    def __getattr__(self, key):
        return getattr(self._loc, key)

    def __setitem__(self, key, value):
        self._loc[key] = value
        self._filter.update_world(self._frame)

    def __getitem__(self, key):
        self._filter.update_filteredframe(self._frame)
        return self._loc[key]


class _FilteredFrame(pd.DataFrame):

    def __init__(self, data, filt):
        self.filt = filt
        self._update = True
        super().__init__(data)

    def __setattr__(self, key, value):
        if key not in ('filt', '_update', '_loc'):
            super().__setattr__(key, value)
        else:
            object.__setattr__(self, key, value)

    @property
    def loc(self):
        return _LocWrapper(self.filt, super().loc, self)

    def __setitem__(self, key, value):
        update = self._update
        self._update = False
        super().__setitem__(key, value)
        if update:
            self.filt.update_world(self)
        self._update = update

    def __getitem__(self, key):
        if self._update:
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
        filtframe._update = False
        for col in filtframe:
            self.world[col[0]].loc[self.ids, col[1:]] = filtframe[col]
        filtframe._update = True

    def update_filteredframe(self, filtframe):
        ids = self.ids
        for comp, frame in self.tracked_data.items():
            for col in frame:
                filtframe[(comp, col)] = frame.loc[ids, col]

    @lazy
    def dataframe(self):
        """
        The data for all components in the filter,
        for the entities which have all these components
        """
        ids = self.ids
        ff = _FilteredFrame({(c, f): frame.loc[ids, f]
                             for c, frame in self.tracked_data.items()
                             for f in frame},
                            self)
        return ff
