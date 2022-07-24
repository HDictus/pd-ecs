"""
The Filter object filters entities and components by specified criteria.
"""
import numpy as np
import pandas as pd
from lazy import lazy

# TODO: it might be simpler to put the functionality of filter in filteredframe/_locwrapper
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

        for comp in self.components:
            self.world.filters_by_component[comp].append(self)

    def add_components(self, component, ids):
        """
        entities ids have had <component> added, check if they belong in
        the filter now
        """
        for comp in self.components:
            ids = np.intersect1d(self.world[comp].index, ids)
            if len(ids) == 0:
                return

        self.ids = np.concatenate([self.ids, ids])

    def remove_components(self, component, ids):
        """
        entities have had ids removed, they no longer belong in this list
        """
        toremove = np.isin(self.ids, ids)
        self.ids = self.ids[~toremove]

    def data(self):
        return tuple(self[comp] for comp in self.components)

    def __getitem__(self, comp):
        return self.world[comp].loc[self.ids]
