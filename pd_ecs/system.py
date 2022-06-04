import pandas as pd
from .component import Component
from .filter import Filter


class System:

    filters = {}

    def __init__(self, world):
        world.add_system(self)
        self.world = world
        self._filters = {
            filt: Filter(*comps, world=world)
            for filt, comps in self.filters.items()}

    def __getattr__(self, key):
        """
        A System contains various filters, represented as dataframes
        """
        if key in self._filters:
            return self._filters[key].dataframe()
        return super().__getattr__(self, key)
