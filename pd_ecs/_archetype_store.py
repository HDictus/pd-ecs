import pandas as pd
import numpy as np


class ArchetypeStore:

    def __init__(self, dtype=np.uint32):
        self.series = pd.Series(dtype=dtype)
        self._component_powers = {}
        self._dtype = dtype

    def add_entities(self, eids):
        if np.isscalar(eids):
            eids = [eids]
        if len(np.unique(eids)) < len(eids):
            raise ValueError(f"duplicate eids in input")
        overlap = self.series.index.intersection(eids)
        if len(overlap):
            raise ValueError(f"entities already exist: {overlap}")
        self.series = pd.concat([
            self.series,
            pd.Series(0, index=eids, dtype=self._dtype)
        ]).sort_index()

    def _validate_eids(self, eids):
        if np.isscalar(eids):
            eids = [eids]
        return eids

    def _positions(self, eids):
        """Return positional indices for eids, raising KeyError for any missing."""
        pos = self.series.index.get_indexer(eids)
        missing = [e for e, p in zip(eids, pos) if p < 0]
        if missing:
            raise KeyError(missing)
        return pos

    def add_component(self, eids, component):
        eids = self._validate_eids(eids)
        if component in self._component_powers:
            powerof2 = self._component_powers[component]
        else:
            next_bit = len(self._component_powers)
            if next_bit >= np.iinfo(self._dtype).bits:
                raise OverflowError(
                    f"cannot add component {component!r}: "
                    f"dtype {self._dtype} only supports {np.iinfo(self._dtype).bits} components"
                )
            powerof2 = self._dtype(2 ** next_bit)
            self._component_powers[component] = powerof2
        self.series.values[self._positions(eids)] |= powerof2

    def remove_entities(self, eids):
        if np.isscalar(eids):
            eids = [eids]
        if len(np.unique(eids)) < len(eids):
            raise ValueError(f"duplicate eids in input")
        missing = pd.Index(eids).difference(self.series.index)
        if len(missing):
            raise KeyError(f"entities do not exist: {list(missing)}")
        self.series = self.series.drop(index=eids)

    def remove_component(self, eids, component):
        eids = self._validate_eids(eids)
        if component not in self._component_powers:
            raise KeyError(component)
        powerof2 = self._component_powers[component]
        # Entities that don't have the component are silently skipped (&= is a no-op on zero bits).
        self.series.values[self._positions(eids)] &= ~powerof2
