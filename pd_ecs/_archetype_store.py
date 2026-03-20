import pandas as pd
import numpy as np


class ArchetypeStore:

    def __init__(self, dtype=np.uint32):
        self.series = pd.Series(dtype=dtype)
        self._component_powers = {}
        self._dtype = dtype

    def add_entities(self, eid):
        if eid in self.series.index:
            raise ValueError(f"entity {eid!r} already exists")
        self.series = pd.concat([
            self.series,
            pd.Series(0, index=[eid], dtype=self._dtype)
        ])

    def add_component(self, eid, component):
        if eid not in self.series.index:
            raise KeyError(eid)
        if component in self._component_powers:
            powerof2 = self._component_powers[component]
        else:
            next_bit = len(self._component_powers)
            if next_bit >= np.iinfo(self._dtype).bits:
                raise OverflowError(
                    f"cannot add component {component!r}: "
                    f"dtype {self._dtype} only supports {np.iinfo(self._dtype).bits} components"
                )
            powerof2 = 2 ** next_bit
            self._component_powers[component] = powerof2
        if self.series[eid] & powerof2:
            return  # already present — idempotent
        self.series[eid] += powerof2

    def remove_component(self, eid, component):
        if eid not in self.series.index:
            raise KeyError(eid)
        if component not in self._component_powers:
            raise KeyError(component)
        powerof2 = self._component_powers[component]
        if not (self.series[eid] & powerof2):
            raise ValueError(f"entity {eid!r} does not have component {component!r}")
        self.series[eid] -= powerof2
