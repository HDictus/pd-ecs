import pandas as pd
import numpy as np


class ArchetypeManager:

    def __init__(self):
        self.series = pd.Series()
        self.component_powers = {}

    def add_entities(self, entities_df):
        # TODO: the form for storage of an archetype is diddifult to segragate into one place
        archetype = tuple(sorted(entities_df.keys()))
        bitmask = self._archetype_to_bitmask(archetype)
        new_entries = pd.Series(bitmask, index=entities_df.index)
        self.series = pd.concat([self.series, new_entries])
        return archetype

    def _archetype_to_bitmask(self, archetype):
        bm = np.int64(0)
        for comp in archetype:
            pow2 = self._get_power(comp)
            bm |= pow2
        return bm

    def _get_power(self, comp):
        if comp not in self.component_powers:
            self.component_powers[comp] = 2**len(self.component_powers)
        return self.component_powers[comp]

    def give(self, entities, components):
        entity_atypes = self.series.loc[entities]
        new_atypes = entity_atypes | self._archetype_to_bitmask(components)
        self.series[entities] = new_atypes
        for nat, oat in entity_atypes.groupby(new_atypes):
            yield (
                self._bitmask_to_archetype(oat.iloc[0]),
                self._bitmask_to_archetype(nat)
            ), oat.index

    def _bitmask_to_archetype(self, bitmask):
        # TODO: best mitigated by caching
        out = []
        for comp, pow2 in self.component_powers.items():
            if bitmask & pow2:
                out.append(comp)
        return tuple(sorted(out))

    def take(self, entities, components):
        entity_atypes = self.series.loc[entities]
        new_atypes = entity_atypes & ~self._archetype_to_bitmask(components)
        self.series[entities] = new_atypes
        # TODO: some code duplication here
        for nat, oat in entity_atypes.groupby(new_atypes):
            if nat == oat.values[0]:
                continue
            yield (
                self._bitmask_to_archetype(oat.iloc[0]),
                self._bitmask_to_archetype(nat)
                ), oat.index

    def group(self, entities):
        """Group entities by archetype and iterate over groups."""
        archetypes = self.series[entities]
        for at, atypes in archetypes.groupby(archetypes):
            yield self._bitmask_to_archetype(at), atypes.index

    def remove_entities(self, entities):
        self.series = self.series.drop(entities)

    def get(self, entity):
        return self._bitmask_to_archetype(self.series[entity])
