import pandas as pd
import numpy as np
from pd_ecs._filter_ops import Exclude


class ArchetypeStore:

    def __init__(self, dtype=np.uint32):
        self.series = pd.Series(dtype=dtype)
        self._component_powers = {}
        self._dtype = dtype
        self._arch_counts = {}
        # comp -> (archs, starts, stops): sorted int64 arrays, always current
        self._ranges = {}

    def add_entities(self, eids):
        """Register entities with the archetype store.

        This adds the entity ids to the underlying series
        mapping entity id to archetype id.
        """
        eids = self._validate_eids(eids)
        overlap = self.series.index.intersection(eids)
        if len(overlap):
            raise ValueError(f"entities already exist: {overlap}")
        new_series = pd.Series(0, index=eids, dtype=self._dtype)
        # Skip sort when new eids are already ordered and all follow existing ones.
        # World.add_entities always passes np.arange(maxind, ...) which satisfies this.
        self.series = pd.concat([self.series, new_series])
        # arch 0 has no component bits; it never appears in _ranges
        self._arch_counts[0] = self._arch_counts.get(0, 0) + len(eids)

    def _validate_eids(self, eids):
        eids = self._coerce_eids(eids)
        if len(np.unique(eids)) < len(eids):
            raise ValueError(f"duplicate eids in input")
        return eids

    def _coerce_eids(self, eids):
        return np.asarray([eids]) if np.isscalar(eids) else np.asarray(eids)

    def _positions(self, eids):
        """Return positional indices for eids, raising KeyError for any missing."""
        pos = self.series.index.get_indexer(eids)
        missing = eids[pos < 0]
        if len(missing):
            raise KeyError(missing.tolist())
        return pos

    def _range_add(self, comp, arch, count):
        """Add count entities to archetype range in component

        Each component maintains a (start, stop) range for each archetype.
        This method adds entities to the relevant ranges for one component.
        arch and

        Arguments:
            comp: a Component for which to add the entities
            arch: Archetypes for which to add entities
            count: number of entities to add to the range
        """
        _ensure_comp_in_ranges(self._ranges, comp)
        archs, starts, stops = self._ranges[comp]
        pos = int(np.searchsorted(archs, arch))
        if pos < len(archs) and archs[pos] == arch:
            # Existing archetype
            stops[pos:] += count
            starts[pos + 1:] += count
            return
        self._ranges[comp] = _add_archetypeto_ranges(
            archs, starts, stops, pos, count, arch
        )

    def _range_remove(self, comp, arch, count):
        """Remove entities from archetype range"""
        archs, starts, stops = self._ranges[comp]
        pos = int(np.searchsorted(archs, arch))
        stops[pos:] -= count
        starts[pos + 1:] -= count
        if stops[pos] == starts[pos]:
            self._ranges[comp] = (
                np.delete(archs, pos),
                np.delete(starts, pos),
                np.delete(stops, pos),
            )

    def range_lookup(self, comp, arch):
        """Return (start, stop) for arch in comp's ranges."""
        archs, starts, stops = self._ranges[comp]
        pos = int(np.searchsorted(archs, arch))
        return int(starts[pos]), int(stops[pos])

    def range_snapshot(self, comps):
        """Copy the current (archs, starts, stops) arrays for `comps`.

        `_range_add`/`_range_remove` mutate the live arrays in place for an
        already-present archetype, so a plain reference isn't a safe "before
        this batch" snapshot -- callers that need to look up boundaries as
        they stood before a series of archetype mutations (e.g. a multi-hop
        `give`) should snapshot first and look up via `range_lookup_in`.
        """
        empty = (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
        return {
            comp: tuple(arr.copy() for arr in self._ranges.get(comp, empty))
            for comp in comps
        }

    def range_lookup_in(self, snapshot, arch):
        """Like `range_lookup`, but against a snapshot from `range_snapshot`.

        If `arch` wasn't present in the snapshot, returns an empty range at
        the boundary where it would sort in -- the natural "nothing here yet"
        answer for an archetype that only came into existence after the
        snapshot was taken.
        """
        archs, starts, stops = snapshot
        pos = int(np.searchsorted(archs, arch))
        if pos < len(archs) and archs[pos] == arch:
            return int(starts[pos]), int(stops[pos])
        boundary = int(stops[pos - 1]) if pos > 0 else 0
        return boundary, boundary

    def _compute_transitions(self, old_values, new_values):
        """Return {(old_arch, new_arch): count} for entities whose arch changed."""
        changed = old_values != new_values
        if not np.any(changed):
            return {}
        old_values, new_values = old_values[changed], new_values[changed]
        if old_values.min() == old_values.max():
            return {(int(old_values[0]), int(new_values[0])): len(old_values)}
        pairs = np.stack([old_values, new_values], axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        return {(int(a), int(b)): int(c) for (a, b), c in zip(unique_pairs, counts)}

    def _ensure_power(self, component):
        """Return the power-of-2 bitmask for component, registering it if needed."""
        if component not in self._component_powers:
            next_bit = len(self._component_powers)
            if next_bit >= np.iinfo(self._dtype).bits:
                raise OverflowError(
                    f"cannot add component {component!r}: "
                    f"dtype {self._dtype} only supports {np.iinfo(self._dtype).bits} components"
                )
            self._component_powers[component] = self._dtype(2 ** next_bit)
        return self._component_powers[component]

    def add_component(self, eids, component):
        eids = self._coerce_eids(eids)
        if len(eids) == 0:
            return 0
        powerof2 = self._ensure_power(component)
        positions = self._positions(eids)
        old_values = self.series.values[positions].copy()
        self.series.values[positions] |= powerof2
        new_values = self.series.values[positions]
        # this may be simpler vectorized
        for (old_arch, new_arch), k in self._compute_transitions(old_values, new_values).items():
            self._update_archetype(component, old_arch, new_arch, k)
            self._range_add(component, new_arch, k)

    def _update_archetype(self, component, old_arch, new_arch, k):
        self._arch_counts[old_arch] -= k
        if self._arch_counts[old_arch] == 0:
            del self._arch_counts[old_arch]
        self._arch_counts[new_arch] = self._arch_counts.get(new_arch, 0) + k
        for c, pw2_c in self._component_powers.items():
            if c is not component and old_arch & pw2_c:
                self._range_remove(c, old_arch, k)
                self._range_add(c, new_arch, k)

    @property
    def _test_ranges(self):
        return {
            comp: {int(a): (int(s), int(e)) for a, s, e in zip(*data)}
            for comp, data in self._ranges.items()
        }

    def remove_entities(self, eids):
        eids = self._validate_eids(eids)
        missing = np.setdiff1d(eids, self.series.index.to_numpy(), assume_unique=True)
        if len(missing):
            raise KeyError(f"entities do not exist: {missing.tolist()}")
        old_values = self.series.loc[eids].values
        for arch, cnt in zip(*np.unique(old_values, return_counts=True)):
            self._arch_counts[arch] -= cnt
            if self._arch_counts[arch] == 0:
                del self._arch_counts[arch]
            self._remove_from_ranges_for_arch(arch, cnt)
        self.series = self.series.drop(index=eids)

    def _remove_from_ranges_for_arch(self, arch, count):
        """Remove count entities from each range relevant to archetype."""
        for c, pw2_c in self._component_powers.items():
            if arch & pw2_c:
                self._range_remove(c, arch, count)

    def remove_component(self, eids, component):
        eids = self._coerce_eids(eids)
        if len(eids) == 0:
            return
        if component not in self._component_powers:
            raise KeyError(component)
        powerof2 = self._component_powers[component]
        positions = self._positions(eids)
        old_values = self.series.values[positions].copy()
        # Entities that don't have the component are silently skipped (&= is a no-op on zero bits).
        self.series.values[positions] &= ~powerof2
        new_values = self.series.values[positions]
        for (old_arch, new_arch), k in self._compute_transitions(old_values, new_values).items():
            self._update_archetype(component, old_arch, new_arch, k)
            self._range_remove(component, old_arch, k)

    def choose_archetypes(self, filt):
        """Select archetype numbers corresponding to filter.

        Return the archetype ids corresponding to a given filter.
        This excludes archetypes not present in the data.

        Arguments:
           filt: list of components or component negations
        Returns
          np.array of archetype ids
        """
        archetypes = np.array(sorted(self._arch_counts), dtype=self._dtype)
        for comp in filt:
            if isinstance(comp, Exclude):
                pw2 = self._ensure_power(comp.component)
                archetypes = archetypes[(archetypes & pw2) == 0]
                continue
            pw2 = self._ensure_power(comp)
            archetypes = archetypes[(archetypes & pw2) != 0]
        return archetypes


def _ensure_comp_in_ranges(ranges, comp):
    if comp not in ranges:
        ranges[comp] = (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )


def _add_archetypeto_ranges(
        archs,
        starts,
        stops,
        pos,
        count,
        arch
):
    new_start = int(stops[pos - 1]) if pos > 0 else 0
    new_archs = np.insert(archs, pos, arch)
    new_starts = np.insert(starts, pos, new_start)
    new_stops = np.insert(stops, pos, new_start + count)
    new_starts[pos + 1:] += count
    new_stops[pos + 1:] += count
    return new_archs, new_starts, new_stops
