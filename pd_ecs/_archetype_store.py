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
        eids = np.asarray([eids]) if np.isscalar(eids) else np.asarray(eids)
        if len(np.unique(eids)) < len(eids):
            raise ValueError(f"duplicate eids in input")
        overlap = self.series.index.intersection(eids)
        if len(overlap):
            raise ValueError(f"entities already exist: {overlap}")
        new_series = pd.Series(0, index=eids, dtype=self._dtype)
        # Skip sort when new eids are already ordered and all follow existing ones.
        # World.add_entities always passes np.arange(maxind, ...) which satisfies this.
        if len(self.series) == 0 or (
            np.all(np.diff(eids) >= 0) and eids[0] > self.series.index[-1]
        ):
            self.series = pd.concat([self.series, new_series])
        else:
            self.series = pd.concat([self.series, new_series]).sort_index()
        # arch 0 has no component bits; it never appears in _ranges
        self._arch_counts[0] = self._arch_counts.get(0, 0) + len(eids)

    def _validate_eids(self, eids):
        return np.asarray([eids]) if np.isscalar(eids) else np.asarray(eids)

    def _positions(self, eids):
        """Return positional indices for eids, raising KeyError for any missing."""
        pos = self.series.index.get_indexer(eids)
        missing = eids[pos < 0]
        if len(missing):
            raise KeyError(missing.tolist())
        return pos

    def _range_add(self, comp, arch, count):
        """Add count entities to archetype arch in ranges for comp."""
        if comp not in self._ranges:
            self._ranges[comp] = (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        archs, starts, stops = self._ranges[comp]
        pos = int(np.searchsorted(archs, arch))
        if pos < len(archs) and archs[pos] == arch:
            # Existing archetype: shift stop of this arch and start/stop of all following
            stops[pos:] += count
            starts[pos + 1:] += count
        else:
            # New archetype: insert sorted, then shift everything after it
            new_start = int(stops[pos - 1]) if pos > 0 else 0
            new_archs = np.insert(archs, pos, arch)
            new_starts = np.insert(starts, pos, new_start)
            new_stops = np.insert(stops, pos, new_start + count)
            new_starts[pos + 1:] += count
            new_stops[pos + 1:] += count
            self._ranges[comp] = (new_archs, new_starts, new_stops)

    def _range_remove(self, comp, arch, count):
        """Remove count entities from archetype arch in ranges for comp."""
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

    def _range_lookup(self, comp, arch):
        """Return (start, stop) for arch in comp's ranges."""
        archs, starts, stops = self._ranges[comp]
        pos = int(np.searchsorted(archs, arch))
        return int(starts[pos]), int(stops[pos])

    def _compute_transitions(self, old_values, new_values):
        """Return {(old_arch, new_arch): count} for entities whose arch changed."""
        changed = old_values != new_values
        if not np.any(changed):
            return {}
        o, n = old_values[changed], new_values[changed]
        # Fast path: single archetype transition (all entities share the same old arch).
        # This is the common case in bulk add_component / remove_component calls.
        if o.min() == o.max():
            return {(int(o[0]), int(n[0])): len(o)}
        pairs = np.stack([o, n], axis=1)
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
        eids = self._validate_eids(eids)
        powerof2 = self._ensure_power(component)
        positions = self._positions(eids)
        old_values = self.series.values[positions].copy()
        self.series.values[positions] |= powerof2
        new_values = self.series.values[positions]
        for (old_arch, new_arch), k in self._compute_transitions(old_values, new_values).items():
            old_int, new_int = int(old_arch), int(new_arch)
            # Update arch counts (replaces _apply_arch_counts)
            self._arch_counts[old_int] -= k
            if self._arch_counts[old_int] == 0:
                del self._arch_counts[old_int]
            self._arch_counts[new_int] = self._arch_counts.get(new_int, 0) + k
            # component is new for these entities: add to new_arch only
            self._range_add(component, new_arch, k)
            # all other components already on these entities: transfer old_arch -> new_arch
            for c, pw2_c in self._component_powers.items():
                if c is not component and old_arch & pw2_c:
                    self._range_remove(c, old_arch, k)
                    self._range_add(c, new_arch, k)

    @property
    def ranges(self):
        return {
            comp: {int(a): (int(s), int(e)) for a, s, e in zip(*data)}
            for comp, data in self._ranges.items()
        }

    @property
    def archetype_counts(self):
        keys = np.array(sorted(self._arch_counts), dtype=self._dtype)
        return pd.Series(
            np.array([self._arch_counts[int(k)] for k in keys], dtype=np.intp),
            index=pd.Index(keys, dtype=self._dtype)
        )

    def remove_entities(self, eids):
        eids = np.asarray([eids]) if np.isscalar(eids) else np.asarray(eids)
        if len(np.unique(eids)) < len(eids):
            raise ValueError(f"duplicate eids in input")
        missing = np.setdiff1d(eids, self.series.index.to_numpy(), assume_unique=True)
        if len(missing):
            raise KeyError(f"entities do not exist: {missing.tolist()}")
        old_values = self.series.loc[eids].values
        self.series = self.series.drop(index=eids)
        # Single np.unique pass handles both arch_counts and ranges updates
        if len(old_values):
            for arch, cnt in zip(*np.unique(old_values, return_counts=True)):
                arch_int, cnt_int = int(arch), int(cnt)
                self._arch_counts[arch_int] -= cnt_int
                if self._arch_counts[arch_int] == 0:
                    del self._arch_counts[arch_int]
                for c, pw2_c in self._component_powers.items():
                    if arch_int & pw2_c:
                        self._range_remove(c, arch_int, cnt_int)

    def remove_component(self, eids, component):
        eids = self._validate_eids(eids)
        if component not in self._component_powers:
            raise KeyError(component)
        powerof2 = self._component_powers[component]
        positions = self._positions(eids)
        old_values = self.series.values[positions].copy()
        # Entities that don't have the component are silently skipped (&= is a no-op on zero bits).
        self.series.values[positions] &= ~powerof2
        new_values = self.series.values[positions]
        for (old_arch, new_arch), k in self._compute_transitions(old_values, new_values).items():
            old_int, new_int = int(old_arch), int(new_arch)
            # Update arch counts (replaces _apply_arch_counts)
            self._arch_counts[old_int] -= k
            if self._arch_counts[old_int] == 0:
                del self._arch_counts[old_int]
            self._arch_counts[new_int] = self._arch_counts.get(new_int, 0) + k
            # component is being removed: drop from old_arch only
            self._range_remove(component, old_arch, k)
            # all other components on these entities: transfer old_arch -> new_arch
            for c, pw2_c in self._component_powers.items():
                if c is not component and old_arch & pw2_c:
                    self._range_remove(c, old_arch, k)
                    self._range_add(c, new_arch, k)

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
