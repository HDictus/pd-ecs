"""
The World stores and manages the state and events of the simulation.

It is defined as consisting of a certain set of component types.
Systems are added to the world.
World.events.<event_name> calls that event for all systems in the world
"""

from collections.abc import Iterable
from typing import Dict

import numpy as np
import pandas as pd
from lazy import lazy

from ._archetype_store import ArchetypeStore
from ._entity_view import EntityView
from ._filter_ops import Exclude
from .component import Component
from .exceptions import ComponentError


class LocIndexer:
    """Entity indexer, works like pandas .loc attribute."""

    def __init__(self, world):
        """Initialize the loc indexer for world."""
        self.world = world

    def __getitem__(self, key):
        """Get components of specific entities."""
        if not isinstance(key, tuple):
            raise ValueError(
                "Loc indexing without components is not supported.")
        return self.world[key[1]].loc[key[0]]

    def __setitem__(self, key, values):
        """Update/add components on entities."""
        if not isinstance(key, tuple):
            raise ValueError(
                "Loc indexing without components is not supported")
        index, columns = key

        if not isinstance(columns, list):
            columns = [columns]
        if pd.api.types.is_scalar(index):
            self._set_single_row(values, index, columns)
            return

        data = pd.DataFrame(values, columns=columns, index=index)
        self.world.update(data)

    def _set_single_row(self, values, index, columns):
        val = pd.Series(values, index=columns)
        for col in columns:
            self._set_column(col, index, val[col])

    def _set_column(self, column, index, values):
        self.world._dict[column] = _set_series_values(
            self.world._dict[column], index, values)

    def __delitem__(self, key):
        """Remove components or delete entities."""
        if not isinstance(key, tuple):
            self.world.remove_entities(key)
            return
        index = key[0]
        cols = key[1]
        if not isinstance(cols, list):
            cols = [cols]
        self.world.take(index, *cols)


class World:
    """
    The World stores and manages the state and events of the simulation.

    It is defined as consisting of a certain set of component types.
    """

    def __init__(self):
        """Create a world."""
        self._dict: dict = {}
        self._archs = ArchetypeStore()
        self.maxind = 0
        self._index = None

    @property
    def index(self):
        """All entity ids."""
        if self._index is None:
            self._index = pd.RangeIndex(0, self.maxind)
            self._idxmx = self.maxind
        return self._index

    def __getitem__(self, key):
        """Get data for components."""
        if isinstance(key, list):
            return self._query_components(key)

        if isinstance(key, Exclude):
            idx = self.index.difference(self._get(key.component).index)
            return pd.DataFrame({}, index=idx)

        if key not in self._dict:
            self._initialize(key)
        if isinstance(key, Component):
            return self._dict[key]

        raise ComponentError("Not a valid Component:", key)

    def _query_components(self, key):
        """Inner-join query across the components in `key` (a list of
        Component/Exclude), by walking only the archetypes that match the
        filter and slicing each component's already (archetype, eid)-sorted
        storage via `ArchetypeStore.range_lookup`, instead of aligning full
        per-component frames.
        """
        # Virtual components (@gets) aren't in the archetype system; fall back
        # to the concat approach so their getters are called correctly.
        includes = [k for k in key if not isinstance(k, Exclude)]
        if not includes:
            return EntityView(pd.Index([]), {})
        for comp in includes:
            if comp not in self._dict:
                self._initialize(comp)
        relevant_archetypes = self._archs.choose_archetypes(key)
        if len(relevant_archetypes) == 0:
            return EntityView(pd.Index([]), {comp: [] for comp in includes})
        index_parts = []
        slices = {comp: [] for comp in includes}
        for arch in relevant_archetypes:
            arch_int = int(arch)
            ref_comp = includes[0]
            start_ref, stop_ref = self._archs.range_lookup(ref_comp, arch_int)
            index_parts.append(self._dict[ref_comp].index.to_numpy()[start_ref:stop_ref])
            for comp in includes:
                start, stop = self._archs.range_lookup(comp, arch_int)
                slices[comp].append((self._dict[comp].values, start, stop))
        combined_index = pd.Index(np.concatenate(index_parts))
        return EntityView(combined_index, slices)

    def _initialize(self, key):
        if isinstance(key, Component):
            self._dict[key] = key.init_series()

    @lazy
    def loc(self):
        """Loc indexer, like pandas.DataFrame.loc."""
        return LocIndexer(self)

    def _initialize_state(self, components: Iterable):
        for comp in components:
            self._dict[comp] = comp.init_series()

    def set_state(self, state: Dict[Component, pd.DataFrame]):
        """
        Set the state of the world (entities, components) to the provided value.

        Arguments:
            state: of the form:
                {<component>: <series>}
                where component is a Component and series is a Series of
                component values, with entity ids as the index
        """
        self._initialize_state(list(self._dict.keys()))
        for component in state:
            _validate_component(component)
        # Rebuild archetype store from scratch before adding any data so that
        # component insertion has complete bitmask information from the start.
        self._archs = ArchetypeStore()
        all_eids = sorted({eid for data in state.values() for eid in data.index})
        if all_eids:
            self._archs.add_entities(all_eids)
        for component, data in state.items():
            if len(data.index):
                self._archs.add_component(list(data.index), component)
        # All archetypes are now final, so each component's per-archetype
        # block sizes/order are already known -- bucket-sort straight into
        # them instead of keying and sorting the whole thing at once.
        for component, data in state.items():
            if len(data.index):
                self._dict[component] = self._build_by_ranges(component, data)

    def add_entities(self, component_values):
        """
        Add entities to the world.

        Arguments:
            component_values is a dict of the form {<component>: values}
                the index is ignored
                a dataframe or list of dicts also works
        """
        component_values = pd.DataFrame(component_values)
        num_entities = len(component_values)
        indices = np.arange(self.maxind, self.maxind + num_entities)
        frames = _component_series(component_values, indices)
        comps = list(frames.keys())
        # Fast path: when all components are fresh (no prior data), every component
        # storage has the same index (= indices), so we compute the sort order once.
        all_fresh = len(indices) > 0 and all(
            comp not in self._dict or self._dict[comp].empty
            for comp in comps
        )
        # Snapshot must be taken before the archetype-store mutations below,
        # so it reflects the boundaries as they stood prior to this batch.
        snapshot = None if all_fresh else self._archs.range_snapshot(self._dict.keys())
        if len(indices):
            self._archs.add_entities(indices)
            for comp in component_values.columns:
                self._archs.add_component(indices, comp)
        if all_fresh:
            # A single add_entities() call gives every new row the same
            # columns (component_values.columns, uniformly), so this whole
            # batch shares one archetype bitmask; combined with `indices`
            # already being np.arange(...)-ascending, the batch is already
            # in final (archetype, eid) order -- no sort needed.
            for comp in comps:
                _validate_component(comp)
                self._dict[comp] = _new_series(comp, indices, np.asarray(frames[comp].values))
        else:
            dest_archs = self._lookup_masks(indices)
            for comp, series in frames.items():
                self._add_component(comp, series, indices, dest_archs, snapshot)
        self.maxind += num_entities
        self._index = None
        return indices.tolist()

    def _build_by_ranges(self, comp, series):
        """Build a fresh, fully-sorted Series for `comp` from `series` (whose
        index may be in arbitrary order), using the archetype store's
        already-final per-archetype block boundaries for `comp` as a bucket
        sort: partition eids by archetype, sort each (much smaller) bucket by
        eid alone, and concatenate the buckets in archetype order -- instead
        of keying and sorting the whole thing in one pass. Only valid when
        those boundaries are already final, e.g. set_state's bulk load.
        """
        eids = np.asarray(series.index, dtype=np.int64)
        values_arr = _as_value_array(series)
        masks = self._lookup_masks(eids)
        archs, starts, stops = self._archs._ranges.get(comp, _EMPTY_RANGES)
        order = np.empty(len(eids), dtype=np.int64)
        for arch, start, stop in zip(archs, starts, stops):
            bucket = np.flatnonzero(masks == arch)
            order[start:stop] = bucket[np.argsort(eids[bucket])]
        return _new_series(comp, eids[order], values_arr[order])

    def _add_component(self, comp, series, indices, dest_archs, ranges_snapshot, old_masks=None):
        """Write (indices, series) into comp's storage at the position(s)
        implied by `dest_archs` (each row's final archetype bitmask), using
        `ranges_snapshot` (an `ArchetypeStore.range_snapshot` taken before
        this batch's archetype mutations) to locate them.

        `old_masks`, if given, enables keep-last overwrite handling for rows
        that already had `comp` before this call (their old row is dropped
        so the new value wins). Omit it when `indices` are guaranteed not to
        already exist in comp's storage, e.g. brand-new entities.
        """
        _validate_component(comp)
        eids = np.asarray(indices, dtype=np.int64)
        values_arr = _as_value_array(series)
        existing = self._dict.get(comp)
        remove_positions = self._find_overwritten_positions(
            comp, existing, eids, old_masks, ranges_snapshot)
        self._dict[comp] = self._splice_component(
            comp, existing, eids, values_arr, dest_archs, ranges_snapshot,
            remove_positions=remove_positions)

    def _find_overwritten_positions(self, comp, existing, eids, old_masks, ranges_snapshot):
        """Positions within `existing` of any `eids` that already had `comp`
        before this call -- keep-last semantics drop that old row so the new
        value wins. Returns None when there's nothing to overwrite, or when
        overwrite detection doesn't apply (`old_masks` omitted, e.g. for
        brand-new entities that can't already have data for `comp`).
        """
        if old_masks is None or existing is None or existing.empty:
            return None
        pw2 = self._archs._component_powers.get(comp)
        if pw2 is None:
            return None
        already_had = (old_masks & pw2) != 0
        if not already_had.any():
            return None
        existing_eids = existing.index.to_numpy()
        return self._locate_by_old_archetype(
            existing_eids, eids[already_had], old_masks[already_had],
            ranges_snapshot.get(comp))

    def _splice_component(self, comp, existing, new_eids, new_values, dest_archs,
                           ranges_snapshot, remove_positions=None):
        """Insert (new_eids, new_values) into `existing`'s storage, keeping it
        ordered by (archetype, eid).

        `dest_archs` is each new row's final archetype bitmask; rather than
        recomputing a sort key for the whole existing array, each distinct
        archetype's rows are placed via a local search within that
        archetype's own sub-block (`_search_archetype_blocks`), using
        `ranges_snapshot` (boundaries as of before this batch's archetype
        mutations) to find it directly.

        `remove_positions`, if given, are positions within the *original*
        `existing` array (as it stood before this batch, matching
        `ranges_snapshot`) to drop as part of the same operation -- e.g. rows
        being relocated elsewhere, or rows being overwritten by `new_eids`.
        """
        new_eids = np.asarray(new_eids)
        new_values = np.asarray(new_values)
        dest_archs = np.asarray(dest_archs)

        order = (
            np.argsort(new_eids, kind='stable') if _is_uniform(dest_archs)
            else np.lexsort((new_eids, dest_archs))
        )
        new_eids, new_values, dest_archs = new_eids[order], new_values[order], dest_archs[order]

        if existing is None or existing.empty:
            # No pre-existing data for comp -- the sort above already put
            # this (small) batch in its final (archetype, eid) order.
            return _new_series(comp, new_eids, new_values)

        dtype = _widened_dtype(existing.dtype, new_values.dtype)
        if dtype != existing.dtype:
            existing = existing.astype(dtype)
        new_values = new_values.astype(dtype)

        existing_eids = existing.index.to_numpy()
        existing_values = existing.to_numpy()
        snapshot = (ranges_snapshot or {}).get(comp) or _EMPTY_RANGES

        positions = self._search_archetype_blocks(existing_eids, dest_archs, new_eids, snapshot)
        existing_eids, existing_values, positions = self._apply_removals(
            existing_eids, existing_values, positions, remove_positions)

        out_eids = np.insert(existing_eids, positions, new_eids)
        out_values = np.insert(existing_values, positions, new_values)
        return pd.Series(out_values, index=pd.Index(out_eids, dtype=np.int64), name=comp)

    def _apply_removals(self, existing_eids, existing_values, positions, remove_positions):
        """Drop `remove_positions` (positions within the *original* arrays)
        from `existing_eids`/`existing_values`, adjusting `positions`
        (insertion points computed against those original arrays) to match:
        every removed row sat strictly before the insertion point of
        whichever new row replaces/follows it, so each position only needs
        to shift down by however many removals preceded it.
        """
        if remove_positions is None or len(remove_positions) == 0:
            return existing_eids, existing_values, positions
        remove_positions = np.sort(remove_positions)
        keep = _keep_mask(len(existing_eids), remove_positions)
        positions = positions - np.searchsorted(remove_positions, positions, side='left')
        return existing_eids[keep], existing_values[keep], positions

    def _search_archetype_blocks(self, existing_eids, archs, eids, snapshot):
        """For each (eids[i], archs[i]), find eids[i]'s search position
        within the sub-block of `existing_eids` that `snapshot` records for
        archs[i] -- a local `searchsorted` within that one block, rather
        than one over the whole array. Positions are returned aligned to the
        given `eids` order; grouping by distinct archetype only costs
        anything extra when the batch isn't already uniform (the common case
        for bulk give()/take()).
        """
        if _is_uniform(archs):
            arch = int(archs[0]) if archs.size else 0
            start, stop = self._archs.range_lookup_in(snapshot, arch)
            return start + np.searchsorted(existing_eids[start:stop], eids)

        order = np.argsort(archs, kind='stable')
        unique_archs, bounds = _archetype_group_bounds(archs[order])
        positions = np.empty(len(eids), dtype=np.int64)
        for i, arch in enumerate(unique_archs):
            lo, hi = bounds[i], bounds[i + 1]
            idxs = order[lo:hi]
            start, stop = self._archs.range_lookup_in(snapshot, int(arch))
            positions[idxs] = start + np.searchsorted(existing_eids[start:stop], eids[idxs])
        return positions

    def _locate_by_old_archetype(self, existing_eids, eids, old_archs, snapshot):
        """Find each of `eids`'s position within `existing_eids`, using the
        per-archetype block boundaries recorded in `snapshot` (a
        pre-mutation `ArchetypeStore.range_snapshot`) for each eid's
        `old_archs` value. Assumes every eid is actually present in the
        block implied by its `old_archs` entry.
        """
        return self._search_archetype_blocks(existing_eids, old_archs, eids, snapshot)

    def _lookup_masks(self, eids):
        """Positional archetype-bitmask lookup for `eids` (all assumed present).

        Plain positional indexing via get_indexer is markedly cheaper than
        Series.reindex/.loc, which both run label-uniqueness bookkeeping we
        don't need here.
        """
        archs = self._archs.series
        positions = archs.index.get_indexer(eids)
        return archs.to_numpy()[positions]

    def _relocate_siblings(self, ids, old_masks, new_masks, skip, ranges_snapshot):
        """After `ids`'s archetype bitmasks changed (from `old_masks` to
        `new_masks`), physically move their rows within every OTHER
        component they already have, so that component's storage stays
        ordered by (archetype, eid) -- without touching any component's full
        array. `ranges_snapshot` must have been taken before this call's
        archetype mutations.
        """
        changed = old_masks != new_masks
        if not np.any(changed):
            return
        moved_ids = ids[changed]
        moved_old = old_masks[changed]
        moved_new = new_masks[changed]
        for comp, pw2 in self._archs._component_powers.items():
            if comp in skip:
                continue
            has_comp = (moved_old & pw2) != 0
            if not has_comp.any():
                continue
            series = self._dict.get(comp)
            if series is None or series.empty:
                continue
            self._dict[comp] = self._relocate_component(
                comp, series, moved_ids[has_comp], moved_old[has_comp],
                moved_new[has_comp], ranges_snapshot)

    def _relocate_component(self, comp, series, relocating, old_archs_for, new_archs_for,
                             ranges_snapshot):
        """Move `relocating`'s rows within `series` from their old archetype
        block to their new one."""
        existing_eids = series.index.to_numpy()
        remove_positions = self._locate_by_old_archetype(
            existing_eids, relocating, old_archs_for, ranges_snapshot.get(comp))
        values = series.to_numpy()[remove_positions]
        return self._splice_component(
            comp, series, relocating, values, new_archs_for, ranges_snapshot,
            remove_positions=remove_positions)

    def _begin_archetype_change(self, ids):
        """Snapshot everything a `give`/`take` call needs to know about the
        world's state *before* it starts mutating archetypes: each id's
        current bitmask, and every component's current range boundaries.
        """
        old_masks = self._lookup_masks(ids)
        snapshot = self._archs.range_snapshot(self._dict.keys())
        return old_masks, snapshot

    def _finish_archetype_change(self, ids, old_masks, snapshot, skip):
        """After a `give`/`take` call has finished mutating archetypes,
        relocate any sibling components (other than those in `skip`) whose
        rows need to move to match, and return each id's final bitmask.
        """
        new_masks = self._lookup_masks(ids)
        self._relocate_siblings(ids, old_masks, new_masks, skip, snapshot)
        return new_masks

    def give(self, ids, components):
        """Add given components to entities corresponding to ids."""
        ids = _coerce_ids(ids)
        given = set(components)
        old_masks, snapshot = self._begin_archetype_change(ids)
        for comp in components:
            self._archs.add_component(ids, comp)
        new_masks = self._finish_archetype_change(ids, old_masks, snapshot, given)
        frames = _component_series(components, indices=ids)
        for comp, series in frames.items():
            self._add_component(comp, series, ids, new_masks, snapshot, old_masks=old_masks)

    def take(self, ids, *components):
        """Remove given components from entities corresponding to ids."""
        ids = _coerce_ids(ids)
        taken_set = set(components)
        old_masks, snapshot = self._begin_archetype_change(ids)
        for component in components:
            self._remove_component_data(component, ids, old_masks, snapshot)
            if component in self._archs._component_powers:
                self._archs.remove_component(ids, component)
        self._finish_archetype_change(ids, old_masks, snapshot, taken_set)

    def _remove_component_data(self, component, ids, old_masks, snapshot):
        """Drop `component`'s rows for whichever of `ids` currently have it."""
        series = self._dict.get(component)
        pw2 = self._archs._component_powers.get(component)
        if series is None or series.empty or pw2 is None:
            return
        has_comp = (old_masks & pw2) != 0
        if not has_comp.any():
            return
        existing_eids = series.index.to_numpy()
        remove_positions = self._locate_by_old_archetype(
            existing_eids, ids[has_comp], old_masks[has_comp], snapshot.get(component))
        self._dict[component] = series[_keep_mask(len(existing_eids), remove_positions)]

    def remove_entities(self, ids):
        """
        Remove given entities from the world.

        Arguments:
            ids: the entity ids, corresponding to the indices of rows
                corresponding to these entities in the component dataframes.
        """
        ids = _coerce_ids(ids)
        for comp, series in list(self._dict.items()):
            mask = np.isin(series.index.to_numpy(), ids)
            if mask.any():
                self._dict[comp] = series[~mask]
        existing = ids[np.isin(ids, self._archs.series.index.to_numpy())]
        if len(existing):
            self._archs.remove_entities(existing)
        self._index = None

    def update(self, components: Dict[Component, pd.DataFrame]):
        """
        Update the world state with given component dataframes.

        Arguments:
            components is a dict of component: dataframe. the values
                in the dataframes represent the new values for those
                components in for the entities corresponding to their
                index.
        """
        for comp, frame in components.items():
            self.loc._set_column(comp, frame.index, frame.values)


_EMPTY_RANGES = (
    np.array([], dtype=np.int64),
    np.array([], dtype=np.int64),
    np.array([], dtype=np.int64),
)


def _widened_dtype(current_dtype, incoming_dtype):
    """The dtype that can safely hold both `current_dtype` and `incoming_dtype`."""
    if np.can_cast(incoming_dtype, current_dtype, casting='safe'):
        return current_dtype
    return np.result_type(current_dtype, incoming_dtype)


def _coerce_ids(ids):
    """Normalize a scalar-or-array entity id argument to a 1-D array."""
    return np.asarray([ids]) if np.isscalar(ids) else np.asarray(ids)


def _as_value_array(series_like):
    """Extract the raw values from a Series (or anything already array-like)."""
    return np.asarray(series_like.values if hasattr(series_like, 'values') else series_like)


def _is_uniform(archs):
    """Whether every entry in `archs` is the same value (vacuously true when empty)."""
    return archs.size == 0 or (archs == archs[0]).all()


def _archetype_group_bounds(sorted_archs):
    """Given archetype values already sorted ascending, return the distinct
    values and, for each one, the index its run starts at (plus a trailing
    bound), so callers can slice out each archetype's own contiguous span.
    """
    unique_archs, group_starts = np.unique(sorted_archs, return_index=True)
    bounds = list(group_starts) + [len(sorted_archs)]
    return unique_archs, bounds


def _keep_mask(length, drop_positions):
    """A boolean mask of `length` True values with `drop_positions` set False."""
    keep = np.ones(length, dtype=bool)
    if drop_positions is not None and len(drop_positions):
        keep[drop_positions] = False
    return keep


def _new_series(comp, eids, values_arr):
    """Build a fresh component Series, honoring/widening the component's declared dtype."""
    dtype = np.dtype(comp.dtype) if comp.dtype is not None else values_arr.dtype
    dtype = _widened_dtype(dtype, values_arr.dtype)
    return pd.Series(
        values_arr.astype(dtype), index=pd.Index(eids, dtype=np.int64), name=comp)


def _set_series_values(series, key, values):
    """Write values at eids `key` into `series`, widening its dtype if necessary."""
    if np.isscalar(key):
        eids = np.array([key], dtype=np.int64)
    else:
        eids = np.asarray(key, dtype=np.int64)
    values_arr = np.asarray(values.values if isinstance(values, pd.Series) else values)
    if values_arr.ndim == 0:
        values_arr = values_arr.reshape(1)
    if len(values_arr) == 0:
        return series
    dtype = _widened_dtype(series.dtype, values_arr.dtype)
    if dtype != series.dtype:
        series = series.astype(dtype)
    positions = series.index.get_indexer(eids)
    if (positions < 0).any():
        raise KeyError("entity ids not found in component storage")
    series.values[positions] = values_arr.astype(dtype)
    return series


def _component_series(components, indices):
    def _get_values(ser):
        try:
            return ser.values
        except AttributeError:
            return ser

    return {
        comp: pd.Series(_get_values(ser), index=indices)
        for comp, ser in components.items()
    }


def _validate_component(comp):
    if not isinstance(comp, Component):
        raise ComponentError(
            "component column names must be Component objects. "
            f"Recieved {comp} instead"
        )
