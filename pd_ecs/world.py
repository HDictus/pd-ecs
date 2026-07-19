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

        if isinstance(key, Exclude):
            idx = self.index.difference(self._get(key.component).index)
            return pd.DataFrame({}, index=idx)

        if key not in self._dict:
            self._initialize(key)
        if isinstance(key, Component):
            return self._dict[key]

        raise ComponentError("Not a valid Component:", key)

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
        for component, data in state.items():
            self._add_component(component, data, data.index)

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
        if len(indices):
            self._archs.add_entities(indices)
            for comp in component_values.columns:
                self._archs.add_component(indices, comp)
        frames = _component_series(component_values, indices)
        comps = list(frames.keys())
        # Fast path: when all components are fresh (no prior data), every component
        # storage has the same index (= indices), so we compute the sort order once.
        all_fresh = len(indices) > 0 and all(
            comp not in self._dict or self._dict[comp].empty
            for comp in comps
        )
        if all_fresh:
            for comp in comps:
                _validate_component(comp)
                self._dict[comp] = _new_series(comp, indices, np.asarray(frames[comp].values))
            if comps:
                ref = self._dict[comps[0]]
                eids = ref.index.to_numpy()
                masks = self._archs.series.reindex(ref.index, fill_value=0)
                order = np.lexsort((eids, masks.to_numpy()))
                for comp in comps:
                    self._dict[comp] = self._dict[comp].iloc[order]
        else:
            for comp, series in frames.items():
                self._add_component(comp, series, indices)
        self.maxind += num_entities
        self._index = None
        return indices.tolist()

    def _add_component(self, comp, series, indices):
        _validate_component(comp)
        eids = np.asarray(indices, dtype=np.int64)
        values_arr = np.asarray(series.values if hasattr(series, 'values') else series)
        existing = self._dict.get(comp)
        if existing is not None and not existing.empty:
            # Remove existing entries for these eids so new values win (keep-last semantics).
            keep = ~np.isin(existing.index.to_numpy(), eids)
            if not keep.all():
                existing = existing[keep]
        self._dict[comp] = self._splice_component(comp, existing, eids, values_arr)

    def _splice_component(self, comp, existing, new_eids, new_values):
        """Insert (new_eids, new_values) into `existing`'s storage, keeping it
        ordered by (archetype, eid), by shifting the surrounding entries into
        place instead of re-sorting entries that are already correctly placed.
        """
        new_eids = np.asarray(new_eids)
        new_values = np.asarray(new_values)

        # The incoming batch is (usually) small, so sorting just this slice by
        # its final (archetype, eid) key -- rather than the whole array -- is
        # what makes inserting it into the existing, already-sorted storage
        # cheap. This sort must happen regardless of whether `existing` has
        # any data, since a batch spanning multiple archetypes isn't
        # necessarily eid-ordered.
        new_keys = _archetype_sort_key(self._lookup_masks(new_eids), new_eids)
        order = np.argsort(new_keys, kind='stable')
        new_eids = new_eids[order]
        new_values = new_values[order]
        new_keys = new_keys[order]

        if existing is None or existing.empty:
            return _new_series(comp, new_eids, new_values)

        dtype = existing.dtype
        if not np.can_cast(new_values.dtype, dtype, casting='safe'):
            dtype = np.result_type(dtype, new_values.dtype)
            existing = existing.astype(dtype)
        new_values = new_values.astype(dtype)

        existing_eids = existing.index.to_numpy()
        existing_keys = _archetype_sort_key(self._lookup_masks(existing_eids), existing_eids)

        positions = np.searchsorted(existing_keys, new_keys, side='right')
        out_eids = np.insert(existing_eids, positions, new_eids)
        out_values = np.insert(existing.to_numpy(), positions, new_values)
        return pd.Series(out_values, index=pd.Index(out_eids, dtype=np.int64), name=comp)

    def _lookup_masks(self, eids):
        """Positional archetype-bitmask lookup for `eids` (all assumed present).

        Plain positional indexing via get_indexer is markedly cheaper than
        Series.reindex/.loc, which both run label-uniqueness bookkeeping we
        don't need here.
        """
        archs = self._archs.series
        positions = archs.index.get_indexer(eids)
        return archs.to_numpy()[positions]

    def _relocate_siblings(self, ids, old_masks, skip):
        """After `ids`'s archetype bitmasks changed, physically move their rows
        within every OTHER component they already have, so that component's
        storage stays ordered by (archetype, eid) -- without a full re-sort.
        """
        new_masks = self._lookup_masks(ids)
        changed = old_masks != new_masks
        if not np.any(changed):
            return
        moved_ids = ids[changed]
        for comp in self._archs._component_powers:
            if comp in skip:
                continue
            series = self._dict.get(comp)
            if series is None or series.empty:
                continue
            series_eids = series.index.to_numpy()
            # Boolean masking (not get_indexer/.loc) avoids paying for an
            # index-uniqueness check on this component's storage.
            is_moving = np.isin(series_eids, moved_ids)
            if not is_moving.any():
                continue
            relocating = series_eids[is_moving]
            values = series.to_numpy()[is_moving]
            remaining = series[~is_moving]
            self._dict[comp] = self._splice_component(comp, remaining, relocating, values)

    def give(self, ids, components):
        """Add given components to entities corresponding to ids."""
        ids = np.asarray([ids]) if np.isscalar(ids) else np.asarray(ids)
        old_masks = self._lookup_masks(ids)
        given = set(components)
        for comp in components:
            self._archs.add_component(ids, comp)
        self._relocate_siblings(ids, old_masks, given)
        frames = _component_series(components, indices=ids)
        for comp, series in frames.items():
            self._add_component(comp, series, ids)

    def take(self, ids, *components):
        """Remove given components from entities corresponding to ids."""
        ids = np.asarray([ids]) if np.isscalar(ids) else np.asarray(ids)
        taken_set = set(components)
        old_masks = self._lookup_masks(ids)
        for component in components:
            if component in self._dict:
                series = self._dict[component]
                mask = np.isin(series.index.to_numpy(), ids)
                if mask.any():
                    self._dict[component] = series[~mask]
            if component in self._archs._component_powers:
                self._archs.remove_component(ids, component)
        self._relocate_siblings(ids, old_masks, taken_set)

    def remove_entities(self, ids):
        """
        Remove given entities from the world.

        Arguments:
            ids: the entity ids, corresponding to the indices of rows
                corresponding to these entities in the component dataframes.
        """
        ids = np.asarray([ids]) if np.isscalar(ids) else np.asarray(ids)
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


def _new_series(comp, eids, values_arr):
    """Build a fresh component Series, honoring/widening the component's declared dtype."""
    dtype = comp.dtype
    if dtype is None:
        dtype = values_arr.dtype
    else:
        dtype = np.dtype(dtype)
        if not np.can_cast(values_arr.dtype, dtype, casting='safe'):
            dtype = np.result_type(dtype, values_arr.dtype)
    return pd.Series(
        values_arr.astype(dtype), index=pd.Index(eids, dtype=np.int64), name=comp)


def _archetype_sort_key(archs, eids):
    """Combined ascending key matching (archetype bitmask, eid) order, so a
    plain 1-D searchsorted can find insertion points without re-sorting
    entries that are already correctly placed.

    Assumes eid < 2**32 (comfortably covers realistic entity counts), which
    leaves the high bits free for the (<=32-bit) archetype bitmask.
    """
    eids = np.asarray(eids, dtype=np.uint64)
    archs = np.asarray(archs, dtype=np.uint64)
    return (archs << np.uint64(32)) | eids


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
    dtype = series.dtype
    if not np.can_cast(values_arr.dtype, dtype, casting='safe'):
        dtype = np.result_type(dtype, values_arr.dtype)
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
