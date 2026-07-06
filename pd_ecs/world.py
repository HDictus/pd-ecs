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
from .data_abstraction import GETTERS, SETTERS
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
        if column in SETTERS:
            SETTERS[column](self.world, index, values)
            return
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

    def _sort(self, comp):
        """Re-sort a component's storage by archetype bitmask then entity id."""
        series = self._dict[comp]
        if series.empty:
            return
        eids = series.index.to_numpy()
        masks = self._archs.series.reindex(series.index, fill_value=0)
        order = np.lexsort((eids, masks.to_numpy()))
        self._dict[comp] = series.iloc[order]

    def __getitem__(self, key):
        """Get data for components."""
        if isinstance(key, list):
            # Virtual components (@gets) aren't in the archetype system; fall back
            # to the concat approach so their getters are called correctly.
            if any(
                (isinstance(k, Exclude) and k.component in GETTERS) or
                (not isinstance(k, Exclude) and k in GETTERS)
                for k in key
            ):
                dfs = []
                for k in key:
                    res = self._get(k)
                    if len(res.index) == 0:
                        return pd.DataFrame(
                            {}, index=[], columns=self._determine_columns(key))
                    dfs.append(res)
                return pd.concat(dfs, join='inner', axis=1, copy=False)

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
            if key in GETTERS:
                return GETTERS[key](self)
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

    def _determine_columns(self, keys):
        return [key for key in keys if not isinstance(key, Exclude)]

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
        # _sort has complete bitmask information from the start.
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
        self._dict[comp] = _append_series(existing, comp, eids, values_arr)
        self._sort(comp)

    def give(self, ids, components):
        """Add given components to entities corresponding to ids."""
        ids = np.asarray([ids]) if np.isscalar(ids) else np.asarray(ids)
        new_eids = np.setdiff1d(ids, self._archs.series.index.to_numpy(), assume_unique=True)
        if len(new_eids):
            self._archs.add_entities(new_eids)
        for comp in components:
            self._archs.add_component(ids, comp)
        frames = _component_series(components, indices=ids)
        for comp, series in frames.items():
            self._add_component(comp, series, ids)

    def take(self, ids, *components):
        """Remove given components from entities corresponding to ids."""
        ids = np.asarray([ids]) if np.isscalar(ids) else np.asarray(ids)
        taken_set = set(components)
        for component in components:
            if component in self._dict:
                series = self._dict[component]
                mask = np.isin(series.index.to_numpy(), ids)
                if mask.any():
                    self._dict[component] = series[~mask]
            if component in self._archs._component_powers:
                self._archs.remove_component(ids, component)
        # Re-sort only components that the affected entities still possess.
        # Read their current bitmasks from the archetype store (already updated
        # above) instead of scanning every component storage with np.isin.
        if len(ids) and self._archs._component_powers:
            pos = self._archs.series.index.get_indexer(ids)
            valid = pos[pos >= 0]
            if len(valid):
                affected_bits = int(np.bitwise_or.reduce(self._archs.series.values[valid]))
                for comp, pw2 in self._archs._component_powers.items():
                    if comp not in taken_set and comp in self._dict and (affected_bits & int(pw2)):
                        self._sort(comp)

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


def _append_series(existing, comp, eids, values_arr):
    """Concatenate new (eids, values) onto an existing component Series, if any."""
    if existing is None or existing.empty:
        return _new_series(comp, eids, values_arr)
    dtype = existing.dtype
    if not np.can_cast(values_arr.dtype, dtype, casting='safe'):
        dtype = np.result_type(dtype, values_arr.dtype)
        existing = existing.astype(dtype)
    new_series = pd.Series(
        values_arr.astype(dtype), index=pd.Index(eids, dtype=np.int64), name=comp)
    return pd.concat([existing, new_series])


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
