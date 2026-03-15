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
        self.world._dict[column].loc[index] = values

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
        return self._get(key)

    def _get(self, key):
        if isinstance(key, list):
            dfs = []
            for k in key:
                res = self._get(k)
                if len(res.index) == 0:
                    return pd.DataFrame(
                        {}, index=[], columns=self._determine_columns(key))
                dfs.append(res)
            return pd.concat(dfs, join='inner', axis=1, copy=False)

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
        for component in components:
            self._dict[component] = component.init_series()

    def _determine_columns(self, keys):
        return [key for key in keys if not isinstance(key, Exclude)]

    def set_state(self, state: Dict[Component, pd.DataFrame]):
        """
        Set the state of the world (entities, components) to the provided value.

        Arguments:
            state: of the form:
                {<component>: <dataframe>}
                where component is a Component and dataframe is a dataframe of
                component values, with entity ids as the index
        """
        self._initialize_state(list(self._dict.keys()))
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
        indices = range(self.maxind, self.maxind + num_entities)
        frames = _component_series(component_values, indices)
        for comp, series in frames.items():
            self._add_component(comp, series, indices)
        self.maxind += num_entities
        self._index = None
        return list(indices)

    def _add_component(self, comp, series, indices):
        _validate_component(comp)
        prev = self[comp]
        new_comp = pd.concat([
            prev,
            pd.Series(series, index=indices)])
        new_comp.name = prev.name
        new_comp = new_comp[~new_comp.index.duplicated(keep="last")].sort_index()
        self._dict[comp] = new_comp

    def give(self, ids, components):
        """Add given components to entities corresponding to ids."""
        if np.isscalar(ids):
            ids = [ids]
        frames = _component_series(components, indices=ids)
        for comp, series in frames.items():
            self._add_component(comp, series, ids)

    def take(self, ids, *components):
        """Remove given components from entities corresponding to ids."""
        for component in components:
            self._dict[component].drop(ids, inplace=True)

    def remove_entities(self, ids):
        """
        Remove given entities from the world.

        Arguments:
            ids: the entity ids, corresponding to the indices of rows
                corresponding to these entities in the component dataframes.
        """
        for _, data in self._dict.items():
            ids_in = np.intersect1d(ids, data.index)
            data.drop(ids_in, inplace=True)
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
