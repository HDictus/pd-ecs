"""
The World stores and manages the state and events of the simulation.

It is defined as consisting of a certain set of component types.
Systems are added to the world.
World.events.<event_name> calls that event for all systems in the world
"""

import warnings
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
        self.world = world

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            raise ValueError("Loc indexing without components is not supported.")
        return self.world[key[1]].loc[key[0]]

    def __setitem__(self, key, values):
        if not isinstance(key, tuple):
            # TODO: support this
            raise ValueError("Loc indexing without components is not supported")
        index, columns = key
        if not isinstance(columns, list):
            columns = [columns]

        if pd.api.types.is_scalar(index):
            self._set_single_row(values, index, columns)
            return

        data = pd.DataFrame(values, columns=columns, index=index)
        for column, series in data.items():
            self._set_column(column, series.index, series.values)

    def _set_single_row(self, values, index, columns):
        val = pd.Series(values, index=columns)
        for col in columns:
            self._set_column(col, index, val[col])

    def _set_column(self, column, index, values):
        if column in SETTERS:
            SETTERS[column](self.world, index, values)
            return
        self.world[column].loc[index] = values

    def __delitem__(self, key):
        if not isinstance(key, tuple):
            self.world.remove_entities(key)
            return
        index = key[0]
        cols = key[1]
        if not isinstance(cols, list):
            cols = [cols]
        components = [c if isinstance(c, Component) else c[0] for c in cols]
        self.world.take(index, *components)


# TODO: a disproportionate amount of functionality is getting clustered in world
#    separate it out, e.g. into _impls ?
class World:
    """
    The World stores and manages the state and events of the simulation.

    It is defined as consisting of a certain set of component types.
    """

    def __init__(self):
        """
        components: component types the world consists of
        """
        self._dict: dict = {}
        self.maxind = 0

    @property
    def index(self):
        ids = []
        for _, df in self._dict.items():
            ids = df.index.union(ids)
        return ids

    def __getitem__(self, key):
        item = self._get_item(key)
        if isinstance(key, Component) and key.is_compound:
            item = item[key]
        return item

    def _get_item(self, key):
        if isinstance(key, list):
            return self._get_multiple(key)
        _validate_component(key)
        # TODO: not really happy with this
        if isinstance(key, Component) and key.is_compound:
            return self._get_multiple(list(key.subcomponents.values()))
        series = self._get(key)
        series.name = key
        return series

    def _get(self, key):
        if key not in self._dict:
            if key in GETTERS:
                return GETTERS[key](self)
            self._initialize_state((key,))
        return self._dict[key]

    def _get_multiple(self, key):
        exclude = [k for k in key if isinstance(k, Exclude)]
        to_concat = [self._get_item(k) for k in key if k not in exclude]

        for exclude_component in exclude:
            excluded = to_concat[0].index.intersection(
                self[exclude_component.component].index
            )
            to_concat[0] = to_concat[0].drop(excluded, axis=0)

        out = pd.concat(to_concat, join="inner", axis=1)
        _ensure_columns_index_level_consistent(out)
        return out

    @lazy
    def loc(self):
        """Loc indexer, like pandas.DataFrame.loc."""
        return LocIndexer(self)

    def _initialize_state(self, components: Iterable):
        for component in components:
            if isinstance(component, tuple):
                series = component[-1].init_series()
                series.name = component
                self._dict[component] = series
                continue
            self._dict[component] = component.init_series()

    def set_state(self, state: Dict[Component, pd.DataFrame]):
        """
        Set the state of the world (entities, components) to the provided value

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
            component_values is a dict of dicts  of the form
                {<component>: {<field>: values}}
                the columns of the dataframe are the fields of the components
                the index is ignored
                a dataframe or list of dicts also works
        """
        component_values = pd.DataFrame(component_values)
        num_entities = len(component_values)
        indices = range(self.maxind, self.maxind + num_entities)
        frames = _component_series(component_values, indices)
        self._add_components(frames, indices)
        self.maxind += num_entities
        return list(indices)

    def _add_components(self, frame, indices):
        for comp, series in frame.items():
            self._add_component(comp, series, indices)

    def _add_component(self, comp, series, indices):
        _validate_component(comp)
        if comp not in self._dict:
            self._initialize_state((comp,))
        new_comp = pd.concat([self._dict[comp], pd.Series(series, index=indices)])
        new_comp.name = self._dict[comp].name

        self._dict[comp] = new_comp[~new_comp.index.duplicated(keep="last")]

    def give(self, ids, components):
        """Add given components to entities corresponding to ids."""
        frames = _component_series(components, indices=ids)
        self._add_components(frames, ids)

    def take(self, ids, *components):
        """Remove given components from entities corresponding to ids."""
        for component in components:
            if component.is_compound:
                for k, comp in component.subcomponents.items():
                    self._dict[comp].drop(ids, inplace=True)
            else:
                self._dict[component].drop(ids, inplace=True)

    def remove_entities(self, ids):
        """
        Removes given entities from the world.

        Arguments:
            ids: the entity ids, corresponding to the indices of rows
                corresponding to these entities in the component dataframes.
        """
        for comp, data in self._dict.items():
            ids_in = np.intersect1d(ids, data.index)
            data.drop(ids_in, inplace=True)

    def update(self, components: Dict[Component, pd.DataFrame]):
        """
        Update the world state with given component dataframes

        Arguments:
            components is a dict of component: dataframe. the values in the
                dataframes represent the new values for those components in
                for the entities corresponding to their index.
        """
        for comp, frame in components.items():
            if isinstance(comp, Component) and comp.is_compound:
                self.update({(comp, col): value for col, value in frame.items()})
            else:
                self[comp].loc[frame.index] = frame


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


def _is_component(comp):
    if isinstance(comp, Component):
        return True
    return isinstance(comp, tuple) and all(isinstance(c, Component) for c in comp)


def _validate_component(comp):
    if not _is_component(comp):
        raise ComponentError(
            "component column names must be Component objects. "
            f"Recieved {comp} instead"
        )


def _ensure_columns_index_level_consistent(df):
    column_index_depth = 1
    for k in df.columns:
        if isinstance(k, tuple):
            column_index_depth = max(column_index_depth, len(k))
    if column_index_depth == 1:
        return df
    if column_index_depth > 1:
        cols = []
        for col in df.columns:
            if not isinstance(col, tuple):
                col = (col,)
            col += ("",) * (column_index_depth - len(col))
            cols.append(col)
        df.columns = pd.MultiIndex.from_tuples(cols)
    return df
