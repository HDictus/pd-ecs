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
            # TODO: support this
            raise ValueError(
                "Loc indexing without components is not supported")
        index, columns = key

        if not isinstance(columns, list):
            columns = [columns]
        _columns = []
        for col in columns:
            if isinstance(col, Component) and col.is_compound:
                for _, comp in col.subcomponents.items():
                    _columns.append(comp)
            else:
                _columns.append(col)
        columns = _columns
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
        self.world[column].loc[index] = values

    def __delitem__(self, key):
        """Remove components or delete entities."""
        if not isinstance(key, tuple):
            self.world.remove_entities(key)
            return
        index = key[0]
        cols = key[1]
        if not isinstance(cols, list):
            cols = [cols]
        components = [c if isinstance(c, Component)
                      else c[0] for c in cols]
        self.world.take(index, *components)


# TODO: a disproportionate amount of functionality is getting clustered in world
#    separate it out, e.g. into _impls ?
class World:
    """
    The World stores and manages the state and events of the simulation.

    It is defined as consisting of a certain set of component types.
    """

    def __init__(self):
        """Create a world."""
        self._dict: dict = {}
        self.maxind = 0

    @property
    def index(self):
        """All entity ids."""
        ids = []
        for _, df in self._dict.items():
            ids = df.index.union(ids)
        return ids.astype(np.int64)

    def __getitem__(self, key):
        """Get data for components."""
        # 3 possibilities
        # key is a simple component
        # key is a compound component
        # key is a tuple (subcomponent)
        # key is exclude
        # key is a list of some combination of these
        if isinstance(key, Component) and key.is_compound:
            return self._get(key)[key]
        return self._get(key)

    def _get(self, key):
        if isinstance(key, list):
            # TODO: here is the issue: converting to have multiple levels
            #   Solution: everything is multi-column, we just hide it for noncompound
            dfs = []
            dframe = False

            for k in key:
                res = self._get(k)
                if len(res.index) == 0:
                    return pd.DataFrame({}, index=[], columns=self._determine_columns(key))
                if isinstance(res, pd.DataFrame) and len(res.columns) > 0:
                    dframe = True
                dfs.append(res)

            if dframe:
                dfs = [_dataframify(df) for df in dfs]

            out = pd.concat(dfs, join='inner', axis=1, copy=False)
            return out

        if isinstance(key, Exclude):
            idx = self.index.difference(self._get(key.component).index)
            return pd.DataFrame(
                {},
                index=idx
            )
        if isinstance(key, tuple):
            return self._get(key[0])[key]
        if key not in self._dict:
            self._initialize(key)
        if isinstance(key, Component):
            if key in GETTERS:
                return GETTERS[key](self)
            return self._dict[key]

        raise ComponentError("Not a valid Component:", key)

    def _initialize(self, key):
        if isinstance(key, Component):
            # this can be moved into component
            if key.is_compound:
                self._dict[key] = pd.DataFrame({
                    combo: combo[-1].init_series()
                    for comp, combo in key.subcomponents.items()})
                return
            self._dict[key] = key.init_series()
            return
        if isinstance(key, tuple):
            self._initialize(key[0])

    @lazy
    def loc(self):
        """Loc indexer, like pandas.DataFrame.loc."""
        return LocIndexer(self)

    def _initialize_state(self, components: Iterable):
        # TODO: a lot of this logic can probably be moved to
        #   Component
        for component in components:
            if isinstance(component, tuple):
                series = component[-1].init_series()
                series.name = component
                self._dict[component] = series
                continue
            self._dict[component] = component.init_series()

    def _determine_columns(self, keys):
        cols = []
        deep = False
        alldeep = True
        for key in keys:
            if isinstance(key, Component) and key.is_compound:
                cols += self._determine_columns(key.subcomponents.values())
                deep = True
                continue
            if isinstance(key, Exclude):
                continue
            if deep:
                key = (key, '')
            else:
                alldeep = False
            cols.append(key)
        if deep and not alldeep:
            cols = [_deepify(col) for col in cols]
        return cols

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
        compound = {}
        for comp, series in frame.items():
            if isinstance(comp, tuple):
                compound[comp[0]] = compound.get(comp[0], {})
                compound[comp[0]][comp] = series
                continue
            self._add_component(comp, series, indices)
        self._add_compound(compound)

    def _add_component(self, comp, series, indices):
        _validate_component(comp)
        prev = self[comp]
        new_comp = pd.concat([
            prev,
            pd.Series(series, index=indices)])
        new_comp.name = prev.name
        # TODO: logic for dealing wiht compound components is quite scattered
        #  could there be a way to localize it, e.g. inside the Component class?new_comp[~new_comp.index.duplicated(keep="last")]
        new_comp = new_comp[~new_comp.index.duplicated(keep="last")].sort_index()
        self._dict[comp] = new_comp

    def _add_compound(self, compounds):
        for comp, subcomponents in compounds.items():
            prev = self._get(comp)
            df = pd.DataFrame(subcomponents)
            self._dict[comp] = pd.concat([
                prev.loc[prev.index.difference(df.index)],
                df
                ],
                axis=0
            ).sort_index()
            

    def give(self, ids, components):
        """Add given components to entities corresponding to ids."""
        if np.isscalar(ids):
            ids = [ids]
        frames = _component_series(components, indices=ids)
        self._add_components(frames, ids)

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
            if isinstance(comp, Component) and comp.is_compound:
                self.update({
                    (comp, col): value
                    for col, value in frame.items()}
                )
            else:
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


def _is_component(comp):
    if isinstance(comp, Component):
        return True
    return isinstance(comp, tuple) and all(
        isinstance(c, Component) for c in comp)


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

def _dataframify(df):
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame({(df.name, ''): df})

def _deepify(col):
    if isinstance(col, tuple):
        return col
    return (col, '')
