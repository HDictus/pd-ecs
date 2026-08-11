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

from ._archetype_manager import ArchetypeManager
from ._entity_view import EntityView
from ._filter_ops import Exclude
from ._state import WorldState
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


class World:
    """
    The World stores and manages the state and events of the simulation.

    It is defined as consisting of a certain set of component types.
    """

    def __init__(self):
        """Create a world."""
        # TODO: separate object for storing state
        #  tricky- how to separate inteface from impl?
        # in fact, most of world' implementatin will
        # need to move to it, making world a thin wrapper
        # that probably composes other things (e.g. eventmanager)
        # nice idea: state uses observer pattern
        # making custom state impls easy.
        self._state = WorldState()
        self.maxind = 0
        self.archetypes = ArchetypeManager()

    @lazy
    def loc(self):
        """Loc indexer, like pandas.DataFrame.loc."""
        return LocIndexer(self)

    def add_entities(self, entities_df: dict):
        """Add entities to the world.

        entities_df: dataframe or dict containing their components
        """
        entities_df = _coerce_entity_dataframe(entities_df)
        initmax = self.maxind
        self.maxind += len(entities_df)
        entities_df.set_index(pd.RangeIndex(initmax, self.maxind), inplace=True)
        archetype = self.archetypes.add_entities(entities_df)
        if archetype not in self._state:
            self._state[archetype] = entities_df
        else:
            self._state[archetype] = (
                pd.concat([self._state[archetype], entities_df], axis=0)
            )
        return entities_df.index.values

    def __getitem__(self, filt):
        """Retrieve entities (as an EntityView) matching filter"""
        was_scalar = False
        if isinstance(filt, Component):
            filt = [filt]
            was_scalar = True
        if not isinstance(filt, list):
            raise ComponentError(f"Not a valid component: {filt}")
        slices, indices = self._compile_arrays(filt)
        if len(indices) == 0:
            return pd.DataFrame(slices)
        eview = EntityView(pd.Index(np.concatenate(indices)), slices)
        if was_scalar:
            return eview[filt[0]]
        return eview

    def _compile_arrays(self, filt):
        slices = {comp: [] for comp in filt}
        indices = []
        for at in self._ats_in_filt(filt):
            at_data = self._state[at]
            # TODO: simplify enti
            indices.append(at_data.index.values)
            for comp, arrs in slices.items():
                if isinstance(comp, Exclude):
                    continue
                arrs.append((at_data[comp].values, 0, len(at_data)))
        return slices, indices

    def _ats_in_filt(self, filt):
        return [at for at in self._state if _at_in_filt(at, filt)]

    def give(self, entities, components):
        """Add components to entities.
        Arguments:
            entities: ids of entities
            components: dataframe or similar of component values
        """
        # TODO: consider sorting by archetype allowing slicing?
        components = _coerce_entity_dataframe(components, entities)
        transitions = self.archetypes.give(entities, components.columns)
        self._state.give(transitions, components)

    def take(self, entities, components):
        """Remove components from entities.
        Arguments:
            entities: ids of entities
            components: iterable of components to remove
        """
        if isinstance(components, Component):
            components = [components]
        transitions = self.archetypes.take(entities, components)
        self._state.take(transitions)


def _at_in_filt(archetype, filt):
    # TODO: filtering should probably also have its own module
    # bitmask will probably be best
    for comp in filt:
        if isinstance(comp, Exclude):
            if comp.component in archetype:
                return False
        elif comp not in archetype:
            return False
    return True

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


def _coerce_entity_dataframe(entity_df, eids=None):
    df = pd.DataFrame(entity_df, index=eids)
    for col in df:
        _validate_component(col)
    return df


def _validate_component(comp):
    if not isinstance(comp, Component):
        raise ComponentError(
            "component column names must be Component objects. "
            f"Recieved {comp} instead"
        )
