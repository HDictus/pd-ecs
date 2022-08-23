"""
The World stores and manages the state and events of the simulation.

It is defined as consisting of a certain set of component types.
Systems are added to the world.
World.events.<event_name> calls that event for all systems in the world
"""
from collections.abc import Iterable
import pandas as pd
import numpy as np
from lazy import lazy
from .exceptions import ComponentError
from .component import Component
from .system import System
from typing import Dict


class World:
    """
    The World stores and manages the state and events of the simulation.

    It is defined as consisting of a certain set of component types.
    Systems are added to the world.
    World.events.<event_name> calls that event for all systems in the world
    """

    def __init__(self, *components: Component):
        """
        components: component types the world consists of
        """
        self._dict: dict = {}
        self._initialize_state(components)
        self.systems: dict = {}
        self.filters_by_component: dict = {
            comp: [] for comp in components}
        self.maxind = 0

    def __getitem__(self, key):
        return self._dict[key]

    def notify_filters_added(self, component, ids):
        """Inform the relevant filters ids have component now."""
        for filt in self.filters_by_component[component]:
            filt.add_components(component, ids)

    def notify_filters_removed(self, component, ids):
        """Inform the relevant filters ids no longer have component.e"""
        for filt in self.filters_by_component[component]:
            filt.remove_components(component, ids)

    def add_system(self, system: System):
        """Add an initialized system to the world.
        This is usually called in the system's init method - it is not
        necessary to manually do so again"""
        self.systems[system.__class__] = system

    def _initialize_state(self, components: Iterable):
        for component in components:
            self._dict[component] = component.init_dataframe()

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

    def add_entities(self, component_values: Dict[Component, pd.DataFrame]):
        """
        Add entities to the world.
        Arguments:
            component_values is a dict of dicts  of the form
                {<component>: {<field>: values}}
                the columns of the dataframe are the fields of the components
                the index is ignored
                a dataframe or list of dicts also works
        """
        num_entities = _number_of_entities(component_values)
        indices = range(self.maxind, self.maxind + num_entities)
        frames = _component_dataframes(component_values, indices)
        self._add_components(frames, indices)
        self.maxind += num_entities
        return list(indices)

    def _add_components(self, frames, indices):
        for comp, frame in frames.items():
            self._add_component(comp, frame, indices)
            self.notify_filters_added(comp, indices)

    def _add_component(self, comp, frame, indices):
        if comp not in self._dict:
            raise ComponentError(
                f"Component {comp} does not exist in this world")

        for key in frame:
            if key not in comp.fields:
                raise ComponentError(
                    f"field {key} does not belong to {comp}")

        self._dict[comp] = pd.concat([
            self[comp],
            frame.set_index(np.array(indices))])

    def give(self, ids, components):
        """Add given components to entities corresponding to ids."""
        frames = _component_dataframes(components, indices=ids)
        self._add_components(frames, ids)

    def take(self, ids, *components):
        """Remove given components from entities corresponding to ids."""
        for component in components:
            self._dict[component].drop(ids, inplace=True)
            self.notify_filters_removed(component, ids)

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
            self.notify_filters_removed(comp, ids)

    @lazy
    def events(self):
        """calls any events, callign system's event functions"""
        return EventManager(self)

    def update(self, components: Dict[Component, pd.DataFrame]):
        """
        Update the world state with given component dataframes

        Arguments:
            components is a dict of component: dataframe. the values in the
                dataframes represent the new values for those components in
                for the entities corresponding to their index.
        """
        for comp, frame in components.items():
            self[comp].loc[frame.index] = frame


# pylint: disable=too-few-public-methods
class EventManager:
    """passes event calls through to the world's systems"""

    def __init__(self, world):
        self.world = world

    def __getattr__(self, key):

        def eventfunction(*args, **kwargs):
            for system in self.world.systems.values():
                if hasattr(system, key):
                    getattr(system, key)(*args, **kwargs)

        setattr(self, key, eventfunction)
        return eventfunction


def _number_of_entities(components):
    """gets the number of entities suggested by component data"""
    nentities = None
    for _, data in components.items():
        for field, values in data.items():
            if isinstance(values, Iterable):
                if nentities is None:
                    nentities = len(values)
                else:
                    if len(values) != nentities:
                        raise ComponentError(
                            f"could not interperet number of entities for "
                            f"components. length of {field}, {values} is not "
                            f"equal to {nentities}")
    if nentities == 0:
        return 0
    return nentities or 1


def _component_dataframes(components, indices):
    frames = {component: pd.DataFrame(value, index=indices)
              for component, value in components.items()}
    return frames
