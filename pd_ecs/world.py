import pandas as pd
import numpy as np
from lazy import lazy
from .exceptions import ComponentError


# TODO: docstrings and stuff, DUH!
class World:

    def __init__(self, *components):
        self._dict = {}
        for component in components:
            self._dict[component] = component.init_dataframe()
        self.systems = {}
        self.maxind = 0
        return

    def __getitem__(self, key):
        return self._dict[key]

    def add_system(self, system):
        self.systems[system.__class__] = system

    def add_entities(self, component_values):
        num_entities, frames = _component_dataframes(component_values)

        indices = range(self.maxind, self.maxind + num_entities)
        self._add_components(frames, indices)
        self.maxind += num_entities

    def _add_components(self, frames, indices):
        for comp, frame in frames.items():
            self._dict[comp] = pd.concat([
                self[comp],
                frame.assign(id=indices).set_index('id')])

    def give(self, ids, components):
        """add given components to entities corresponding to ids"""
        num_entities, frames = _component_dataframes(
            components, num_entities=len(ids))
        self._add_components(frames, ids)
        return

    def take(self, ids, *components):
        """remove given components from entities corresponding to ids"""
        for component in components:
            self._dict[component].drop(ids, inplace=True)
        return

    def remove_entities(self, ids):
        for comp, data in self._dict.items():
            ids_in = np.intersect1d(ids, data.index)
            data.drop(ids_in, inplace=True)

    @lazy
    def events(self):
        """calls any events, callign system's event functions"""
        return EventManager(self)


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


def _component_dataframes(components, num_entities=None):
    for component, data in components.items():
        for key in data:
            if key not in component.fields:
                print(component.fields)
                raise ComponentError(
                    f"field {key} does not belong to {component}")

    frames = {component: pd.DataFrame(value)
              for component, value in components.items()}
    if num_entities is None:
        num_entities = list(frames.values())[0].shape[0]
    for component, frame in frames.items():
        if frame.shape[0] != num_entities:
            raise ComponentError(
                f"number of values for component {component}, {frame.shape[0]} "
                "differs from the expected number, {num_entities}")
    return num_entities, frames
