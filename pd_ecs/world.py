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
        num_entities = _number_of_entities(component_values)
        indices = range(self.maxind, self.maxind + num_entities)
        frames = _component_dataframes(component_values, indices)
        self._add_components(frames, indices)
        self.maxind += num_entities
        return list(indices)

    def _add_components(self, frames, indices):
        for comp, frame in frames.items():
            self._dict[comp] = pd.concat([
                self[comp],
                frame.assign(id=indices).set_index('id')])

    def give(self, ids, components):
        """add given components to entities corresponding to ids"""
        frames = _component_dataframes(components, indices=ids)
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


def _number_of_entities(components):
    nentities = None
    for comp, data in components.items():
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                if nentities is None:
                    nentities = len(v)
                else:
                    if len(v) != nentities:
                        raise ComponentError(
                            f"could not interperet number of entities for components."
                            f"length of {k}, {v} is not equal to {nentities}")
    return nentities or 1


def _component_dataframes(components, indices):
    for component, data in components.items():
        for key in data:
            if key not in component.fields:
                print(component.fields)
                raise ComponentError(
                    f"field {key} does not belong to {component}")

    frames = {component: pd.DataFrame(value, index=indices)
              for component, value in components.items()}
    return frames
