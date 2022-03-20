import pandas as pd


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
        _validate_components(component_values)
        frames = {component: pd.DataFrame(value)
                  for component, value in component_values.items()}
        num_entities = list(frames.values())[0].shape[0]
        assert all(frame.shape[0] == num_entities
                   for frame in frames.values())
        indices = range(self.maxind, self.maxind + num_entities)
        for comp, frame in frames.items():
            self._dict[comp] = pd.concat([
                self[comp],
                frame.assign(id=indices).set_index('id')])
        self.maxind += num_entities

# class EventManager:

#     def __init__(self, world):
#         self.world = world

#     def __getattr__(self, key):
#         def eventfunction(*args, **kwargs):
#             for system in self.world.systems.values():
#                 if hasattr(system, key):
#                     getattr(system, key)(*args, **kwargs)
#         return eventfunction

def _validate_components(components):
    for component, data in components.items():
        for key in data:
            if key not in component.fields:
                raise KeyError(f"field {key} does not belong to {component}")
