import pandas as pd


class World:

    def __init__(self, **components):
        self.components = AttrDict(
            {name: component.init_dataframe()
             for name, component in components.items()})
        self._component_to_name = {comp: name for name, comp
                                   in components.items()}
        self.systems = {}
        self.events = EventManager(self)
        return

    # TODO: lol, does not currently make unique ids
    def add_entities(self, **components):
        for name, comp in components.items():
            frame = pd.DataFrame(comp)
            self.components[name] = pd.concat([self.components[name], frame])
        pass

    def update(self, dt):
        self.events.update(dt)
        return

    def add_system(self, system):
        self.systems[system.__class__] = system

    def get_component(self, component):
        # TODO: this gets kind of confusing: what is a component, what isn't?
        #   classical solution would be to subclass Component.... but thats dumb.
        #   the actual instance is a row in a dataframe
        return self.components[self._component_to_name[component]]


class EventManager:

    def __init__(self, world):
        self.world = world

    def __getattr__(self, key):
        def eventfunction(*args, **kwargs):
            for system in self.world.systems.values():
                if hasattr(system, key):
                    getattr(system, key)(*args, **kwargs)
        return eventfunction


class AttrDict(dict):

    def __getattr__(self, key):
        return self[key]

    def __index__(self, key):
        return self._dict[key]
