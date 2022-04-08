import pandas as pd
from .component import Component
from .filter import Filter


class System:

    filters = {}

    def __init__(self, world):
        world.add_system(self)
        self.world = world
        for key, fields in self.filters.items():
            setattr(self, key, Filter(*fields, world=world))
