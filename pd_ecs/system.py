import pandas as pd
from .component import Component


class System:

    def __init__(self, world):
        world.add_system(self)
        self.world = world
