import pandas as pd
from lazy import lazy


class World:

    def __init__(self, max_entities):
        return

    def add_processor(self):
        return

    def add_component(self):
        return

    def new_entity(self):
        return


class Component:

    def __init__(self, data_types):
        return


POSITION = Component({"X": float, "Y": float})
VELOCITY = Component({"X": float, "Y": float})


class MotionProcessor:

    def __init__(self, world):
        return

    @property
    def moving_entities(self):
        return world.filter(POSITION, VELOCITY)

    def update(self, state, dt):
        state[POSITION] += state[VELOCITY] * dt


class CollisionProcessor:

    def __init__(self):