"""System class"""
from .filter import Filter


# pylint: disable=too-few-public-methods
class System:
    """
    Base class for systems.

    allows one to define events as operations on a subset of components.
    When a system is defined with a method, e.g.

    ```
    class Physics(System):

        filters = dict(
             moveable = {position, velocity})

        def update(self, dt, moveable):
            do_stuff_with(self.moveable)
            return
    ```

    self.moveable will refer to all entities with both position and velocity
    components.
    When Physics is initialized:
    ```
    world = World()
    physics = Physics(world)
    ```
    and the update event is called
    ```
    world.events.update(0.001)
    ```
    `physics.update` will be called with 0.001

    If multiple systems may use the same events they will be called in the
    order in which they were initialized.
    A system can listen for as many events as you like, though in general
    smaller systems are cleaner code-wise.
    """
    filters: dict = {}

    def __init__(self, world):
        """world: a World instance"""
        world.add_system(self)
        self.world = world
        self._filters = {
            filt: Filter(*comps, world=world)
            for filt, comps in self.filters.items()}

    def __getattr__(self, key):
        """
        A System contains various filters, represented as dataframes
        """
        try:
            filt = self._filters[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return filt
