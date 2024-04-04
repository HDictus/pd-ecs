import numpy as np
from pd_ecs import Component, World


def test_component_get_based_on_arithmetic():
    vx = Component("vx")
    vy = Component("vy")


    class Direction(Component):
        
        def get(self, world):
            return np.arctan2(world[vy], world[vx])

    direction = Direction('dir')
   
    world = World()
    # TODO: additional tests
    world.add_entities({
        vx: {'vel': [0, 1]},
        vy: {'vel': [1, 0]}
    })
    
    exp = np.arctan2(
    assert pd.testing.assert_frame_equal(
        world[direction],
        
    