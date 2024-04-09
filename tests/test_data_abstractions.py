import numpy as np
import pandas as pd
from pd_ecs import Component, World, gets, sets


def test_component_get_based_on_arithmetic():
    vel = Component('vx', 'vy')

    direction = Component('dir')
    
    # TODO: the way we currently get/set makes this kinda hard

    @gets(direction)
    def direction_from_vel(world):
        # TODO: recreate this bug somewher
        velocity = world[vel]
        df = pd.DataFrame({'dir': np.arctan2(velocity['vy'], velocity['vx'])})
        return df

    world = World()

    world.add_entities({
        vel: {
            'vx': [0., 1.],
            'vy': [1., 0.]
        }
    })

    exp = direction_from_vel(world)

    pd.testing.assert_frame_equal(
        world[direction],
        exp
    )
    assert np.allclose(world[vel].values, [[0, 1], [1, 0]])

    @sets(direction)
    def vel_from_direction(world, ids, dir):
        magnitude = np.linalg.norm(world.loc[ids, vel].values, axis=-1)
        ux, uy = np.cos(dir.dir), np.sin(dir.dir)
        # TODO: here I've identified another bug - it doesn't play nice when setting with arrays
        #  create a seprate test, then simplify here
        world.loc[ids, vel.vx] = ux * magnitude
        world.loc[ids, vel.vy] = uy * magnitude
        print(world[vel].values.dtype)
    
    world.loc[0, direction] = pd.DataFrame({'dir': [0]})

    assert np.allclose(
        world[vel].values,
        [[1, 0], [1, 0]]
    )

# TODO: test using sub-component
# TODO: test using loc indexes and stuff to efficiently set/get only subset
# TODO: how should we behave if getter is defined after component is first used?
# TODO: test deletion, wtf do we do?
# TODO: test that it works with both dataframe and simple numbers, values
# TODO: what should we do if a setter is duplicated?