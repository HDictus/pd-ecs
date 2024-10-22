import numpy as np
import pandas as pd
from pd_ecs import Component, World, gets, sets

# TODO: i may want a way to benchmark each specific type of operation?
# TODO: right now it is inefficient to simply define the relationships between
#   data things, because when you need just a subset the whole thing will be calculated.
#   this means that the implementer of the abstraction needs to worry about caching, getting specific idx, etc.
#   If we're going to expect the implementer to optimize themselves we shouldn't have so many layers of abstraction

def test_component_get_based_on_arithmetic():
    vx = Component('vx', dtype=np.float32)
    vy = Component('vy', dtype=np.float32)
    vel = [vx, vy]
    direction = Component('dir')
    
    # TODO: the way we currently get/set makes this kinda hard

    @gets(direction)
    def direction_from_vel(world):
        # TODO: recreate this bug somewher
        velocity = world[[vx, vy]]
        dir = np.arctan2(velocity[vx], velocity[vy])
        return dir

    world = World()

    world.add_entities({
        vx: [0., 1.],
        vy: [1., 0.]
    })

    exp = direction_from_vel(world)
    exp.name = direction
    pd.testing.assert_series_equal(
        world[direction],
        exp
    )

    assert np.allclose(world[vel].values, [[0, 1], [1, 0]])

    @sets(direction)
    def vel_from_direction(world, ids, dir):
        magnitude = np.linalg.norm(world.loc[ids, vel].values, axis=-1)
        ux, uy = np.cos(dir), np.sin(dir)
        # TODO: here I've identified another bug - it doesn't play nice when setting with arrays
        #  create a seprate test, then simplify here
        world.loc[ids, vx] = ux * magnitude
        world.loc[ids, vy] = uy * magnitude

    world.loc[0, direction] = 0

    assert np.allclose(
        world[vel].values,
        [[1, 0], [1, 0]]
    )


def test_set_mutliple():
    a = Component('a')
    b = Component('b')
    c = Component('c')
    world = World()

    @gets(c)
    def c_from_a(world):
        return world[a] * 2
    
    # TODO: determine what the behavior should be if
    #  c and a are set at the same time.
    #  just let it depend on the order?
    @sets(c)
    def a_from_c(world, idx, val):
        world.loc[idx, a] = val / 2
    
    world.add_entities({a: [0, 1, 2], b: [1, 2, 3]})

    world.loc[[1, 2], [b, c]] = pd.DataFrame(
        {b: [0, 0], c: [4, 8]}, 
        index=[1,2]
    )
    
    pd.testing.assert_frame_equal(
        world[[a, b, c]],
        pd.DataFrame({
            a: [0, 2, 4],
            b: [1, 0, 0],
            c: [0, 4, 8]
        })
    )


def test_compound_abstraction():
    vx = Component('vx', dtype=np.float32)
    vy = Component('vy', dtype=np.float32)
    vel = Component('vel', x=vx, y=vy)
    direction = Component('dir')
    speed = Component('speed')

    @gets(vel)
    def vel_from_speeddir(world):
        dir_and_spd = world[[direction, speed]]
        ux = np.cos(dir_and_spd[direction])
        uy = np.sin(dir_and_spd[direction])
        return pd.DataFrame({
            vel.x: ux, vel.y: uy
        }) * dir_and_spd[speed].values

    @sets(vel)
    def speedir_from_vel(world, idx, val):
        velocity = pd.DataFrame(val, columns=[vel.X, vel.Y], index=idx)
        spd = np.linalg.norm(vel.values, axis=1)
        dir = np.atan2(velocity[vel.X], velocity[vel.Y])
        world.loc[idx, speed] = spd
        world.loc[idx, dir] = dir

    world = World()

    world.add_entities({
        direction: [0., np.pi / 2],
        speed: [10, 10]
    })

    exp = vel_from_speeddir(world)
    pd.testing.assert_frame_equal(
        world[vel],
        exp
    )

    world.loc[0, direction] = 0

    assert np.allclose(
        world[vel].values,
        [[1, 0], [1, 0]]
    )
# TODO: test using loc indexes and stuff to efficiently set/get only subset
# TODO: how should we behave if getter is defined after component is first used?
# TODO: test deletion, wtf do we do?
# TODO: test that it works with both dataframe and simple numbers, values
# TODO: what should we do if a setter is duplicated?
# TODO: add the ability to create a getter/setter for groups of components
#   this will be called whenever those components are all gotten/set together.
# TODO: test it works with give
# TODO: test working with compound components