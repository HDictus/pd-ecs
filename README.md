pd-ecs: a pandas-based entitiy component system
===============================================

When doing scientific computing in python, you rely on batch-processing arrays and tables through libraries like numpy and pandas.
Entity-component systems similarly rely on vectorization for efficiency, so it seems like a natural match to combine scientific software with an ECS.
ECS are usually used for games, but I think they could be useful for simulations, especially agent-based modeling as well.
Indeed, within games, they are often used for the purposes of simulation.

ECS provide a lot of flexibility in the behavior of entities in a simulation, and would be very helpful when you expect to need to adjust the underlying theories and assumptions of your model as you develop it.

ECS also, in principle, lends itself well to parallelization and GPU acceleration.

Usage
-----

Entities are just identifiers (integers) for things in your simulation.

Components define data attached to entities:
```
X = Component('x (m)', dtype=np.float32)
Y = Component('y (m)', dtype=np.float32)
VX = Component('x velocity (m/s)', dtype=np.float32)
VY = Component('y velocity (m/s)', dtype=np.float32)
```

A world is where entities and components exist
```
world = World()
entity_ids = World.add_entities({
    X: [0, 1],
    Y: [0, 1],
    VX: [0, 2],
    VY: [2, 0]
})
# filter entities based on components
movers = world[[X, Y, VX, VY]]
# mutate filtered entities
movers[[X, Y]] += movers[[VX, VY]]

# remove components
world.take(entity_ids[0], VX, VY)
# add components
world.give(entity_ids[0], {VX: [1], VY: [1]})
# remove entities
world.remove_entities(enity_ids[0])
```
Examples
--------

examples include benchmarks to test performance of pd-ecs

https://github.com/HDictus/pd-ecs-anaszi-example , an implementation of the artificial anasazi simuation
https://github.com/HDictus/pd-ecs-dcrawl-example , an incomplete game prototype

state of project
----------------

This is a prototype, or rather a series of different prototypes.
I developed it mainly to learn more about data oriented design.
At present it likely isn't good for much.

The master branch has the fastest implementation.
It stores each component in its own series, but sorts them by archetype (component set)  so they can be easily filtered and sliced without losing cache-locality.

A different implementation can be found on the refactor-archetypes branch.
The approach taken there is more similar to Unity DOTS, storing each archetype in its own array.
The current implementation is slower than master, but may be better after further optimization.

Another option, which may be most performant in python, would be to revisit the original implementation in which components were grouped together based on the required filters, or according to user specification. For example, instead of x, y, vx, vy, w, h  components for an AA-BB collision system, you just have a 'physics' component with x, y, vx, vy, w, h subcomponents, and so any physics operations just work contiguously without any filtering. Operations between components, for example aligning sprites with physical positions, become more expensive however, and certain cases become tricky to navigate (such as an object with an x and y but no collisions), and may break cache-locality.


In the long run, I would like to improve the separation of concerns in the module so that multiple implementations can exist side-by-side and can be selected based on which performs better for your usecase. The current design hides the details of the data storage behind some wrappers, allowing such a swapping of implementations. However, this information hiding is anti-pattern to data-oriented design, and may introduce insurmountable performance barriers. In that case, we will need to pick an implementation, and adjust the design for its performance (which is generally the opposite of what you want to do for well-designed software, but performance is critical for simulations). I personally hope that data-oriented design and good design can be reconciled in this case.

In theory it should not be very difficult to parallelize your systems with pd-ecs, but I do not provide explicit support for it yet. In time, I would like to provide better support for this.

