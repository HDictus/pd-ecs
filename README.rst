Pandas-based ecs
================


an entity-component system built on pandas, which shouldd allow very efficient processing.
Scientific programmers, already familiar with pandas could easily write quick models and games



Usage
=====

Components define how data is stored.
(dtypes are optional, and actually handling them is stil TODO for now lol)
```
X=Component('x', dtype=np.float32)
Y=Component('y', dtype=np.float32)
# components with sub-components. Stored together for efficiency
POSITION = Component('position', x=X, y=Y)
VELOCITY = Component('velocity', x=X, y=Y)
TEAM = Component('team', dtype='category')
```

Initialize a world and add entities

```
world = pd_ecs.World()
world.add_entities({
    POSITION.x: [1, 2, 3, 4],
    POSITION.y: [3, 4, 5, 6],
    VELOCITY.x: [0, 0, 0, 1],
    VELOCITY.y: [1, 1, 1, 0],
    TEAM: [1, 2, 3, 4],
})
```

Filter entities by attached components
```
entities = world[[POSITION, VELOCITY]]
```
Or use property-based filtering (still TODO)
```
entities = world[{
   TEAM: [1, 2], # single values or lists
   POSITION.x: (0, 1), # tuple represents min, max
   POSITION.y: None # only checks for presence of component
})
```

Change the state of the entities
```
entities[POSITION] += entities[VELOCITY]
```
Or if you used attr-friendly component names
```
entities.position += entities.velocity
```

You can retrieve the selected components, but this is slow and should only be used for debugging or saving
```
print(entities.df())
```
Or their full state, including unselected components (inserts NaNs)
```
print(entities.df(all=True))
```

At present we provide neither support nor constraint on how the processes of the simulation are implemented.


To do 
=====
 - find a way to be type-hint friendly
 - allow short attrname for components alongside more informative name
 - make configurable background optimizations, such as extending dataframes in advance
 - toggleable big fat dataframe with nans implementation - less memory efficient
 - support for parallel and distributed computing
 - GPU acceleration support
