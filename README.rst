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



 I just realized that I don't necessarily need ExtensionArrays that propagate writes.
 I just need ExtensionArrays that delay copying until necessary - copy-on-write probably.
 replacing the array dtype on write is therefore fine, and it can simplify things a lot.

 NOTE: this is a completely different strategy, and so should go on a different branch
 In this case, I don't need an EntityCollecton, I just need to use archetypes
 and I need the World to return dataframes that have a FrameView extensiondtype.

 The steps for this are as follows:
  1. use archetypes to store the components, and concatenate them just as now
  2. implement a FrameView extensionarray that can extend a view across multiple dataframes
  3. update World to use the FrameView instead of concatenating

 ANOTHER NOTE:
 LOL and LMAO: copy-on-write is the default for pandas concat in pandas 3.0 and later,
 in earlier, it seems you can use copy=False?

Ok so first, try with copy=False, see if that addresses the performance issues.
Then you can try the other implementation if that doesn't work.
One advantage I think of my own approach is that you can do copy-on-write on a per-series basis.


So the options before me are:

1. use copy=False 
2. use Archetypes and a read-only FrameView
3. combine write-propagating FrameViews with a custom DataFrame subclass that intercepts operations which would mess with the datatype.

I had best do these options in order, assessing the performance impact of the first, and then taking a look at the others.

MAKE THIS AN ISSUE INSTEAD

Can I have both implementations?
Separate components and archetypes?


It may be best to try and avoid copying out data altogether.
Instead, you request the ids of the entities in a given filter,
you directly get and set the columns you want from the World and minimize the creation of other arrays.
This does make the efficiency of the programming very dependent on the implementation of the user.


This is a lot simpler than returning an EntityCollection for example.
A key advantage of an EntityCollection is that it can maintain its mapping onto the data and therefore read/write more efficiently.

SEPARATE ISSUE
I need to carefully consider whether compound components are worth the hassle.
A lot of complexity is added to the code in order to handle them.
In the end, if you have position and speed and want to add them, it isn't much worse to do:
```
POSITION=[X, Y]
VELOCITY=[VX, VY]

data[POSITION] += data[VELOCITY].values
```
than
```
POSITION = Component('position', x=X, y=Y)
VELOCITY = Component('velocity', x=X, y=Y)

data[POSITION] += data[VELOCITY]
```

From Claude: consider,

 Then the real question is whether the vectorized bulk-operation use case is   
  even compatible with implementation-neutral storage. Consider what an
  archetype store would need to do to serve world[A, B, C] as a contiguous array
   suitable for vectorized ops: it would have to gather data from multiple
  archetype tables (entities with [A,B,C,D], entities with [A,B,C], etc.) and
  concatenate them — you're back to a concat, just inside the implementation.

  So the concat isn't an artifact of your current design, it's load-bearing for
  the vectorized API. Any implementation that stores data non-contiguously by
  component combination will pay it.

  The implementations that could avoid it:

  - Column-per-component DataFrame (your current direction): inner join is the
  bottleneck, but there's no fundamental concat if you move to index
  intersection + direct access
  - Single wide DataFrame: column selection is O(1), no join needed — sparse
  entities are the only complication, handled by nullable columns rather than
  missing rows
  - Per-archetype packed arrays: fast for homogeneous entity populations, but
  requires a gather/concat at query time for mixed archetypes

  Given the vectorization constraint, the single wide DataFrame is actually the
  most honest fit for what you want — queries are just column selections, no
  join at all. The sparsity problem is real but manageable: nullable dtypes or
  explicit presence columns handle it without materializing new DataFrames.

  The archetype model is a better fit for iteration-based ECS (which is why EnTT
   uses it). For a vectorized Python ECS, you're essentially reinventing a
  column store, and pandas is already one.



Another interesting point:

    The Canonical Pattern: Table Iterators
                                                                                
  In archetype-based ECS (flecs, Bevy, Unity DOTS, hecs), a query doesn't return
   a single virtual array. It returns an iterator that yields one archetype
  chunk at a time, where each chunk is a real contiguous array.

  flecs (C):
  ecs_iter_t it = ecs_query_iter(world, q);
  while (ecs_query_next(&it)) {
      Position *p = ecs_field(&it, Position, 0);
      Velocity *v = ecs_field(&it, Velocity, 1);
      // p and v are raw C pointers into the archetype's contiguous storage
      for (int i = 0; i < it.count; i++) {
          p[i].x += v[i].x;
      }
      // then the while loop advances to the next archetype table
  }

  Each call to ecs_query_next() sets it.count and updates the field pointers to
  point into the next table. No virtual concatenation — the user processes each
  contiguous chunk, the iterator advances the "cursor" between chunks.

  Bevy (Rust) does the same thing with traits. QueryIter is essentially a
  FlatMap over archetypes:

  // Conceptually what Bevy's query iteration compiles to:
  for archetype in matching_archetypes {
      for entity in archetype.entities() {
          // access components from archetype's contiguous storage
      }
  }

  The user-facing for (pos, vel) in query.iter() looks like one sequence but is
  chunked internally. Rust's zero-cost iterator composition means there's no
  overhead.

  Why Not Virtual Concatenation?

  Several practical reasons:

  1. SIMD/autovectorisation — compilers can vectorise a loop over a contiguous
  pointer + length. A virtual array with per-element indirection breaks this.
  Chunked iteration preserves it within each chunk.
  2. No random access needed — ECS workloads are almost always full sequential
  scans ("process all entities with X and Y"). Random access by logical index
  across archetypes is rarely needed and not part of the query API.
  3. Simpler implementation — an iterator struct with a (ptr, count,
  table_index) cursor is trivial. A virtual array with O(log n) segment lookup
  is more complex for no benefit in the common case.
  4. Parallel scheduling — chunks map naturally to job granularity. Unity's
  IJobChunk and Bevy's par_iter() split work at archetype/chunk boundaries.

  When They Do Need Index-Like Access

  The one place virtual concatenation-like logic appears is in entity lookup by
  ID, not component data access. The entity index maps an entity ID →
  (archetype, row) in O(1) via a flat array or sparse set. But this is a
  metadata lookup, not data access.

  EnTT takes a different approach entirely — sparse sets rather than archetypes
  — so component arrays for a single type are always one contiguous array. The
  "multiple arrays" problem is avoided by the storage model.

  Takeaway for Your Context

  For pd_ecs, the idiomatic path is probably to follow flecs: don't try to
  present a unified virtual array to the user. Instead expose a chunk iterator
  that yields a real DataFrame (or numpy array slice) per archetype. This keeps
  each chunk zero-copy and makes the boundary between archetypes explicit rather
   than hidden behind an abstraction that leaks anyway (e.g. when a slice
  crosses a boundary).

This is the core tension in ECS design — archetypes solve the "process all    
  entities with components X,Y" case beautifully, but entity-to-entity
  references (relationships) reintroduce pointer chasing. There are a few       
  canonical solutions:                                                        
                                                                                
  1. Direct Entity Lookup (Simple, Cache-Unfriendly)                            
                                                                                
  Every ECS maintains an entity index: a flat array mapping entity ID →
  (archetype, row). This makes get(farm_id, Position) O(1) regardless of which
  archetype the farm is in.

  // flecs
  ecs_iter_t it = ecs_query_iter(world, farmer_query);
  while (ecs_query_next(&it)) {
      FarmRef *farm_refs = ecs_field(&it, FarmRef, 1);
      for (int i = 0; i < it.count; i++) {
          // O(1) lookup but random memory access — cache miss per farmer
          const Position *farm_pos = ecs_get(world, farm_refs[i].id, Position);
      }
  }

  This is simple and correct but defeats cache locality — each ecs_get likely
  causes a cache miss since farms are scattered across archetypes and rows.

  2. Relationships as First-Class Features (flecs)

  flecs has a built-in (Relation, Target) pair system that lets the query engine
   handle this. You can query:

  // "Give me all farmers together with the Position of their farm target"
  ecs_query_t *q = ecs_query(world, {
      .terms = {
          { .id = FarmerComp },
          { .id = ecs_pair(Owns, EcsWildcard) },   // farmer owns a farm
          { .id = Position, .src.flags = EcsUp }   // Position from the owned
  entity
      }
  });

  The query system resolves the relationship internally and can group results to
   improve locality. This is flecs' answer to the problem — model the reference
  as a relationship, let the engine optimise traversal.

  3. Two-Phase / Gather-Scatter for Cache Efficiency

  If you care about throughput, the pattern used in data-oriented game engines
  (Unity DOTS, etc.) is:

  Phase 1 — Gather: iterate farmers, collect (farmer_index, farm_id) pairs into
  a temp buffer.

  Phase 2 — Sort: sort by farm_id (or by (archetype, row) if you have access to
  it). Now accesses to farm data will be roughly sequential within each
  archetype's storage.

  Phase 3 — Process: walk the sorted buffer, fetching farm data in order.

  [farmer_0 → farm_17]          sort by farm location
  [farmer_1 → farm_3 ]    →→→   [farmer_1 → farm_3 ] ← archetype A, row 2
  [farmer_2 → farm_91]          [farmer_0 → farm_17] ← archetype A, row 8
  [farmer_3 → farm_3 ]          [farmer_3 → farm_3 ] ← archetype A, row 2
  (same!)
                                 [farmer_2 → farm_91] ← archetype B, row 4

  This turns random access into sequential access at the cost of a sort. Whether
   the tradeoff is worth it depends on your entity counts.


 4. Mutations: Command Buffers

  The harvesting case is harder because you cannot structurally modify entities
  (add/remove components) during iteration — it would invalidate the archetype
  tables you're iterating over.

  The standard solution is a command buffer (called Commands in Bevy,
  ecs_cmd_buf in flecs, EntityCommandBuffer in Unity DOTS):

  During iteration:
    farmer sees farm is ready to harvest
    → push "RemoveCrops(farm_id)" to command buffer
    → push "AddHarvested(farm_id)" to command buffer
    → push "AddToInventory(farmer_id, wheat, 10)" to command buffer

  After iteration completes:
    → flush command buffer — structural changes applied in batch

  This keeps the iteration loop clean and all archetype modifications happen
  after the query completes. The cost is one frame of latency for the effects.

  If the change is not structural (just mutating a field value, not
  adding/removing components), you can write directly during iteration — most
  ECS systems allow this since it doesn't move entities between archetypes.

  5. Avoid the Problem: Denormalisation

  Sometimes the right answer is to copy the farm's position onto the farmer,
  updated each frame by a cheap "sync positions" system that runs before the
  harvest system. This is deliberately denormalised but keeps the harvest
  system's hot loop fully contiguous with no pointer chasing. Unity DOTS
  documentation explicitly recommends this pattern for performance-critical
  paths.

  
