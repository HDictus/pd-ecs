sPandas-based ecs
================


an entity-component system built on pandas, which shouldd allow very efficient processing.
Scientific programmers, already familiar with pandas could easily write quick models and games



How it should work
==================

Each component corresponds to a dataframe.
The dataframe contains one row for each entity that has this component.
The index of each row is the id of that entity.
Each column represents a specific piece of data in that component.

Alongside this, there are systems.
Following the design of Conrcord, these are classes.
Each method on the class represents an event.
The processors have filters consisting of components.
Entities which have all the components listed and none of the components blacklisted, are made accessible to the system in the form of a dataframe.
This dataframe contains only the whitelisted components.

Modifying this dataframe modifies the appropriate component dataframes.

Alternatively, we can just give the system access to all the components and a filter for each.
This seems more complex in usage, but I don't know how feasible the preferred design is.


ooh! we can do one better: we use add_entities(componentname=dict(varname=anarray))


(we could make it that Component's init method makes a version with the arguments as fields, but a __subclass__ can't be intiialized?


Declaring components
====================

a component basically represents a group of column headers
it could just be a list of strings...
But the tricky thing is that you need to be able to ask the world for the appropriate component.
And it shouldn't just be a tuple, because two components may have the same fields but represent different things

We can just declare a Component instance containing the fields, maybe even tuples of (field, type).
Then we index the world with the component instance.



Developing
==========

I need to strucutre this in a way that I can self-review.
I should start with the simplest thing: component initialization and world initialization.

Make, self-review.

Next, creating entities
self-review

Next, adding systems
self-review

Next, adding components to entities
Next, removing components from entities
review
Next, processing events



filters can wait, we can proceed with the filtered ids?

Next, filtering systems: what is the best way?


filters
=======

We can make filters work well with pandas by having every event return the modified components
Component filters don't need to be coupled to systems.
We can make a dict and pass it to world.
Then the events return the modified world state.
What about events called within other events?
Ideally we would want the worldstate used to be updated.
Can we make that happen? without wierd surprises?

Whenever any event is called, the dict of dataframes it returns is applied to the world using its index.

(systems could in principle just be event callbacks, but lets stick to convention for now)

The risk we may run into is that someone will create a variable corresponding to a filter.
When they call an event, that value will not be updated.
So, we should instead use a function get_filter(filtername), which suggests that 

maybe they can be a multiindexed dataframe. The modified form can be returned at the end of an event.
Wait, since only a select system should directly change any given component... couldn't we handle updating the world that way?


NDdata, e.g. maps?
==================

How should we handle multidimensional world data in an ECS? What if, for example. we wanted to model forest fires...
each row of the space would be an entity, and each column represent an x position within that entity
but actually each space is its own entity...
volumetric data has a disadvantage here anyway - all components must be stored for all...

instead, each patch has a position, just like anything else, then any relevant components.
rendersystem can render patches by indexing an image with them... is that at all efficient?
meh, that's for the user to decide I guess.
