Pandas-based ecs
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

Next, filtering systems




