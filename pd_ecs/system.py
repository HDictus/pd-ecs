import pandas as pd
from .component import Component


class System:

    def __init__(self, world):
        world.add_system(self)
        self.world = world
        if hasattr(self, filters):
            self._setup_filters(filters)
        return

    def _setup_filters(self, filters):
        for name, components in self.filters():
            assert not hasattr(self, name)
            setattr(self, name, Filter(components, self.world))


    def _create_func(self, value):
        def attr_func():
            ids = self.filtered_ids(value)
            return ids
        return attr_func

    def filtered_ids(self, components):
        # TODO: note, not yet working for multiple object types
        return self.world.get_component(components[0]).index


def is_list_of_components(someattr):
    return (isinstance(someattr, list)
            and all([isinstance(item, Component)
                     for item in someattr]))


class FakeFrame:

    def __init__(self, frame):
        self.frame = frame
        return

    def set_filter_indices(self, index):
        self._indices = index

    @property
    def data(self):
        self.data = self.frame.loc[self._indices]

    def __getattribute__(self, key):
        return getattr(self.data, key)

    def __getitem__(self, key):
        return self.data[key]

    # def __setitem__(


class Filter:

    def __init__(self, components, world):
        self.comps = {}
        for component in components:
            self._comps[component] = FakeFrame(world[component])

        self.ids = self.filter_entities(**{component: world[component]}
                                        for component in components)

    def filter_entities(self, **components):
        idslist = []
        for comp in components:
            if self._comps[component]:
                idslist.append(component.index.values))
        if len(idslist) == 0:
            return []
        ids = idslist[0]
        for otherids in idslist:
            ids = ids[np.isin(ids, otherids)]
        return ids

    def entities_added(self, **components):
        self.ids += self.filter_entities(**components)
        return


# # TODO: this will be the most complex part, I'm not sure how to do it well.
# class ComponentFilter:

#     # we somehow need component names
#     def __init__(self, components, world):
#         self.components = components
#         self.world = world

#     @property
#     def _frames_and_ids(self):
#         frames = {}
#         ids = set()
#         for component in self.components:
#             frame = self.world.get_component(component)
#             frames[component] = frame
#             ids |= set(frame.index)
#         return frames, list(ids)


#     @property
#     def _whole_frame(self):
#         frames, ids = self._frames_and_ids
#         return pd.concat([f.loc[ids] for _, f in frames.items()])

#     def __getitem__(self, key):
#         return self._whole_frame[key]

#     def __setitem__(self, columns, values):
#         # TODO: this would need to work with index as well
#         frames, ids = self._frames_and_ids
#         for comp, frame in frames.items():
#             print(frame)
#             print(ids, columns)

#                 print(frame.loc[ids, columns])
#                 frame.loc[ids, columns] = values
