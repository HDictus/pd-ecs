import pandas as pd
from .component import Component


class System:

    def __init__(self, world):
        world.add_system(self)
        self.world = world
        for attr in dir(self):
            attrvalue = getattr(self, attr)
            if is_list_of_components(attrvalue):
                print(attrvalue)
                self._components = {attr: attrvalue}
                setattr(self, attr, self._create_func(attrvalue))
        return

    def _create_func(self, value):
        def attr_func():
            ids = self.filtered_ids(value)
            return ids
        return attr_func

    def filtered_ids(self, components):
        # TODO: note, not yet working for multiple object types
        return set(self.world.get_component(components[0]).index)


def is_list_of_components(someattr):
    return (isinstance(someattr, list)
            and all([isinstance(item, Component)
                     for item in someattr]))


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
