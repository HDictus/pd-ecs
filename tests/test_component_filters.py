from pd_ecs import System, World, Component
import pandas as pd

# TODO: eventually we shouldd be able to include filters on the data contained in them

a_component = Component('a variable')
other_component = Component('somevar')
thrd_component = Component('multiple', 'vars')

# TODO: unnecessary, we should just index the world with the components themselves

has_all = dict(a_component={'a variable': [1, 2, 3, 4, 5, 6]},
               other_component={'somevar': [1, 2, 3, 4, 5,6]},
               another_component={'multiple': [2, 3, 4, 5, 6, 7],
                                  'vars': [3, 4, 5, 6, 7, 8]})
has_a = dict(a_component={'a_variable': [4, 4, 4]})
a_and_thrd=dict(a_component={'a variable': [1, 1, 1]},
                thrd_component={'multiple': 3,
                                'vars': 4})
lacks_thrd = dict(a_component=[3, 3, 1], other_component=[3, 3, 1])
other_no_thrd = dict(other_component=[1, 2, 3])

allentities = [has_all, has_a, a_and_thrd, lacks_thrd, other_no_thrd]


class ASystem(System):

    filters = dict(has_a=[a_component],
                   has_all=[a_component, other_component, thrd_component],
                   a_and_thrd=[a_component, thrd_component],
                   lacks_thrd=[~thrd_component],
                   other_no_thrd=[~thrd_component, other_component])

    def update(self):
        return


world = World(a_component, other_component, thrd_component)
system = ASystem(world)

for thing in allentities:
    world.add_entities(**thing)


# TODO: what about filter with nothing in it?
def test_filters_one_component():
    pd.testing.assert_frame_equal(
        system.has_a[a_component],
        pd.concat([
            pd.DataFrame(entities['a_component'])
            for entities in allentities if 'a_component' in entities]))


def test_filters_multiple():
    hasall = system.has_all
    pd.testing.assert_frame_equal(
        hasall[a_component],
        pd.DataFrame(has_all['a_component']))
    pd.testing.assert_frame_equal(
        hasall[other_component],
        pd.DataFrame(has_all['other_component']))

    pd.testing.assert_frame_equal(
        hasall[thrd_component],
        pd.DataFrame(has_all['thrd_component']))


def test_filters_including_other():
    pd.testing.assert_frame_equal(
        system.a_and_thrd[a_component],
        pd.concat(
            [pd.DataFrame(has_all['a_component']),
             pd.DataFrame(a_and_thrd['a_component'])]))

    pd.testing.assert_frame_equal(
        system.a_and_thrd[thrd_component],
        pd.concat(
            [pd.DataFrame(has_all['thrd_component']),
             pd.DataFrame(a_and_thrd['thrd_component'])]))



# need to update with world.add_entities
# world.add_components,
# world.remove_components
# world.delete_entities


# def test_filters_change_with_world():
#     world.add_entities(has_all)
#     pd.testing.assert_frame_equal(
#         system.has_all[a_component](world),
#         pd.concat(
#             [pd.DataFrame(has_all['a_component']),
#              pd.DataFrame(has_all['a_component'])]))
