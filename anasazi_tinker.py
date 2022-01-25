# TODO: incorporate water sources
# TODO: figure out how many plots and households could occupy the same space
# TODO: is there a role of proximity to previous neighbors/family?
# TODO: identify gameplay scenarios and decisions
# TODO: add historical settlements, whatever that means in this context
# TODO: add house positions in addition to farm plots
# TODO: better to do more minimalistic first
#    e.g. first just watch harvests over time
#    then fission and deth
#    then moving households
#    then add in house positions (settlements)
# though this is very much unfinished, I think I have what I need to
# put together the first parts of the ECS

from scipy.stats import norm, uniform
from anasazi_mapping import MapData, START_YEAR, END_YEAR
import numpy as np

MIN_AGE = 16
LIFESPAN = 30
MAX_AGE = MIN_AGE + LIFESPAN - 1

MIN_FOOD, MAX_FOOD = 2000, 24000
HARVEST_VARIANCE = 0.5


# TODO each of these things requires access to the world state
#   in many ways
#   however, it is good practice to split the write permissions between
#   different systems
def initialize_soil_quality(shape):
    return np.random.normal(size=shape, loc=1, scale=HARVEST_VARIANCE)


def initialize_households(map_shape, n=14):
    xp = np.random.choice(range(0, map_shape[0]), size=n)
    yp = np.random.choice(range(0, map_shape[1]), size=n)
    positions = np.array([xp, yp])
    food = np.random.uniform(MIN_FOOD, MAX_FOOD, size=n)
    age = np.random.uniform(MIN_AGE, MAX_AGE, size=n)
    return positions, food, age


def calculate_occupancy(map_shape, household_positions):
    occupancy = np.zeros(map_shape, dtype=np.int32)
    print(household_positions)
    occupancy[tuple(household_positions)] = 1
    return occupancy


def calculate_harvest(household_positions, mean_yields):
    mean_harvests = mean_yields[tuple(household_positions)]
    return np.random.normal(mean_harvests, HARVEST_VARIANCE)

def update_food_stores(households_food, harvest, households_need):
    return households_food - households_need + harvest

def estimate_harvest(households_food, previous_harvest, households_need):
    dissatisfied_households = households_food + previous_harvest < households_need
    return dissatisfied_households

# this sort of thing made easier by entity grouping
# ah fuck this one is tricky... should use proper methods
def move_households(moving_households, mean_yields, occupancy, household_positions, households_need):
    occupancy[tuple(household_positions[moving_households])] = 0
    moving_positions = household_positions[moving_households].transpose()
    moving_needs = households_need[moving_households]
    for posn, need in zip(moving_positions, moving_needs):
        potential_farms = np.array(np.nonzero(occupancy == 0 && mean_yields >= need))
        distances = np.linalg.norm(potential_farms - posn, axis=0)
        minimum = np.argmin(distances)
    return

def check_starvation(households_food):
    return

def check_old_age(households_age):
    return

def household_fissions(households_age, households_food):
    return

def update_water_sources(water_data):
    return

def increase_age(householdS_age):
    return

def load_map(textfile):
    return



def run():
    year = START_YEAR
    mapdata = MapData('/home/hugs/Desktop/app/models/Sample Models/Social Science/Unverified/data/')
    soil_quality = initialize_soil_quality(mapdata.shape)
    household_positions, households_food, households_age, households_need = initialize_households(mapdata.shape)
    occupancy = calculate_occupancy(mapdata.shape, household_positions)
    mean_yields = mapdata.mean_yield_in_year(year) * soil_quality
    move_households(household_positions, mean_yields, occupancy)

    # TODO: can separate into yearly actions and shorter-term actions
    def step(dt):
        nonlocal year
        nonlocal mapdata
        nonlocal soil_quality
        nonlocal household_positions
        nonlocal households_food
        nonlocal households_need
        nonlocal households_age
        nonlocal mean_yields
        nonlocal occupancy
        year += 1
        mean_yields = mapdata.mean_yield_in_year(year) * soil_quality
        harvest = calculate_harvest(household_positions, mean_yields)
        households_food = update_food_stores(households_food, harvest, households_need)
        check_starvation(households_food)
        check_old_age(households_age)
        dissatisfied_households = estimate_harvest(households_food, harvest, households_need)
        household_positions, occupancy = move_households(dissatisfied_households, mean_yields, occupancy)
        # again, changes on a worldstate are more convenient here
        households_positions, households_food, households_need, households_age, new_households = household_fissions(households_age, households_food)
        move_households(new_households, mean_yields, occupancy)
        increase_age(households_age)
        return

    while True:
        step(0)
        print(year, end="\r")

if __name__ == "__main__":
    from minicli import command
    command(run)
