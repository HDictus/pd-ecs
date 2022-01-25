import numpy as np
import pandas as pd


# if x > 74 apparently 20 is yield 2, else yield 1

# lol idk what these acutally mean
MAP_VALUES = {0: ('General', 'Yield_2'),
              10: ('North', 'Yield_1'),
              15: ('North Dunes', 'Sand_dune'),
              20: ('Mid', 'see above'),
              25: ('Mid Dunes', 'Sand_dune'),
              30: ('Natural', 'No_Yield'),
              40: ('Uplands', 'Yield_3'),
              50: ('Kinbiko', 'Yield_1'),
              60: ('Empty', 'Empty')}


# wtf do they do with the dunes?
# also why do they reuse north for kibinko and natural for upland
# should probably read the paper lol
# TODO: read the first paper
APDSI_OFFSETS = {
    'General': -200,
    'North': 1100,
    'Mid': 2400,
    'Natural': 3700,
    'Upland': 3700,
    'Kibinko': 1100
}


MAP_HEIGHT = 120

START_YEAR = 800  # AD
END_YEAR = 1350  # AD

def get_shape(map_string, height=MAP_HEIGHT):
    # why -1?
    total_entries = len(map_string.split(" "))
    print(total_entries)
    return int(total_entries / height), height


def load_data(map_path):
    with open(map_path) as mapfile:
        return mapfile.read().strip()


def iterate_mapdata(map_string, height=MAP_HEIGHT):
    entries = map_string.split(" ")
    y = height - 1
    x = 0
    for entry in entries:
        yield x, y, int(entry)
        if y > 0:
            y -= 1
        else:
            x +=1
            y = height -1


def read_map(map_path):
    map_string = load_data(map_path)
    width, height = get_shape(map_string)
    array_dtype = max(len(v) for z, y in MAP_VALUES.values() for v in (z, y))
    print(width, height)
    zones = np.empty([width, height], dtype=f'S{array_dtype}')
    yields = np.empty([width, height], dtype=f'S{array_dtype}')
    for x, y, value in iterate_mapdata(map_string):
        zone, yld = MAP_VALUES[value]
        zones[x, y] = zone
        yields[x, y] = yld
    return zones, yields


def read_PDSI(pdsi_path):
    """
    Read the data for palmer-drought-severity-index data
    converts these to a mapping by year for each zone
    """
    pdsi_string = load_data(pdsi_path)
    values = list(pdsi_string.split(" "))
    data = []
    for year in range(START_YEAR, END_YEAR + 1):
        data.append(dict(year=year, **{
            zone: float(values[offset + year])
            for zone, offset in APDSI_OFFSETS.items()}))
    return pd.DataFrame(data).set_index('year')


# not clarified where this data comes from. once again, read the paper
# lol why is sand dune the highest?
YIELD_DATA = pd.DataFrame(
    {'min_apdsi': [3.0, 1.0, -1.0, -3.0, -np.inf],
     'Yield_1': [1153, 988, 821, 719, 617],
     'Yield_2': [961, 824, 684, 599, 514],
     'Yield_3': [769, 659, 547, 479, 411],
     'Sand_dune': [1201, 1030, 855, 749, 642]}).set_index('min_apdsi')


def yield_in_year(year, zones, yield_types, drought_index):
    yields = np.zeros(zones.shape)
    for zone in drought_index.columns:
        # TODO: not the most elegant way to do this
        are_zone = zones == zone.encode('utf-8')
        for min_pdsi in YIELD_DATA.index:
            # note they actually had boundaries different, at -1.0 they fell in -3.0 class
            if drought_index[zone][year] >= min_pdsi:
                for yieldtype in YIELD_DATA.columns:
                    are_yieldtype = yield_types == yieldtype.encode('utf-8')
                    affected_patches = np.logical_and(are_zone, are_yieldtype)
                    yields[affected_patches] = YIELD_DATA[yieldtype][min_pdsi]
                break

    return yields

class MapData:

    def __init__(self, datadir):
        self.zones, self.yields = read_map(datadir + 'map.txt')
        self.drought_index = read_PDSI(datadir + 'adjustedPDSI.txt')

    def mean_yield_in_year(self, year):
        return yield_in_year(year, self.zones, self.yields, self.drought_index)

    @property
    def shape(self):
        return self.zones.shape

# all this indicates it would be good to have a data structure to
# compute, store and return all this data
if __name__ == "__main__":
    datadir = '/home/hugs/Desktop/app/models/Sample Models/Social Science/Unverified/data/'
    z, y = read_map(datadir + 'map.txt')
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    for v in np.unique(z):
        plt.scatter(*np.nonzero(z==v))
    plt.show()
    f, ax = plt.subplots()
    for v in np.unique(y):
        plt.scatter(*np.nonzero(y==v))
    plt.show()
    drought_index = read_PDSI(datadir + "adjustedPDSI.txt")
    print(drought_index)
    for year in range(START_YEAR, END_YEAR):
        plt.imshow(yield_in_year(year, z, y, drought_index))
        plt.title(f'crop yield {year} AD')
        plt.show()
