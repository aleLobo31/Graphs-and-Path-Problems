# Required imports
import numpy as np
from Location import Location
from Boundaries import Boundaries
from Radar import Radar
from tqdm import tqdm

# Constant that avoids setting cells to have an associated cost of zero
EPSILON = 1e-4

class Map:
    """ Class that models the map for the simulation """
    def __init__(self, 
                 boundaries: Boundaries,
                 height:     np.int32, 
                 width:      np.int32, 
                 radars:     np.array=None):
        self.boundaries = boundaries        # Boundaries of the map
        self.height     = height            # Number of coordinates in the y-axis
        self.width      = width             # Number of coordinates int the x-axis
        self.radars     = radars            # List containing the radars (objects)

    def generate_radars(self, n_radars: np.int32) -> None:
        """ Generates n-radars randomly and inserts them into the radars list """
        # Select random coordinates inside the boundaries of the map
        lat_range = np.linspace(start=self.boundaries.min_lat, stop=self.boundaries.max_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)
        rand_lats = np.random.choice(a=lat_range, size=n_radars, replace=False)
        rand_lons = np.random.choice(a=lon_range, size=n_radars, replace=False)
        self.radars = []        # Initialize 'radars' as an empty list

        # Loop for each radar that must be generated
        for i in range(n_radars):
            # Create a new radar
            new_radar = Radar(location=Location(latitude=rand_lats[i], longitude=rand_lons[i]),
                              transmission_power=np.random.uniform(low=1, high=1000000),
                              antenna_gain=np.random.uniform(low=10, high=50),
                              wavelength=np.random.uniform(low=0.001, high=10.0),
                              cross_section=np.random.uniform(low=0.1, high=10.0),
                              minimum_signal=np.random.uniform(low=1e-10, high=1e-15),
                              total_loss=np.random.randint(low=1, high=10),
                              covariance=None)

            # Insert the new radar
            self.radars.append(new_radar)
        return
    
    def get_radars_locations_numpy(self) -> np.array:
        """ Returns an array with the coordiantes (lat, lon) of each radar registered in the map """
        locations = np.zeros(shape=(len(self.radars), 2), dtype=np.float32)
        for i in range(len(self.radars)):
            locations[i] = self.radars[i].location.to_numpy()
        return locations
    
    def compute_detection_map(self) -> np.array:
        """ Computes the detection map for each coordinate in the map (with all the radars) """
        detection_map = np.full((self.height, self.width), EPSILON, dtype=np.float32)

        # Use a dictionary for detection levels: key = "i_j", value = list of levels
        detection_levels = {}

        # Generate geodetic coordinates for each cell in the map
        lat_range = np.linspace(stop=self.boundaries.max_lat, start=self.boundaries.min_lat, num=self.height)
        lon_range = np.linspace(start=self.boundaries.min_lon, stop=self.boundaries.max_lon, num=self.width)

        # Compute detection levels for each cell from each radar
        min_level = np.inf
        max_level = -np.inf
        for i, lat in enumerate(lat_range):
            for j, lon in enumerate(lon_range):
                key = f"{i}_{j}"
                if key not in detection_levels:
                    detection_levels[key] = []
                for radar in self.radars:
                    level = radar.compute_detection_level(latitude=lat, longitude=lon)
                    detection_levels[key].append(level)
                    if level > max_level:
                        max_level = level
                    if level < min_level:
                        min_level = level
                # level = radar.compute_detection_level(latitude=lat, longitude=lon)
                # if level > detection_map[i, j]:
                #     detection_map[i, j] = level 

        # Min-Max normalization for each cell
        for key, levels in detection_levels.items():
            if max_level - min_level > 0:
                # Use the max normalized value for this cell
                normalized_levels = [(level - min_level) / (max_level - min_level) * (1 - EPSILON) for level in levels]
                i, j = map(int, key.split('_'))
                detection_map[i, j] += max(normalized_levels)
        
        return detection_map