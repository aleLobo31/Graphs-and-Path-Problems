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
                 radars:     np.array=None):
        self.boundaries = boundaries        # Boundaries of the map
        self.radars     = radars            # List containing the radars (objects)

    def generate_radars(self, n_radars: np.int32) -> None:
        """ Generates n-radars randomly and inserts them into the radars list """
        # Select random coordinates inside the boundaries of the map
        lat_range = self.boundaries.lat_range
        lon_range = self.boundaries.lon_range
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
        # Initialize the detection map array
        detection_map = np.full(shape=(self.boundaries.height, self.boundaries.width), fill_value=EPSILON, dtype=np.float32)
        lat_range = self.boundaries.lat_range
        lon_range = self.boundaries.lon_range

        # Add the detection level to each cell of the map
        for radar in tqdm(self.radars, desc="Computing detection map", unit="radar"):
            for i, lat in enumerate(lat_range):
                for j, lon in enumerate(lon_range):
                    level = radar.compute_detection_level(latitude=lat, longitude=lon)
                    # If the detection level is greater than the current one, update it
                    if detection_map[i, j] < level:
                        detection_map[i, j] = level

        # Normalize the detection map
        min_value = np.min(detection_map)
        max_value = np.max(detection_map)
        if min_value == max_value:
            detection_map = np.full(shape=(self.boundaries.height, self.boundaries.width), fill_value=EPSILON, dtype=np.float32)
        else:
            detection_map = ((detection_map - min_value) / (max_value - min_value)) * (1 - EPSILON) + EPSILON
        return detection_map