# Required imports
import numpy as np

class Boundaries:
    """ Class that defines the limits (in geodetic coordinates) of a map """
    def __init__(self, 
                 max_lat: np.float32, 
                 min_lat: np.float32,
                 max_lon: np.float32,
                 min_lon: np.float32,
                 height: np.int32=None,
                 width: np.int32=None) -> None:
        self.max_lat = max_lat      # Maximum latitude of the map
        self.min_lat = min_lat      # Minimum latitude of the map
        self.max_lon = max_lon      # Maximum longitude of the map
        self.min_lon = min_lon      # Minimum longitude of the map
        self.height = height        # Number of coordinates in the y-axis
        self.width = width          # Number of coordinates in the x-axis
        self.lat_range = np.linspace(self.min_lat, self.max_lat, num=height)      # Range of latitudes in the map
        self.lon_range = np.linspace(self.min_lon, self.max_lon, num=self.width)  # Range of longitudes in the map