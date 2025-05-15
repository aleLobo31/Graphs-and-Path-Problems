import numpy as np
import networkx as nx

from Map import EPSILON
from networkx import DiGraph
from Boundaries import Boundaries

class SearchEngine(DiGraph):
    """
    A class that wraps a directed graph (DiGraph) and provides additional functionality.
    """
    def __init__(self, detection_map: np.array, tolerance: np.float32, boundaries: Boundaries) -> None:
        """
        Initialize the SearchEngine with the given arguments.
        """
        super().__init__()
        self.boundaries = boundaries
        self.nodes_expanded = 0
        self.build_graph(detection_map=detection_map, tolerance=tolerance)

    def build_graph(self, detection_map: np.array, tolerance: np.float32) -> None:
        # Get the dimensions of the map
        height, width = detection_map.shape

        # Only explore down and right to avoid duplicating edges
        moves = [(1, 0), (0, 1)]

        # Iterate over each cell in the map
        for y in range(height):
            for x in range(width):
                current_node = (y, x)
                # Add the node to the graph (even if it has no connections)
                self.add_node(current_node)
                # Iterate over possible moves
                for dy, dx in moves:
                    neighbor_y, neighbor_x = y + dy, x + dx
                    # Check if the neighbor is within the map boundaries
                    if 0 <= neighbor_y < height and 0 <= neighbor_x < width:
                        neighbour_node = (neighbor_y, neighbor_x)
                        # Calculate the movement cost as the detection value of the destination cell.
                        cost_to_neighbor = detection_map[neighbor_y, neighbor_x]
                        # Calculate the cost of the reverse movement to the current node
                        cost_to_current = detection_map[y, x]
                        # Add the edge to the graph if the cost is within the tolerance
                        if cost_to_neighbor <= tolerance:
                            self.add_edge(current_node, neighbour_node, weight=cost_to_neighbor)
                        if cost_to_current <= tolerance:
                            self.add_edge(neighbour_node, current_node, weight=cost_to_current)

    def h1(self, current_node: tuple, objective_node: tuple) -> np.float32:
        """ First heuristic: Manhattan distance multiplied by EPSILON """
        # Calculate the Manhattan distance
        manhattan_distance = abs(current_node[0] - objective_node[0]) + abs(current_node[1] - objective_node[1])

        # Multiply by EPSILON
        h = manhattan_distance * EPSILON

        # Increment the expanded nodes counter
        self.nodes_expanded += 1

        return h

    def h2(self, current_node: tuple, objective_node: tuple) -> np.float32:
        """ Second heuristic: Euclidean distance multiplied by EPSILON """
        # Calculate the Euclidean distance between the current node and the objective node
        euclidean_distance = np.sqrt((current_node[0] - objective_node[0])**2 + (current_node[1] - objective_node[1])**2)

        # Multiply the distance by EPSILON to ensure admissibility
        h = euclidean_distance * EPSILON

        # Increment the expanded nodes counter
        self.nodes_expanded += 1
        return h
    
    def discretize_coords(self, high_level_plan: np.array) -> np.array:
        """ Converts coordinates from (lat, lon) into (x, y) grid indices """
        # Generate evenly spaced grid points for latitude and longitude
        lat_grid = self.boundaries.lat_range
        lon_grid = self.boundaries.lon_range

        # Initialize the result array
        discretized_plan = np.zeros((high_level_plan.shape[0], 2), dtype=np.int32)
        
        for i, (lat_coord, lon_coord) in enumerate(high_level_plan):
            # Find the closest latitude index
            lat_idx = np.argmin(np.abs(lat_grid - lat_coord))
            discretized_plan[i][0] = lat_idx

            # Find the closest longitude index
            lon_idx = np.argmin(np.abs(lon_grid - lon_coord))
            discretized_plan[i][1] = lon_idx

        return discretized_plan
    
    def order_locations(self, locations: np.array, initial_location_index: int):
        # Initialize the ordered list of locations
        ordered_locations = []
        # Convert the input locations to a list for easier manipulation
        remaining_locations = locations.tolist()
        # Remove and store the initial location based on the given index
        current_location = remaining_locations.pop(initial_location_index)
        ordered_locations.append(current_location)

        # Continue until all locations have been ordered
        while len(remaining_locations) != 0:
            # Compute Manhattan distances from the current location to all remaining locations
            distances = [abs(current_location[0] - loc[0]) + abs(current_location[1] - loc[1]) for loc in remaining_locations]
            # Find the index of the closest location
            closest_index = np.argmin(distances)
            # Remove the closest location from the list and set it as the current location
            current_location = remaining_locations.pop(closest_index)
            # Add the closest location to the ordered list
            ordered_locations.append(current_location)

        # Convert the ordered list back to a numpy array and return it
        ordered_locations = np.array(ordered_locations)
        return ordered_locations

    def path_finding(self,
                    heuristic_function: callable,
                    locations: np.array, 
                    initial_location_index: np.int32) -> tuple:
        """ Implementation of the main searching / path finding algorithm """
        # Step 0: Reset expanded nodes counter
        self.nodes_expanded = 0
        
        # Step 1: Order the locations based on proximity
        ordered_locations = self.order_locations(locations, initial_location_index)

        # Step 2: Discretize the ordered locations into grid coordinates
        discretized_locations = self.discretize_coords(
            high_level_plan=ordered_locations
        )

        # Step 3: Find paths between all locations in order
        solution_plan = []
        current_index = 0  # Start from the first location in the ordered list

        for next_index in range(1, len(discretized_locations)):
            # Get the current and next location in the grid
            start = tuple(discretized_locations[current_index])
            goal = tuple(discretized_locations[next_index])

            # Use A* to find the path between the current and next location
            try:
                path = nx.astar_path(
                    self,
                    source=start,
                    target=goal,
                    heuristic=heuristic_function,
                    weight='weight'
                )
            except nx.NetworkXNoPath:
                raise ValueError(f"No path found between {start} and {goal}.")

            # Append the path to the solution plan
            solution_plan.append(path)

            # Update the current index
            current_index = next_index

        # Return the solution plan and the number of expanded nodes
        return solution_plan, self.nodes_expanded
    
    def compute_path_cost(self, solution_plan: list) -> np.float32:
        """ Computes the total cost of the whole planning solution """
        total_cost = 0.0
        # Iterate through each path in the solution plan
        for path in solution_plan:
            # Iterate through consecutive nodes in the path
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]

                # Add the weight (cost) of the edge between start and end
                total_cost += self[start][end]['weight']

        return np.float32(total_cost)