# Required imports
import os
import sys
import json
import numpy as np

from Map import Map
from Boundaries import Boundaries
from SearchEngine import SearchEngine
from utils.plot_utils import plot_radar_locations, plot_detection_fields, plot_graph_on_detection_map, plot_solution

def parse_args() -> dict:
    """ Parses the main arguments of the program and returns them stored in a dictionary """
    json_path            = f"{os.getcwd()}/src/unittest/scenarios.json"
    scenario_json        = sys.argv[1]
    tolerance            = float(sys.argv[2])
    execution_parameters = {}
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            key = list(entry.keys())[0]
            if key == scenario_json:
                execution_parameters = entry[key]
                break
    execution_parameters["tolerance"] = tolerance
    return execution_parameters

# System's main function
def main() -> None:

    # Parse the input parameters (arguments) of the program (current execution)
    execution_parameters = parse_args()

    # Set the pseudo-random number generator seed (DO NOT MODIFY)
    np.random.seed(42)

    # Set boundaries
    boundaries = Boundaries(max_lat=execution_parameters['max_lat'],
                            min_lat=execution_parameters['min_lat'],
                            max_lon=execution_parameters['max_lon'],
                            min_lon=execution_parameters['min_lon'],
                            height=execution_parameters['H'],
                            width=execution_parameters['W'],)
    
    # Define the map with its corresponding boundaries and coordinates
    M = Map(boundaries=boundaries)
    
    # Generate random radars
    n_radars = execution_parameters['n_radars']
    M.generate_radars(n_radars=n_radars)
    radar_locations = M.get_radars_locations_numpy()

    # Plot the radar locations (latitude increments from bottom to top)
    plot_radar_locations(boundaries=boundaries, radar_locations=radar_locations, POIs=execution_parameters['POIs'])	

    # Compute the detection map (sets the costs for each cell)
    detection_map = M.compute_detection_map()

    # Plot the detection map (detection fields)
    plot_detection_fields(detection_map=detection_map)

    # Build the graph from the detection map
    G = SearchEngine(detection_map=detection_map,
                     tolerance=execution_parameters['tolerance'],
                     boundaries=boundaries)

    # Plot the graph on top of the detection map
    plot_graph_on_detection_map(G=G, detection_map=detection_map)
    
    # Get the POI's that the plane must visit
    POIs = np.array(execution_parameters['POIs'], dtype=np.float64)

    # Compute the solution
    solution_plan, nodes_expanded = G.path_finding(
                                 heuristic_function=G.h1,
                                 locations=POIs, 
                                 initial_location_index=0)
        
    # Compute the solution cost
    path_cost = G.compute_path_cost(solution_plan=solution_plan)

    # # Some verbose of the total cost and the number of expanded nodes
    print(f"Total path cost: {path_cost}")
    print(f"Number of expanded nodes: {nodes_expanded}")

    # Plot the solution
    plot_solution(detection_map=detection_map, solution_plan=solution_plan)

if __name__ == '__main__':
    main()
