# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0

def h1(current_node, objective_node) -> np.float32:
    """ First heuristic: Manhattan distance multiplied by EPSILON """
    global NODES_EXPANDED

    # Calcular la distancia Manhattan
    manhattan_distance = abs(current_node[0] - objective_node[0]) + abs(current_node[1] - objective_node[1])

    # Multiplicar por EPSILON
    h = manhattan_distance * EPSILON

    # Incrementar el contador de nodos expandidos
    NODES_EXPANDED += 1

    return h

def h2(current_node, objective_node) -> np.float32:
    """ Segunda heurística: distancia euclidiana multiplicada por EPSILON """
    global NODES_EXPANDED

    # Calcular la distancia euclidiana entre el nodo actual y el nodo objetivo
    euclidean_distance = np.sqrt((current_node[0] - objective_node[0])**2 + (current_node[1] - objective_node[1])**2)

    # Multiplicar la distancia por EPSILON para garantizar que sea admisible
    h = euclidean_distance * EPSILON

    NODES_EXPANDED += 1
    return h

def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    # Crear un grafo dirigido
    G = nx.DiGraph()

    # Obtener las dimensiones del mapa
    height, width = detection_map.shape

    # Solo exploramos hacia abajo y a la derecha para evitar duplicar aristas
    moves = [(1, 0), (0, 1)]

    # Iterar sobre cada celda del mapa
    for y in range(height):
        for x in range(width):
            current_node = (y, x)

            # Agregar el nodo al grafo (incluso si no tiene conexiones)
            G.add_node(current_node)

            # Iterar sobre los movimientos posibles
            for dy, dx in moves:
                neighbor_y, neighbor_x = y + dy, x + dx

                # Verificar si el vecino está dentro de los límites del mapa
                if 0 <= neighbor_y < height and 0 <= neighbor_x < width:
                    neighbor_node = (neighbor_y, neighbor_x)
                    # Calcular el costo del movimiento como el valor de detección de la celda destino.
                    cost_to_neighbor = detection_map[neighbor_y, neighbor_x]
                    # Calcular el coste del movimiento inverso al nodo actual
                    cost_to_current = detection_map[y, x]
                    # Agregar la arista al grafo si el costo está dentro de la tolerancia
                    if cost_to_neighbor <= tolerance:
                        G.add_edge(current_node, neighbor_node, weight=cost_to_neighbor)
                    if cost_to_current <= tolerance:
                        G.add_edge(neighbor_node, current_node, weight=cost_to_current)
    return G

def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """ Converts coordinates from (lat, lon) into (x, y) grid indices """
    # Generate evenly spaced grid points for latitude and longitude
    lat_grid = np.linspace(boundaries.min_lat, boundaries.max_lat, map_height)
    lon_grid = np.linspace(boundaries.min_lon, boundaries.max_lon, map_width)

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
        

def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array, 
                 initial_location_index: np.int32, 
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32) -> tuple:
    """ Implementation of the main searching / path finding algorithm """
    global NODES_EXPANDED
    NODES_EXPANDED = 0  # Reset the nodes expanded counter

    # Step 1: Order the locations based on proximity
    ordered_locations = []
    remaining_locations = locations.tolist()
    current_location = remaining_locations.pop(initial_location_index)
    ordered_locations.append(current_location)

    while remaining_locations:
        # Find the closest location to the current location
        distances = [np.linalg.norm(np.array(current_location) - np.array(loc)) for loc in remaining_locations]
        closest_index = np.argmin(distances)
        current_location = remaining_locations.pop(closest_index)
        ordered_locations.append(current_location)

    ordered_locations = np.array(ordered_locations)

    # Step 2: Discretize the ordered locations into grid coordinates
    discretized_locations = discretize_coords(
        high_level_plan=ordered_locations,
        boundaries=boundaries,
        map_width=map_width,
        map_height=map_height
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
                G,
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
    return solution_plan, NODES_EXPANDED

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """
    total_cost = 0.0

    # Iterate through each path in the solution plan
    for path in solution_plan:
        # Iterate through consecutive nodes in the path
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]

            # Add the weight (cost) of the edge between start and end
            total_cost += G[start][end]['weight']

    return np.float32(total_cost)
