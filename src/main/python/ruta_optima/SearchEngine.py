# Required imports
import numpy as np
import networkx as nx
from Boundaries import Boundaries
from Map import EPSILON

# Number of nodes expanded in the heuristic search (stored in a global variable to be updated from the heuristic functions)
NODES_EXPANDED = 0

def h1(current_node, objective_node) -> np.float32:
    """ First heuristic to implement """
    global NODES_EXPANDED
    h = 0
    ...
    NODES_EXPANDED += 1
    return h

def h2(current_node, objective_node) -> np.float32:
    """ Second heuristic to implement """
    global NODES_EXPANDED
    h = 0
    ...
    NODES_EXPANDED += 1
    return h

def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """ Builds an adjacency graph (not an adjacency matrix) from the detection map """
    # The only possible connections from a point in space (now a node in the graph) are:
    #   -> Go up
    #   -> Go down
    #   -> Go left
    #   -> Go right
    # Not every point has always 4 possible neighbors
    G = nx.DiGraph()

    # Obtener las dimensiones del mapa
    height, width = detection_map.shape

    # Definir los movimientos posibles (arriba, abajo, izquierda, derecha)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dy, dx)

    # Iterar sobre cada celda del mapa
    for y in range(height):
        for x in range(width):
            # Nodo actual
            current_node = (y, x)

            # Agregar el nodo al grafo (incluso si no tiene conexiones)
            G.add_node(current_node)

            # Iterar sobre los movimientos posibles
            for dy, dx in moves:
                neighbor_y, neighbor_x = y + dy, x + dx

                # Verificar si el vecino está dentro de los límites del mapa
                if 0 <= neighbor_y < height and 0 <= neighbor_x < width:
                    # Calcular el costo del movimiento
                    cost = detection_map[neighbor_y, neighbor_x]

                    # Agregar la arista al grafo si el costo está dentro de la tolerancia
                    if cost <= tolerance:
                        neighbor_node = (neighbor_y, neighbor_x)
                        G.add_edge(current_node, neighbor_node, weight=cost)
                        G.add_edge(neighbor_node, current_node, weight=cost)

    return G

def discretize_coords(high_level_plan: np.array, boundaries: Boundaries, map_width: np.int32, map_height: np.int32) -> np.array:
    """ Converts coordiantes from (lat, lon) into (x, y) """
    ...

def path_finding(G: nx.DiGraph,
                 heuristic_function,
                 locations: np.array, 
                 initial_location_index: np.int32, 
                 boundaries: Boundaries,
                 map_width: np.int32,
                 map_height: np.int32) -> tuple:
    """ Implementation of the main searching / path finding algorithm """
    ...

def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> np.float32:
    """ Computes the total cost of the whole planning solution """
    ...
