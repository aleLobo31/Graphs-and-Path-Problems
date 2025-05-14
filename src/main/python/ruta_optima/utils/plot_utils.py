import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from Boundaries import Boundaries
from SearchEngine import SearchEngine


def plot_radar_locations(boundaries: Boundaries, radar_locations: np.array, POIs: list) -> None:
    """ Auxiliary function for plotting the radar locations and the POIs """
    POIs = np.array(POIs, dtype=np.float64)

    plt.figure(figsize=(8, 8))
    plt.title("Radar locations in the map")
    plt.plot([boundaries.min_lon, boundaries.max_lon, boundaries.max_lon, boundaries.min_lon, boundaries.min_lon],
             [boundaries.max_lat, boundaries.max_lat, boundaries.min_lat, boundaries.min_lat, boundaries.max_lat],
             label='Boundaries',
             linestyle='--',
             c='black')
    # Plot radar locations
    plt.scatter(radar_locations[:, 1], radar_locations[:, 0], label='Radars', c='green')
    # Plot POIs
    plt.scatter(POIs[:, 1], POIs[:, 0], label='POIs', c='red', marker='x')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.legend()
    plt.show()
    return

def plot_detection_fields(detection_map: np.array, bicubic: bool=True) -> None:
    """ Auxiliary function for plotting the detection fields """
    plt.figure(figsize=(8, 8))
    plt.title("Radar detection fields")
    im = plt.imshow(X=detection_map, cmap='Greens', interpolation='bicubic' if bicubic else None)
    plt.colorbar(im, label='Detection values')
    plt.show()
    return

def plot_solution(detection_map: np.array, solution_plan: list, bicubic: bool=True) -> None:
    """ Auxiliary function for plotting the solution plan with markers in each POI """
    plt.figure(figsize=(8,8))
    plt.title("Solution plan")
    for i in range(len(solution_plan)):
        start_point = solution_plan[i][0]
        plt.scatter(start_point[1], start_point[0], c='black', marker='*', zorder=2)
        path_array = np.zeros(shape=(len(solution_plan[i]), 2))
        for j in range(len(path_array)):
            path_array[j] = solution_plan[i][j]
        plt.plot(path_array[:, 1], path_array[:, 0], zorder=1)
    final_point = solution_plan[-1][-1]
    plt.scatter(final_point[1], final_point[0], c='black', marker='*', label=f'Waypoints', zorder=2)
    im = plt.imshow(X=detection_map, cmap='Greens', interpolation='bicubic' if bicubic else None)
    plt.colorbar(im, label='Detection values')
    plt.legend()
    plt.show()
    return

def plot_graph_on_detection_map(G: SearchEngine, detection_map: np.array) -> None:
    """ Plots the graph on top of the detection map """
    plt.figure(figsize=(8, 8))
    plt.title("Graph over Detection Map")

    # Show the detection map
    im = plt.imshow(X=detection_map, cmap='Greens', interpolation='bicubic')
    plt.colorbar(im, label='Detection values')

    # Plot the graph
    pos = {(y, x): (x, y) for y, x in G.nodes()}  # Convertir nodos a coordenadas (x, y) para graficar
    nx.draw(G, pos, node_size=10, edge_color='blue', arrowsize=5, with_labels=False)

    plt.xlabel("Longitude (discretized)")
    plt.ylabel("Latitude (discretized)")
    plt.show()
