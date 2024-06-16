"""
Contains the utility functions to return the pruned neighbor list
"""

import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from utils import constants, data_utils, extract_data


def calculate_distance(data_point1, data_point2):
    """
    Calculates distance between two data points
    """
    return np.sqrt(np.sum((data_point1 - data_point2) ** 2))


def find_k_nearest_neighbors(data, k):
    """
    Returns the k nearest neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=constants.CLUSTERING_ALGORITHM).fit(data)
    _, indices = nbrs.kneighbors(data)
    return indices


def calculate_weight_vector(g_val, gamma):
    """
    Calculates the weight vector
    """
    g_inv = np.linalg.inv(g_val)
    omega = (gamma / 2) @ g_inv
    return omega / np.sum(omega)


def calculate_final_contributions(data, omega, i, neigh):
    """
    Calculates the neighborhood construction weights
    """
    w_val = np.zeros(len(neigh))
    for t_val, neighbor in enumerate(neigh):
        w_val[t_val] = omega[t_val] / calculate_distance(data[i], data[neighbor])
    return w_val


def prune_neighbors(w_values, neigh, epsilon):
    """
    Returns the list of pruned neighbors
    """
    sorted_indices = np.argsort(-w_values)
    w_values = w_values[sorted_indices]
    neigh = neigh[sorted_indices]
    t_val = 1
    while t_val < len(w_values):
        if abs(w_values[t_val] - w_values[t_val - 1]) / w_values[t_val - 1] > epsilon:
            break
        t_val += 1
    return neigh[:t_val]


def optimal_neighborhood_selection(k, epsilon, sigma):
    """
    Returns the optimal neighborhood list
    """
    data = data_utils.get_data(extract_data.get_raw_data_path())
    num_of_data_points = len(data)
    pruned_neighbors_list = []
    k_nearest_neighbors_of_all_datapoints = find_k_nearest_neighbors(data, k)

    print("\nStarting pruned neighborhood calculation...")
    for num in tqdm(range(num_of_data_points)):
        neigh = k_nearest_neighbors_of_all_datapoints[num]
        actual_k = len(neigh)
        gamma = np.random.rand(actual_k)
        g_val = sigma * np.identity(k)
        omega = calculate_weight_vector(g_val, gamma)
        w_val = calculate_final_contributions(data, omega, num, neigh)
        pruned_neigh = prune_neighbors(w_val, neigh, epsilon)
        pruned_neighbors_list.append(pruned_neigh)

    return pruned_neighbors_list
