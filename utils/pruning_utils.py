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


def find_k_nearest_neighbors(data, k, seed=90):
    """
    Returns the k nearest neighbors
    """
    np.random.seed(seed)
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
    Calculates the neighborhood construction weights.
    """
    distances = np.linalg.norm(data[neigh] - data[i], axis=1)
    w_val = omega / distances
    return w_val


def prune_neighbors(w_values, neigh, epsilon):
    """
    Returns the list of pruned neighbors.
    """
    sorted_indices = np.argsort(-w_values)
    sorted_w_values = w_values[sorted_indices]
    sorted_neigh = np.array(neigh)[sorted_indices]
    differences = np.abs(np.diff(sorted_w_values) / sorted_w_values[:-1])
    t_val = np.argmax(differences > epsilon) + 1
    return sorted_neigh[:t_val]


def optimal_neighborhood_selection(k, epsilon, sigma):
    """
    Returns the optimal neighborhood list.
    """
    data = data_utils.get_data(extract_data.get_raw_data_path())
    num_of_data_points = len(data)
    pruned_neighbors_list = []
    k_nearest_neighbors_of_all_datapoints = find_k_nearest_neighbors(data, k)
    sigma_identity_k = sigma * np.identity(k)

    print("\nStarting pruned neighborhood calculation...")
    for num in tqdm(range(num_of_data_points)):
        neigh = k_nearest_neighbors_of_all_datapoints[num]
        actual_k = len(neigh)
        gamma = np.random.rand(actual_k)
        omega = calculate_weight_vector(sigma_identity_k[:actual_k, :actual_k], gamma)
        w_val = calculate_final_contributions(data, omega, num, neigh)
        pruned_neigh = prune_neighbors(w_val, neigh, epsilon)
        pruned_neighbors_list.append(pruned_neigh)

    return pruned_neighbors_list
