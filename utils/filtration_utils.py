"""
Contains the helper functions for filtration of potential anomalies
"""

import numpy as np
from tqdm import tqdm
from utils import data_utils, extract_data, clustering_utils, pruning_utils, constants


def calculate_cluster_icd(cluster_points, data):
    """
    Calculate the mean intra-cluster distance for the given cluster points
    using vectorized operations
    """
    cluster_points = list(cluster_points)
    num_points = len(cluster_points)
    if num_points <= 1:
        return 0

    cluster_data = data[cluster_points]
    diff = cluster_data[:, np.newaxis, :] - cluster_data[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    upper_tri_indices = np.triu_indices(num_points, k=1)
    total_distance = np.sum(distances[upper_tri_indices])
    count = len(upper_tri_indices[0])

    return total_distance / count


def find_nearest_inlier(potential_anomaly_index, labels, data):
    """
    Find the nearest non-anomalous neighbor of the potential anomaly
    """
    potential_anomaly = data[potential_anomaly_index]
    inliers = np.where((labels != -1) & (labels != 0))[0]
    distances = [pruning_utils.calculate_distance(potential_anomaly, data[inlier_index]) for
                 inlier_index in inliers]
    nearest_inlier_index = inliers[np.argmin(distances)]
    return nearest_inlier_index


# pylint: disable=R0914
def filter_potential_anomalies(labels, all_node_maps, densities):
    """
    Returns the labels after selection of confirmed anomalies and inliers
    """
    data = data_utils.get_data(extract_data.get_raw_data_path())
    potential_anomalies = np.where(labels == -1)[0]

    print("\nStarting filtration of potential anomalies...")
    cluster_icds = {}

    for anomaly_index in tqdm(potential_anomalies):
        nearest_inlier_index = find_nearest_inlier(anomaly_index, labels, data)

        cluster_id = labels[nearest_inlier_index]
        node_map = all_node_maps[cluster_id]

        if cluster_id not in cluster_icds:
            icd_inlier = calculate_cluster_icd(list(node_map.keys()), data)
            cluster_icds[cluster_id] = icd_inlier
        else:
            icd_inlier = cluster_icds[cluster_id]

        dist1 = icd_inlier
        dist2 = pruning_utils.calculate_distance(data[anomaly_index],
                                                 data[nearest_inlier_index])

        if dist2 < constants.DELTA_FOR_FILTRATION * dist1:
            labels[anomaly_index] = labels[nearest_inlier_index]
            cluster_id = labels[nearest_inlier_index]
            node_map = all_node_maps[cluster_id]

            parent_node = node_map[nearest_inlier_index]

            new_root_node = parent_node
            anomaly_converted_to_inlier_density = densities[anomaly_index]
            while anomaly_converted_to_inlier_density > new_root_node.get_density():
                if new_root_node.get_parent() is None:
                    current_root_node = new_root_node
                    new_root_node = None
                    break
                new_root_node = new_root_node.get_parent()

            new_node = clustering_utils.TreeNode(anomaly_index,
                                                 anomaly_converted_to_inlier_density,
                                                 new_root_node, cluster_id)
            new_node.set_parent(new_root_node)

            if new_root_node is None:
                new_node.add_child(current_root_node)
            else:
                new_root_node.add_child(new_node)
            node_map[anomaly_index] = new_node

        else:
            labels[anomaly_index] = -1  # Confirmed anomaly

    return labels
