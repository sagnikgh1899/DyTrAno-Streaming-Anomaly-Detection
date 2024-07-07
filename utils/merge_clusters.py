"""
Contains the necessary functions to perform cluster-merging
"""

import numpy as np
from tqdm import tqdm
from utils import data_utils, extract_data, pruning_utils, clustering_utils, constants


def find_nearest_point_in_cluster(data_point, cluster_indices, data, actual_idx):
    """
    Finds the nearest neighbor within the same cluster.
    """
    cluster_points = data[cluster_indices]
    cluster_points = cluster_points[cluster_indices != actual_idx]
    distances = np.linalg.norm(cluster_points - data_point, axis=1)
    nearest_index = cluster_indices[cluster_indices != actual_idx][np.argmin(distances)]
    min_distance = np.min(distances)

    return nearest_index, min_distance


def check_different_cluster_neighbors_helper(filtered_labels, pruned_neighbors_list):
    """
    Helper function to check and print points that have pruned neighbors
    belonging to different clusters.
    """
    data = data_utils.get_data(extract_data.get_raw_data_path())
    different_cluster_neighbors = []

    print("\nIdentifying neighbors belonging to different clusters...")
    for idx, neighbors in tqdm(enumerate(pruned_neighbors_list)):
        current_label = filtered_labels[idx]
        if current_label == -1:
            continue

        for neighbor_idx in neighbors:
            neighbor_label = filtered_labels[neighbor_idx]
            if neighbor_label in (current_label, -1):
                continue

            distance_between_points = pruning_utils.calculate_distance(data[idx],
                                                                       data[neighbor_idx])

            different_cluster_neighbors.append((idx, neighbor_idx, neighbor_label,
                                                distance_between_points))

    different_cluster_neighbors.sort(key=lambda x: x[3])
    return different_cluster_neighbors


def get_different_cluster_neighbors(filtered_labels, pruned_neighbors_list):
    """
    Returns the points that have pruned neighbors
    belonging to different clusters.
    """
    return check_different_cluster_neighbors_helper(filtered_labels, pruned_neighbors_list)


def check_different_cluster_neighbors(filtered_labels, pruned_neighbors_list):
    """
    Prints out points having neighbors belonging to a different cluster.\
    To be used only for debugging.
    Note: Set debugging=True in process_different_cluster_neighbors to enable this
    """
    different_cluster_neighbors = get_different_cluster_neighbors(filtered_labels,
                                                                  pruned_neighbors_list)
    print("Points with pruned neighbors belonging to different clusters:")
    for point, neighbor, neighbor_label, distance_between_points in different_cluster_neighbors:
        print(
            f"Point {point} (Cluster {filtered_labels[point]}) has neighbor "
            f"{neighbor} (Cluster {neighbor_label}) with d1 = {distance_between_points:.3f}")


def update_child_labels(node, new_label, labels, all_node_maps, old_label):
    """
    Recursively update the labels and cluster IDs of the children of the given node.
    """
    node_index = node.get_index()
    node.cluster_id = new_label
    labels[node_index] = new_label
    all_node_maps[new_label][node_index] = node
    if old_label in all_node_maps and node_index in all_node_maps[old_label]:
        del all_node_maps[old_label][node_index]

    for child in node.get_children():
        update_child_labels(child, new_label, labels, all_node_maps, old_label)


# pylint: disable=R0913
def cluster_reduction_helper(data_idx, all_node_maps, old_cluster_label,
                             new_cluster_label, neighbor_idx, labels, densities):
    """
    Helper function that updates the node's labels,
    its child node's labels, removes it from the previous parent,
    and adds it to the new parent.
    """
    if data_idx in all_node_maps[old_cluster_label]:
        node = all_node_maps[old_cluster_label][data_idx]
        previous_parent_node = node.get_parent()
        update_child_labels(node, new_cluster_label, labels, all_node_maps,
                            old_cluster_label)
        # Delete the datapoint from children list of its previous parent node
        if previous_parent_node is not None:
            previous_parent_node.get_children().remove(node)
        # We already delete the datapoint from all_node_maps of its previous cluster
        # in the update_child_labels function
        # No datapoint in a cluster - remove the cluster from all_node_maps
        if len(all_node_maps[old_cluster_label]) == 0:
            del all_node_maps[old_cluster_label]

    labels[data_idx] = new_cluster_label
    cluster_id = new_cluster_label

    node_map = all_node_maps[cluster_id]

    parent_node = node_map[neighbor_idx]

    new_root_node = parent_node
    point_density = densities[data_idx]
    while point_density > new_root_node.get_density():
        if new_root_node.get_parent() is None:
            current_root_node = new_root_node
            new_root_node = None
            break
        new_root_node = new_root_node.get_parent()

    new_node = clustering_utils.TreeNode(data_idx, point_density, new_root_node,
                                         cluster_id)
    new_node.set_parent(new_root_node)

    if new_root_node is None:
        new_node.add_child(current_root_node)
    else:
        new_root_node.add_child(new_node)
    node_map[data_idx] = new_node

    return labels, all_node_maps


def process_different_cluster_neighbors(labels, pruned_neighbors_list, all_node_maps,
                                        densities, debugging=False):
    """
    Main function that checks if merging is possible or not
    Note: For the time being density-criterion is not considered.
          Only distance has been considered
    """
    print("\nStarting merging of clusters...")
    different_cluster_neighbors = get_different_cluster_neighbors(labels,
                                                                  pruned_neighbors_list)
    if debugging:
        check_different_cluster_neighbors(labels, pruned_neighbors_list)
    for data_idx, neighbor_idx, neighbor_label, distance_between_points in different_cluster_neighbors:
        data_label = labels[data_idx]
        new_neighbor_label = labels[neighbor_idx]

        if data_label == new_neighbor_label:
            continue

        data_point_cluster_size = sum(1 for label in labels if label == data_label)
        neighbor_cluster_size = sum(1 for label in labels if label == new_neighbor_label)

        if data_point_cluster_size > neighbor_cluster_size:
            if distance_from_current_parent(neighbor_idx, all_node_maps, new_neighbor_label) < \
                    distance_between_points and \
                    satisfies_density_criterion(neighbor_idx, data_idx, densities,
                                                constants.BETA, constants.DELTA):
                labels, all_node_maps = cluster_reduction_helper(neighbor_idx, all_node_maps,
                                                                 new_neighbor_label, data_label,
                                                                 data_idx, labels, densities)

        else:
            if distance_from_current_parent(data_idx, all_node_maps, data_label) < \
                    distance_between_points and \
                    satisfies_density_criterion(data_idx, neighbor_idx, densities,
                                                constants.BETA, constants.DELTA):
                labels, all_node_maps = cluster_reduction_helper(data_idx, all_node_maps,
                                                                 data_label, new_neighbor_label,
                                                                 neighbor_idx, labels, densities)

    new_filtered_labels, all_node_maps = renumber_labels(labels, all_node_maps)
    return assign_singular_nodes_as_anomalies(all_node_maps, new_filtered_labels)


def renumber_labels(new_filtered_labels, all_node_maps, anomaly_label=-1):
    """
    Re-adjusts the label numbers after cluster merging
    """
    unique_labels = sorted(set(new_filtered_labels) - {anomaly_label})
    label_mapping = {old_label: new_label + 1 for new_label, old_label
                     in enumerate(unique_labels)}
    label_mapping[anomaly_label] = anomaly_label
    new_filtered_labels = [label_mapping[label] for label in new_filtered_labels]

    for old_cluster_id, node_map in all_node_maps.items():
        if old_cluster_id in label_mapping:
            new_cluster_id = label_mapping[old_cluster_id]
            for node in node_map.values():
                node.cluster_id = new_cluster_id

    all_node_maps = {label_mapping[old_key]: new_key for old_key, new_key
                     in all_node_maps.items() if old_key in label_mapping}

    return new_filtered_labels, all_node_maps


def satisfies_density_criterion(data_idx, neighbor_idx, densities, beta, delta):
    current_ewma_value = densities[neighbor_idx]
    data_point_density = densities[data_idx]
    ewma_value = clustering_utils.ewma(data_point_density, current_ewma_value, beta)
    if abs((ewma_value - data_point_density) / ewma_value) <= delta:
        return True
    return False


def distance_from_current_parent(data_idx, all_node_maps, current_data_label):
    current_node = all_node_maps[current_data_label][data_idx]
    current_parent_node = current_node.get_parent()
    if current_parent_node is not None:
        return pruning_utils.calculate_distance(data_idx, current_parent_node.get_index())
    return 0


def assign_singular_nodes_as_anomalies(all_node_maps, labels):
    for cluster_id, node_map in all_node_maps.items():
        if len(node_map) == 1:
            data_idx = next(iter(node_map))
            labels[data_idx] = -1
            node_map[data_idx].cluster_id = -1
    return labels, all_node_maps
