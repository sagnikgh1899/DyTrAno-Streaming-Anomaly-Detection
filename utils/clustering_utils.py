"""
Module for hierarchical tree-based clustering using density criteria.
"""

import numpy as np
from utils import data_utils, extract_data, pruning_utils


class TreeNode:
    """
    Represents a node in a tree structure for hierarchical clustering.
    """

    def __init__(self, index, density, parent, cluster_id):
        """
        Initialize a TreeNode with index, density, parent node, and cluster ID
        """
        self.index = index
        self.density = density
        self.parent = parent
        self.children = []
        self.cluster_id = cluster_id

    def get_index(self):
        """
        Returns index of current node
        """
        return self.index

    def set_parent(self, parent_node):
        """
        Assigns parent node to the current node
        """
        self.parent = parent_node

    def add_child(self, child_node):
        """
        Adds child to the current node
        """
        self.children.append(child_node)

    def get_parent(self):
        """
        Returns the parent node
        """
        return self.parent

    def get_children(self):
        """
        Returns list of child nodes
        """
        return self.children

    def get_parent_index(self):
        """
        Returns the parent node's index
        """
        return self.parent.get_index()

    def get_child_indexes(self):
        """
        Returns the child indexes
        """
        child_indexes = []
        for child in self.children:
            child_indexes.append(child.index)
        return child_indexes

    def get_density(self):
        """
        Returns density of the current node
        """
        return self.density

    def get_parent_density(self):
        """
        Returns the parent node's density
        """
        if self.parent:
            return self.parent.get_density()
        return None

    def get_children_densities(self):
        """
        Returns the density of child nodes
        """
        return [child.get_density() for child in self.children]

    def get_cluster_id(self):
        """
        Returns cluster id
        """
        return self.cluster_id

    def check_and_adjust_hierarchy(self):
        """
        Checks and adjusts the tree hierarchy
        """
        if self.parent is None:
            return

        if self.parent.get_density() < self.density:
            old_parent = self.parent
            new_parent = self.parent.get_parent()
            if self in self.parent.children:
                self.parent.children.remove(self)
            self.parent = new_parent
            new_parent.add_child(self)
            if old_parent in new_parent.children:
                new_parent.children.remove(old_parent)
            old_parent.parent = self
            self.add_child(old_parent)

        for child in self.children:
            child.check_and_adjust_hierarchy()


def calculate_density(data, pruned_neighbors_list):
    """
    Calculate densities for each point based on its pruned neighbors.
    """
    densities = []
    for i in range(len(data)):
        neighbors = pruned_neighbors_list[i]
        k_opt = len(neighbors)
        if k_opt == 0:
            densities.append(0)
            continue

        e_values = [pruning_utils.calculate_distance(data[i], data[neighbor]) for
                    neighbor in neighbors]
        e_k_opt = e_values[-1] if e_values else 0

        if e_k_opt == 0:
            densities.append(0)
        else:
            density = sum(e_values) / (np.pi * e_k_opt ** 2)
            densities.append(density)
    return densities


def ewma(current_value, previous_ewma, beta):
    """
    Calculate Exponentially Weighted Moving Average (EWMA).
    """
    return beta * previous_ewma + (1 - beta) * current_value


# pylint: disable=R0913,R0914
def cluster_tree(root_index, data, pruned_neighbors_list, labels, densities,
                 delta, cluster_id, beta, parent_node):
    """
    Recursively build a cluster tree starting from a root node.
    """
    root_density = densities[root_index]
    ewma_value = root_density
    root_node = TreeNode(root_index, root_density, parent_node, cluster_id)
    node_map = {root_index: root_node}
    root_neighbors = pruned_neighbors_list[root_index]

    for child_index in root_neighbors:
        if child_index == root_index or labels[child_index] != 0:
            continue

        child_density = densities[child_index]
        ewma_value = ewma(child_density, ewma_value, beta)

        if abs((ewma_value - child_density) / ewma_value) <= delta:
            labels[child_index] = cluster_id

            new_root_node = root_node
            while child_density > new_root_node.get_density():
                # pylint: disable=W0511
                # TODO: What if the child density is higher than the actual root node
                new_root_node = new_root_node.get_parent()

            child_node = TreeNode(child_index, child_density, new_root_node, cluster_id)
            child_node.set_parent(new_root_node)
            new_root_node.add_child(child_node)
            node_map[child_index] = child_node

            _, child_node_map = cluster_tree(child_index, data, pruned_neighbors_list,
                                             labels, densities, delta, cluster_id,
                                             beta, new_root_node)
            node_map.update(child_node_map)

    return root_node, node_map


def tree_based_clustering(pruned_neighbors_list, delta, beta):
    """
    Perform tree-based clustering using density criteria.
    """
    data = data_utils.get_data(extract_data.get_raw_data_path())
    labels = [0] * len(data)
    densities = np.array(calculate_density(data, pruned_neighbors_list))
    cluster_id = 1
    all_node_maps = {}

    while 0 in labels:
        root_index = np.argmax(densities * (np.array(labels) == 0))
        if labels[root_index] != 0:
            break

        labels[root_index] = cluster_id
        _, node_map = cluster_tree(root_index, data, pruned_neighbors_list,
                                   labels, densities, delta, cluster_id,
                                   beta, None)
        all_node_maps[cluster_id] = node_map

        # Check if the cluster has only one point
        # If yes, then its an anomaly
        if len(node_map) == 1:
            labels[root_index] = -1
            del all_node_maps[cluster_id]
        else:
            cluster_id += 1

    return labels, densities, all_node_maps


def print_tree_densities(all_node_maps):
    """
    Print the densities of nodes in each cluster tree.
    """
    def print_node_and_children(node, node_map, depth=0):
        indent = " " * depth
        print(f"{indent}Density of Node {node.index}: {node.get_density():.3f}")
        for child in node_map[node.index].get_children():
            print_node_and_children(child, node_map, depth + 1)

    for cluster_id, node_map in all_node_maps.items():
        print(f"Cluster ID: {cluster_id}")
        # pylint: disable=W0640
        root_node = node_map[max(node_map, key=lambda x: node_map[x].get_density())]
        print(f"Density of Root Node {root_node.index}: {root_node.get_density():.3f}")
        print_node_and_children(root_node, node_map)
