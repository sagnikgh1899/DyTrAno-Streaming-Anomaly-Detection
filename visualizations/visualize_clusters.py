"""
Contains helper functions to plot the clusters
"""

import matplotlib.pyplot as plt
import numpy as np
# pylint: disable=E0401
from utils import data_utils, extract_data


def cluster_visualization(labels, all_node_maps):
    """
    Plots the clusters
    """
    data = data_utils.get_data(extract_data.get_raw_data_path())
    unique_labels = np.unique(labels)

    plt.figure(figsize=(12, 10))
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
        '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#6b6ecf',
        '#9c9ede', '#d6616b', '#ce6dbd',
        '#de9ed6', '#3182bd', '#6baed6', '#9ecae1', '#e6550d', '#fd8d3c', '#fdae6b',
        '#31a354', '#74c476', '#a1d99b'
    ]

    for cluster_id in unique_labels:
        if cluster_id == 0:
            continue
        cluster_points = data[np.array(labels) == cluster_id]
        if cluster_id == -1:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        color='black', marker='*', edgecolor='black', s=100,
                        label='Anomaly')
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        color=colors[cluster_id],
                        label=f'Cluster {cluster_id}', s=10)

            # pylint: disable=W0640
            root_node = max(all_node_maps[cluster_id],
                            key=lambda x: all_node_maps[cluster_id][x].get_density())
            root_coords = data[root_node]
            plt.scatter(root_coords[0], root_coords[1], color='red', edgecolor='black',
                        s=100, marker='o',
                        label=f'Root {cluster_id}')

    plt.title('Tree-Based Clustering with Anomalies')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    plt.show()
