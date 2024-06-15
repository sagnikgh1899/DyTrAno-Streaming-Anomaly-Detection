"""
Contains helper functions to test/debug using plotting
"""

import matplotlib.pyplot as plt


def plot_nearest_inlier_and_potential_anomaly_for_filtration(data, anomaly_index,
                                                             nearest_inlier_index):
    """
    Plots the nearest inlier and potential anomaly
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], s=10, c='blue', label='Data Points')
    plt.scatter(data[anomaly_index, 0], data[anomaly_index, 1], s=50, color='magenta', marker='o',
                label='Potential Anomaly')
    plt.scatter(data[nearest_inlier_index, 0], data[nearest_inlier_index, 1], s=50,
                color='black', marker='x', label='Nearest Inlier')
    plt.title('Anomaly and Nearest Inlier Debugging')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
