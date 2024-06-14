"""
Contains the function to display the density heatmap
"""

import matplotlib.pyplot as plt
import numpy as np
# pylint: disable=E0401
from utils import data_utils, extract_data


def display_density_heatmap(densities):
    """
    Displays the density heatmap
    """
    densities = np.array(densities)
    data = data_utils.get_data(extract_data.get_raw_data_path())
    log_densities = np.log1p(densities)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=log_densities, cmap='gnuplot', s=10)
    plt.colorbar(scatter, label='Log(Density + 1)')
    plt.title('Heat Map of Point Densities')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
