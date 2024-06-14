"""
Lists all utility functions related to data extraction
"""

import numpy as np


def read_data(data_path):
    """
    Read data from a CSV file using numpy.

    Parameters:
        data_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing the loaded data and its dimension.
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        data = np.genfromtxt(file, delimiter=',')
    dimension = data.shape[1]
    return data, dimension


def get_data(data_path):
    """
    This function returns the raw data after extraction
    """
    data, _ = read_data(data_path)
    return data


def get_data_dimension(data_path):
    """
    This function returns the ground truth
    data after extraction
    """
    _, dimension = read_data(data_path)
    return dimension
