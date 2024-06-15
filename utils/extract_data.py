"""
This file contains helper functions to return
the raw and ground truth data file paths
"""
import os
from utils import constants


def get_raw_data_path():
    """
    Returns the raw data file path
    """
    return os.path.join('data', 'clustering', 'raw_data', f"{constants.DATASET_NAME}.csv")


def get_ground_truth_data_path():
    """
    Returns the ground truth data file path
    """
    return os.path.join('data', 'clustering', 'ground_truth', f"{constants.DATASET_NAME}_gt.csv")
