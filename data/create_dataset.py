"""
Not important: Just creates a dummy dataset if required
"""

import numpy as np


def create_filled_circle_data(center, radius, num_points):
    """
    Creates a dummy dataset
    """
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, num_points))
    x_val = center[0] + radii * np.cos(angles)
    y_val = center[1] + radii * np.sin(angles)
    return np.vstack((x_val, y_val)).T


NUM_POINTS_PER_CIRCLE = 250
centers = [(-5, -5), (5, -5), (-5, 5), (5, 5)]
RADIUS = 3
data = np.vstack([create_filled_circle_data(center, RADIUS,
                                            NUM_POINTS_PER_CIRCLE) for center in centers])
