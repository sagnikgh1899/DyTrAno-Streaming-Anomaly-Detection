import numpy as np
from utils import filtration_utils


mock_data = {
    0: np.array([1, 2, 3]),
    1: np.array([4, 5, 6]),
    2: np.array([7, 8, 9]),
}


def test_empty_cluster():
    cluster_points = []
    result = filtration_utils.calculate_cluster_icd(cluster_points, mock_data)
    assert result == 0, "Expected result to be 0 for an empty cluster"


def test_single_point_cluster():
    cluster_points = [0]
    result = filtration_utils.calculate_cluster_icd(cluster_points, mock_data)
    assert result == 0, "Expected result to be 0 for a cluster with a single point"


def test_two_points_cluster():
    cluster_points = [0, 1]
    result = filtration_utils.calculate_cluster_icd(cluster_points, mock_data)
    expected_distance = calculate_distance(mock_data[0], mock_data[1])
    assert result == expected_distance, f"Expected result to be {expected_distance}"


def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def test_multiple_points_cluster():
    cluster_points = [0, 1, 2]
    result = filtration_utils.calculate_cluster_icd(cluster_points, mock_data)
    num_points = len(mock_data)
    total_distance = 0
    count = 0

    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance_ij = calculate_distance(mock_data[i], mock_data[j])
            total_distance += distance_ij
            count += 1

    expected_distance = total_distance / count

    assert result == expected_distance, f"Expected result to be {expected_distance}"


def test_identical_points():
    cluster_points = [0, 0, 0]  # Testing with duplicate points
    result = filtration_utils.calculate_cluster_icd(cluster_points, mock_data)
    assert result == 0, "Expected result to be 0 for a cluster with identical points"
