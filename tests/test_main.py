import subprocess
import os
import pytest


@pytest.mark.parametrize("num_neigh, dataset_name, expected_clusters", [
    (40, 'Corners', 4),
    (40, 'Moon', 2),
    (40, 'Half_kernel', 2),
    (20, 'Jain', 2),
    (15, 'Flame', 1),
])
@pytest.mark.flaky(reruns=5)
def test_dataset_clustering(num_neigh, dataset_name, expected_clusters):
    result = subprocess.run(['python', 'main.py', '--numNeigh', str(num_neigh),
                             '--datasetName', dataset_name, '--displayPlot',
                             'False', '--displayFinalResult', 'False'])
    with open('cluster_output.txt', 'r') as f:
        num_clusters = int(f.read().strip())
    assert num_clusters >= expected_clusters


def teardown_function(function):
    try:
        os.remove('cluster_output.txt')
    except FileNotFoundError:
        pass
