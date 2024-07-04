import pytest

from utils import constants, pruning_utils, data_utils, extract_data, clustering_utils
from validation import check_tree_structure


@pytest.fixture
def pruned_neighbor_list():
    return pruning_utils.optimal_neighborhood_selection(
        constants.NUMBER_OF_NEIGHBORS,
        constants.NEIGHBORHOOD_CONTRIBUTION_DIFFERENCE,
        constants.SIGMA)


@pytest.fixture
def get_data():
    return data_utils.get_data(extract_data.get_raw_data_path())


@pytest.fixture
def tree_based_clustering(pruned_neighbor_list):
    return clustering_utils.tree_based_clustering(
        pruned_neighbor_list, constants.DELTA, constants.BETA)


def test_pruning(pruned_neighbor_list, get_data):
    assert len(pruned_neighbor_list) == len(get_data)
    for neighbors in pruned_neighbor_list:
        assert 1 <= len(neighbors) <= constants.NUMBER_OF_NEIGHBORS


def test_tree_based_clustering(tree_based_clustering, get_data):
    labels, densities, all_node_maps = tree_based_clustering
    assert all(label != 0 for label in labels)
    assert len(labels) == len(get_data)
    assert len(densities) == len(get_data)


def test_tree_density_structure(tree_based_clustering):
    _, _, all_node_maps = tree_based_clustering
    try:
        check_tree_structure.check_tree_structure(all_node_maps)
    except Exception as e:
        pytest.fail(f"test_tree_structure.test_tree_structure raised an exception: {str(e)}")
