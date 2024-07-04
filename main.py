"""
This is the main file which runs the DyTrAno program
"""

import argparse
import warnings
import numpy as np
from utils import pruning_utils, constants, clustering_utils, filtration_utils
from visualizations import interactive_plot, visualize_clusters
from validation import check_tree_structure


def parse_arguments():
    """
    This is used to parse the arguments passed from the CML
    """
    parser = argparse.ArgumentParser()

    # Add arguments below
    parser.add_argument('--numNeigh', type=int, required=True,
                        help='Number of Neighbors Estimate')
    parser.add_argument('--datasetName', type=str, required=True,
                        help='Name of the dataset')
    parser.add_argument('--displayDensity', type=str, default=False,
                        help='Display the densities in tree structure')
    parser.add_argument('--displayPlot', type=str, default="True",
                        help='Display the plot after pruning')
    parser.add_argument('--displayFinalResult', type=str, default="True",
                        help='Display the final plot with clusters')
    # parser.add_argument('--displayStats', type=str, default=True,
    # help='Display inlier-outlier stats at the end')

    arguments = parser.parse_args()

    constants.NUMBER_OF_NEIGHBORS = arguments.numNeigh
    constants.DATASET_NAME = arguments.datasetName
    constants.DISPLAY_DENSITY = arguments.displayDensity
    constants.DISPLAY_PLOT = arguments.displayPlot
    constants.DISPLAY_FINAL_RESULT = arguments.displayFinalResult
    # constants.DISPLAY_DATA_POINT_STATS = arguments.displayStats


def main():
    """
    This is the main function
    """
    warnings.filterwarnings('ignore')
    parse_arguments()
    pruned_neighbors_list = pruning_utils.optimal_neighborhood_selection(
        constants.NUMBER_OF_NEIGHBORS,
        constants.NEIGHBORHOOD_CONTRIBUTION_DIFFERENCE,
        constants.SIGMA)
    if constants.DISPLAY_PLOT == "True":
        interactive_plot.InteractivePlot(pruned_neighbors_list)

    # Run the tree-based clustering algorithm
    # pylint: disable=W0612
    labels, densities, all_node_maps = clustering_utils.tree_based_clustering(
        pruned_neighbors_list,
        constants.DELTA, constants.BETA)

    # Perform filtration of potential anomalies
    filtered_labels = filtration_utils.filter_potential_anomalies(np.array(labels),
                                                                  all_node_maps, densities)

    # Test to see if the tree structure is satisfied
    check_tree_structure.check_tree_structure(all_node_maps)

    # Display the tree densities - for all trees
    if constants.DISPLAY_DENSITY:
        clustering_utils.print_tree_densities(all_node_maps)

    # Visualize the clusters
    if constants.DISPLAY_FINAL_RESULT == "True":
        visualize_clusters.cluster_visualization(filtered_labels, all_node_maps)

    # Save the number of clusters to a file for testing
    with open('cluster_output.txt', 'w', encoding='utf-8') as file:
        num_clusters = len(set(filtered_labels))
        file.write(f'{num_clusters}\n')


if __name__ == "__main__":
    main()
