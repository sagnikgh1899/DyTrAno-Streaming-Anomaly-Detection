"""
Validate tree structures by checking parent-child density relationships.
"""

from tqdm import tqdm


def check_tree_structure(all_node_maps):
    """Test the validity of tree structures in all_node_maps."""

    def is_valid_node(node):
        """Check if a node has valid parent-child density relationships."""
        if node_map[node.index].parent is not None and node_map[node.index].get_density() > \
                node_map[node.index].get_parent().get_density():
            print(f"{node.index} has higher density ({node_map[node.index].get_density()}) "
                  f"than its parent "
                  f"{node_map[node.index].get_parent_index()} "
                  f"({node_map[node.index].get_parent().get_density()})")
            return False
        for child in node_map[node.index].get_children():
            if node_map[node.index].get_density() < node_map[child.index].get_density():
                print(f"{node.index} has lower density ({node_map[node.index].get_density()}) "
                      f"than its child {node.index} ({node_map[child.index].get_density()})")
                return False
        return True

    print("\nStarting tree structure validation...")
    for _, node_map in tqdm(all_node_maps.items()):
        for index, node in node_map.items():
            assert is_valid_node(node), f"Node {index} does not " \
                                        f"have a valid parent-child relationship."
    print("Tree structure validation passed successfully.")
