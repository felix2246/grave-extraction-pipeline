import zss
from anytree import Node

from grave_extraction.models import Header


def build_tree_from_lines(lines: list[str]) -> Node:
    """Builds a tree from a list of indented lines using anytree."""
    root = Node("ROOT")
    parent_stack = [(root, -1)]

    for line in lines:
        if not line.strip():
            continue

        level = len(line) - len(line.lstrip(" "))
        node = Node(line.strip(), parent=None)

        while parent_stack and parent_stack[-1][1] >= level:
            parent_stack.pop()

        node.parent = parent_stack[-1][0]
        parent_stack.append((node, level))

    return root


def build_tree_from_headers(headers: list[Header]) -> Node:
    """Builds an anytree.Node tree from a list of Header objects."""
    root = Node("ROOT")
    parent_stack = [(root, -1)]

    for header in headers:
        level = header.get("heading_level")
        if level is None:
            continue

        node = Node(header["header_text"], data=header)

        while parent_stack and parent_stack[-1][1] >= level:
            parent_stack.pop()

        node.parent = parent_stack[-1][0]
        parent_stack.append((node, level))

    return root


# Kaizhong Zhang and Dennis Shasha. Simple fast algorithms for the editing distance between trees and related problems. SIAM Journal of Computing, 18:1245–1262, 1989.
def calculate_tree_distance(tree_a: Node, tree_b: Node) -> float:
    """
    Calculates the tree edit distance between two anytree Nodes using zss.

    Args:
        tree_a: The root node of the first tree.
        tree_b: The root node of the second tree.

    Returns:
        The edit distance between the two trees.
    """
    return zss.simple_distance(
        tree_a,
        tree_b,
        # A function to get the children of a node
        get_children=lambda node: node.children,
        # A function to get the label of a node to compare
        get_label=lambda node: node.name,
    )
