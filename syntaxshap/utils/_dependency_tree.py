from collections import deque

import pandas as pd


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

def spacy_doc_to_tree(doc):
    # Create a TreeNode for each token in the doc
    nodes = [TreeNode(token) for token in doc]

    # Create the tree structure by connecting parent-child relationships
    for token, node in zip(doc, nodes):
        if token.head.i == token.i:  # Skip root node (head is itself)
            continue
        parent_node = nodes[token.head.i]
        parent_node.add_child(node)

    # Find and return the root node
    root = next(node for node in nodes if node.parent is None)
    return root



def create_dataframe_from_tree(root):
    if not root:
        return pd.DataFrame()

    # Data list to hold information about each node
    data = []
    queue = deque([(root, 0, None)])

    while queue:
        node, level, parent = queue.popleft()

        # If the node has a parent, find the positions of its siblings
        if parent:
            sibling_positions = [child.data.i for child in parent.children]
        else:
            sibling_positions = []

        node_data = {
            "token": node.data.text,
            "position": node.data.i,
            "level": level,
            "level_weight": 1 / (level + 1),
            "parent": parent.data.text if parent else None,
            "sibling_positions": sibling_positions
        }
        data.append(node_data)

        for child in node.children:
            queue.append((child, level + 1, node))

    # Create DataFrame from the collected data
    df = pd.DataFrame(data)
    df = df.sort_values(by='level')
    return df



# Example usage:
# nlp = spacy.load("en_core_web_sm")
# text = "This is an example sentence."
# doc = nlp(text)

# Convert Spacy dependency tree to a Tree object
# tree_root = spacy_doc_to_tree(doc)

# Example usage with the Tree structure
# tree_df = create_dataframe_from_tree(tree_root)
# print(tree_df)
