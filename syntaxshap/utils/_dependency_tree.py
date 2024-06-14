from collections import deque
import pandas as pd
import re
import spacy
import textdescriptives as td

class TreeNode:
    def __init__(self, data):
        """Initialize a TreeNode with data, parent, and children."""
        self.data = data
        self.parent = None
        self.children = []

    def add_child(self, child):
        """Add a child TreeNode to this node."""
        child.parent = self
        self.children.append(child)

def spacy_doc_to_tree(doc):
    """
    Convert a SpaCy Doc object into a tree structure represented by TreeNodes.

    Args:
        doc (spacy.tokens.Doc): SpaCy document object.

    Returns:
        TreeNode: The root node of the constructed tree.
    """
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
    """
    Create a pandas DataFrame from a tree structure.

    Args:
        root (TreeNode): Root node of the tree.

    Returns:
        pd.DataFrame: DataFrame representing the tree structure.
    """
    if not root:
        return pd.DataFrame()

    # Data list to hold information about each node
    data = []
    queue = deque([(root, 0, None)])

    while queue:
        node, level, parent = queue.popleft()

        # Gather node data
        node_data = {
            "word": node.data.text,
            "word_position": node.data.i,
            "level": level,
            "level_weight": 1 / (level + 1),
            "parent": parent.data.text if parent else None
        }
        data.append(node_data)

        # Add children to the queue for processing
        for child in node.children:
            queue.append((child, level + 1, node))

    # Create DataFrame from collected data
    df = pd.DataFrame(data)
    df = df.sort_values(by='level')
    return df

def spacy_dependency_tree(sentence):
    """
    Convert a sentence into a token dependency tree DataFrame using SpaCy.

    Args:
        sentence (str): Input sentence to analyze.

    Returns:
        pd.DataFrame: Token dependency tree DataFrame.
    """
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the input sentence using SpaCy
    text = sentence
    doc = nlp(text + ' MASK')

    # Convert SpaCy dependency tree to a TreeNode object
    tree_root = spacy_doc_to_tree(doc)

    # Generate a DataFrame representing the tree structure
    tree_df = create_dataframe_from_tree(tree_root)

    # Filter out the MASK token from the DataFrame
    tree_df = tree_df[tree_df['word'] != 'MASK']

    return tree_df


def compute_position_mapping(sentence, tokenizer):
    """
    Compute the mapping between words and subtokens in a sentence.
    """
    words = [t.strip() for t in re.findall(r'\b.*?\S.*?(?:\b|$)', sentence)]
    df_words = pd.DataFrame({'word': words, 'word_position': range(len(words))})

    token_ids = tokenizer(sentence)['input_ids']
    df_tokens = pd.DataFrame({'token_id': token_ids, 'token_position': range(len(token_ids))})
    df_tokens['token'] = [tokenizer.decode([token_id]) for token_id in df_tokens['token_id']]

    pos_token_to_word = {}
    k = 0

    for i in range(len(words)):
        word = words[i]
        word_len = 0
        while word_len < len(word):
            decoded_word = tokenizer.decode([token_ids[k]]).replace(' ','')
            if not (decoded_word.startswith('<') and decoded_word.endswith('>')):
                word_len += len(decoded_word)
            pos_token_to_word[k] = i
            k += 1

    return df_words, df_tokens, pos_token_to_word

def get_token_dependency_tree(sentence, tokenizer):
    """
    Get the token dependency tree DataFrame for a given sentence.

    Args:
        sentence (str): Input sentence.
        tokenizer: Tokenizer object.

    Returns:
        pd.DataFrame: Token dependency tree DataFrame.
    """
    
    df_words, df_tokens, pos_token_to_word = compute_position_mapping(sentence, tokenizer)
    # Add 'word' column based on token_id_to_word mapping
    df_tokens['word_position'] = df_tokens['token_position'].map(pos_token_to_word)

    # Merge based on 'word'
    merged_df = pd.merge(df_tokens, df_words, on='word_position', how='inner')

    tree_df = spacy_dependency_tree(sentence)
    dependency_tree = pd.merge(merged_df, tree_df, on=['word', 'word_position'], how='inner')
    return dependency_tree

def compute_dependency_distance(string_list):
    
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('textdescriptives')

    dependency_distances = []
    for i in range(len(string_list)):
        doc = nlp(string_list[i])
        dd = doc._.dependency_distance['dependency_distance_mean']
        dependency_distances.append(dd)

    return dependency_distances