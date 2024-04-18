from collections import deque
import pandas as pd
import re
import spacy

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
            #"position": node.data.i,
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


def get_word_subtoken_dict(sentence, tokenizer):
    """
    Create a dictionary mapping words to their subtoken IDs.

    Args:
        sentence (str): Input sentence.
        tokenizer: Tokenizer object capable of encoding the sentence.

    Returns:
        dict: Dictionary mapping words to lists of subtoken IDs.
    """
    word_subtoken_dict = {}
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    words = [t.strip() for t in re.findall(r'\b.*?\S.*?(?:\b|$)', sentence)]
    k = 0

    for word in words:
        word_len = 0
        list_subtokens = []

        # Iterate until the word is fully tokenized
        while word_len < len(word):
            list_subtokens.append(token_ids[k])
            decoded_word = tokenizer.decode([token_ids[k]]).replace(' ', '')
            word_len += len(decoded_word)
            k += 1

        word_subtoken_dict[word] = list_subtokens

    return word_subtoken_dict

def token_table(sentence, tokenizer):
    """
    Create a DataFrame representing token IDs and positions in a sentence.

    Args:
        sentence (str): Input sentence.
        tokenizer: Tokenizer object capable of encoding the sentence.

    Returns:
        pd.DataFrame: DataFrame with columns for token ID, position, and token text.
    """
    encoding = tokenizer(sentence)
    num_tokens = len(encoding['input_ids'])
    positions = range(num_tokens)
    df_tokens = pd.DataFrame({'token_id': encoding['input_ids'], 'position': positions})
    df_tokens['token'] = [tokenizer.decode([token_id]) for token_id in df_tokens['token_id']]
    return df_tokens

def token_dependency_tree(tree_df, word_subtoken_dict, tokenizer, df_tokens):
    """
    Create a token dependency tree DataFrame.

    Args:
        tree_df (pd.DataFrame): DataFrame representing a tree structure.
        word_subtoken_dict (dict): Dictionary mapping words to subtoken IDs.
        tokenizer: Tokenizer object.
        df_tokens (pd.DataFrame): DataFrame representing token IDs and positions.

    Returns:
        pd.DataFrame: Token dependency tree DataFrame.
    """
    new_rows = []
    for index, row in tree_df.iterrows():
        word = row['word']
        token_ids = word_subtoken_dict[word]  # Use your tokenizer here
        for token_id in token_ids:
            new_row = row.copy()
            new_row['token_id'] = token_id
            new_row['token'] = tokenizer.decode([token_id])
            if row['parent'] is not None:
                new_row['parent_ids'] = word_subtoken_dict[row['parent']]
            new_rows.append(new_row)
    token_dt = pd.DataFrame(new_rows)
    token_dt = pd.merge(token_dt, df_tokens, on=['token_id', 'token'], how='left')
    return token_dt

def get_token_dependency_tree(sentence, tokenizer):
    """
    Get the token dependency tree DataFrame for a given sentence.

    Args:
        sentence (str): Input sentence.
        tokenizer: Tokenizer object.

    Returns:
        pd.DataFrame: Token dependency tree DataFrame.
    """
    df_tokens = token_table(sentence, tokenizer)
    tree_df = spacy_dependency_tree(sentence)
    word_subtoken_dict = get_word_subtoken_dict(sentence, tokenizer)
    token_dt = token_dependency_tree(tree_df, word_subtoken_dict, tokenizer, df_tokens)
    return token_dt 
