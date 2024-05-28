import numpy as np
import spacy
import os
import csv
import string
import transformers
from transformers import AutoTokenizer
from utils import create_dataframe_from_tree, spacy_doc_to_tree, arg_parse, fix_random_seed


def get_dependency_dt(text):  # Example usage:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text + " MASK")

    # Convert Spacy dependency tree to a Tree object
    tree_root = spacy_doc_to_tree(doc)

    # Example usage with the Tree structure
    tree_df = create_dataframe_from_tree(tree_root)
    tree_df = tree_df[tree_df["token"] != "MASK"]
    return tree_df


def filter_invalid_inputs(data, tokenizer, keep_prefix, keep_suffix):
    invalid_ids = []
    invalid_inputs = []
    for id, input in enumerate(data):
        if any(p in input for p in string.punctuation):
            invalid_ids.append(id)
            invalid_inputs.append(input)
        else:
            M = len(tokenizer.encode(input))
            M -= keep_prefix
            M -= keep_suffix
            dependency_dt = get_dependency_dt(input)
            max_pos = dependency_dt["position"].max() + 1
            if (M != max_pos) or (len(input.split(' ')) != max_pos):
                invalid_ids.append(id)
                invalid_inputs.append(input)
    invalid_inputs = np.unique(invalid_inputs)
    return invalid_ids, invalid_inputs

def filter(data, tokenizer, keep_prefix, keep_suffix, max_n_tokens):
    invalid_ids = []
    invalid_inputs = []
    nlp = spacy.load("en_core_web_sm")
    for id, input in enumerate(data):
        if any(p in input for p in string.punctuation):
            invalid_ids.append(id)
            invalid_inputs.append(input)
            continue
        ### Check that the number of tokens is less than 15
        M = len(tokenizer.encode(input))
        M -= keep_prefix
        M -= keep_suffix
        if M > max_n_tokens:
            invalid_ids.append(id)
            invalid_inputs.append(input)
            continue
        ### Check that the number of spans is 1
        doc = nlp(input+' MASK')
        sentence_spans = list(doc.sents)
        if len(sentence_spans) > 1:
            invalid_ids.append(id)
            invalid_inputs.append(input)
        ### Compute target and check that it is not empty string
        #target = lmmodel.tokenizer.decode(lmmodel(input)[0], skip_special_tokens=True)
       #if target == '':
            #invalid_ids.append(id)
            #invalid_inputs.append(input)
    invalid_inputs, invalid_ids = np.unique(invalid_inputs), np.unique(invalid_ids)
    return invalid_ids, invalid_inputs



def filter_data(data, tokenizer, args, keep_prefix=0, keep_suffix=0, max_n_tokens=15):
    """
    Filter data on 3 criteria and remove inputs:
        - If the input contains more than 15 tokens
        - If the input contains more than 1 sentence span (according to spaCy dependency tree)
        - If the target is an empty string
    """
    filter_ids_path = os.path.join(args.data_save_dir, f"{args.dataset}/seed_{args.seed}")
    os.makedirs(filter_ids_path, exist_ok=True)
    filename = os.path.join(filter_ids_path, f"{args.dataset}_{args.model_name}_{args.seed}_long_inputs_ids.npy")
    filename_inputs = os.path.join(filter_ids_path, f"{args.dataset}_{args.model_name}_{args.seed}_long_inputs.npy")
    if os.path.exists(filename):
        invalid_ids = np.load(filename, allow_pickle=True)
    else:
        invalid_ids, invalid_inputs = filter(data, tokenizer, keep_prefix, keep_suffix, max_n_tokens)
        np.save(filename, invalid_ids)
        np.save(filename_inputs, invalid_inputs)

    if len(invalid_ids) == 0:
        print(f"No invalid inputs found in {args.dataset}")
        return data, np.arange(len(data))
    filtered_data = np.delete(data, invalid_ids, axis=0)
    filtered_ids = np.delete(np.arange(len(data)), invalid_ids, axis=0)
    print(f"Filtered {args.dataset}: {len(filtered_data)}")
    return filtered_data, filtered_ids
