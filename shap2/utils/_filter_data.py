import numpy as np
import spacy
import os
import csv
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
        M = len(tokenizer.encode(input))
        M -= keep_prefix
        M -= keep_suffix
        dependency_dt = get_dependency_dt(input)
        max_pos = dependency_dt["position"].max() + 1
        if M != max_pos:
            invalid_ids.append(id)
            invalid_inputs.append(input)
    invalid_inputs = np.unique(invalid_inputs)
    return invalid_ids, invalid_inputs

def filter_data(dataset, data, tokenizer, data_save_dir, keep_prefix=0, keep_suffix=0):
    #### Filter invalid data ####
    # Tokenization might split words into multiple tokens, which is not supported by the current implementation
    filter_ids_path = os.path.join(data_save_dir, f"{dataset}")
    os.makedirs(filter_ids_path, exist_ok=True)
    filename = os.path.join(filter_ids_path, f"{dataset}_invalid_inputs.npy")
    if os.path.exists(filename):
        invalid_ids = np.load(filename, allow_pickle=True)
    else:
        invalid_ids, _ = filter_invalid_inputs(data, tokenizer, keep_prefix, keep_suffix)
        np.save(filename, invalid_ids)
    filtered_data = np.delete(data, invalid_ids, axis=0)
    print(f"Filtered {dataset}: {len(filtered_data)}")
    return filtered_data


if __name__ == "__main__":
    parser, args = arg_parse()

    if args.model_name == "gpt2":
        tokenizer_load = args.model_name
    elif args.model_name == "mistral":
        tokenizer_load = os.path.join(args.model_save_dir, args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load)
    tokenizer.pad_token = tokenizer.eos_token

    #### Prepare the data ####
    tsv_file = open("./data/Inconsistent-Dataset-Negation.tsv")
    read_tsv = list(csv.reader(tsv_file, delimiter="\t"))
    data = []
    for row in read_tsv:
        data.append(row[1][:-8])
    data = np.array(data)
    print(f"Inconsistent-Dataset-Negation.tsv: {len(data)}")

    filter_ids_path = os.path.join(args.result_save_dir, "data/invalid_ids.npy")
    filter_inputs_path = os.path.join(args.result_save_dir, "data/invalid_inputs.npy")
    if os.path.exists(filter_ids_path):
        invalid_ids = np.load(filter_ids_path, allow_pickle=True)
    else:
        invalid_ids, invalid_inputs = filter_invalid_inputs(data, tokenizer)
        np.save(filter_ids_path, invalid_ids)
        np.save(filter_inputs_path, invalid_inputs)