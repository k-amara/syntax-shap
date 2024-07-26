# Import necessary libraries and modules
import os
import torch
import pickle as pkl
import pandas as pd

# Import custom modules and functions
from metrics import get_scores, save_scores
import models
from explain import compute_explanations, save_path
from model import load_model
from datasets import load_data
from utils import arg_parse, fix_random_seed
from utils.transformers import parse_prefix_suffix_for_tokenizer


def main(args):
    # Set random seed
    fix_random_seed(args.seed)
    
    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model(device, args)
    
    # Initialize TextGeneration model
    lmmodel = models.TextGeneration(model, tokenizer, device=device)
    
    # Parse tokenizer prefix and suffix
    parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(lmmodel.tokenizer)
    args.keep_prefix = parsed_tokenizer_dict['keep_prefix']
    args.keep_suffix = parsed_tokenizer_dict['keep_suffix']

    # Print special tokens information
    args.special_tokens = tokenizer.special_tokens_map
    num_special_tokens = len(args.special_tokens)
    print(f"Number of special tokens: {num_special_tokens}")
    print("Special tokens map:")
    for token_name, token_value in args.special_tokens.items():
        print(f"{token_name}: {token_value}")

    # Prepare the data
    data, data_ids, all_data = load_data(tokenizer, args)
    print("Length of data:", len(data))

    # Check if explanations exist
    save_explanation_path = save_path(args)
    if os.path.exists(save_explanation_path):
        print("Loading explanations...")
        with open(save_explanation_path, "rb") as f:
            results = pkl.load(f)
    else:
        results = compute_explanations(lmmodel, model, data, data_ids, all_data, args)
        with open(save_explanation_path, "wb") as f:
            pkl.dump(results, f)

    print("Done!")

    # Evaluate the explanations
    results = pd.DataFrame(results)
    print(f"Calculating scores for {len(results['input_id'])} explained instances...")
    scores = get_scores(results, lmmodel, args.threshold)
    print("Scores:", scores)
    save_scores(args, scores)
    
def test_method(args):
    # Set random seed
    fix_random_seed(args.seed)
    
    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model(device, args)
    
    # Initialize TextGeneration model
    lmmodel = models.TextGeneration(model, tokenizer, device=device)
    
    # Parse tokenizer prefix and suffix
    parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(lmmodel.tokenizer)
    args.keep_prefix = parsed_tokenizer_dict['keep_prefix']
    args.keep_suffix = parsed_tokenizer_dict['keep_suffix']

    # Print special tokens information
    args.special_tokens = tokenizer.special_tokens_map
    num_special_tokens = len(args.special_tokens)
    print(f"Number of special tokens: {num_special_tokens}")
    print("Special tokens map:")
    for token_name, token_value in args.special_tokens.items():
        print(f"{token_name}: {token_value}")

    # Prepare the data
    data, data_ids, all_data = load_data(tokenizer, args)
    print("Length of data:", len(data))

    results = compute_explanations(lmmodel, model, data, data_ids, all_data, args)
    print("Done!")



if __name__ == "__main__":
    parser, args = arg_parse()
    print('Results directory:', args.result_save_dir)

    # Get the absolute path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Parent directory:", parent_dir)

    # Uncomment and use if you have a config file to load additional arguments
    """
    # Load the config file
    config_path = os.path.join(parent_dir, "configs", "dataset.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add config values to args
    for key, value in config[args.dataset_name].items():
        setattr(args, key, value)
    """

    # Execute main function
    # main(args)
    test_method(args)
