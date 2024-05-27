import numpy as np
import torch
import random
import os
import pickle
from typing import List, Optional, Tuple, Union

import links
from utils import MaskedModel
from maskers import Text

# Constants
UNABLE_TO_SWITCH = -1


# Explanatory Masks Generation
def generate_explanatory_masks(
    str_inputs: List[str], 
    shapley_scores, 
    tokens,
    k: float, 
    tokenizer, 
    next_token_id: int
) -> List[Optional[np.ndarray]]:
    """
    Generate explanatory masks based on SHAP values.

    Args:
        str_inputs (List[str]): List of input strings.
        shapley_scores: SHAP values.
        k (float): Percentage of important indices.
        tokenizer: Tokenizer object.
        next_token_id (int): Token ID.

    Returns:
        List: Explanatory masks.
    """
    masks = []
    for i, prompt in enumerate(str_inputs):
        print('prompt', prompt)
        n_token = len(tokenizer.tokenize(prompt))
        shapley_scores_i = shapley_scores[i][:, next_token_id]
        print('shapley_scores_i', shapley_scores_i)
        print('n_token', n_token)
        assert n_token == len(shapley_scores[i])
        if n_token != len(shapley_scores_i):
            masks.append(None)
        else:
            split_point = int(k * n_token)
            important_indices = (-shapley_scores_i).argsort()[:split_point]
            mask = np.zeros(n_token)
            mask[important_indices] = 1
            masks.append(mask)
    return masks

# Padding Left Mask
def padleft_mask(
    masks: List[Optional[np.ndarray]], 
    max_length: int
) -> torch.Tensor:
    """
    Pad masks on the left to match max length.

    Args:
        masks (List[Optional[np.ndarray]]): List of masks.
        max_length (int): Maximum length.

    Returns:
        torch.Tensor: Padded masks.
    """
    att_masks = torch.zeros((len(masks), max_length))
    for i, sub in enumerate(masks):
        att_masks[i][-len(sub):] = torch.Tensor(sub)
    return att_masks

def get_top_k_token_id(probabilities_orig: np.ndarray, k: int) -> np.ndarray:
    """
    Get the indices of the top k tokens based on their probabilities.

    Args:
        probabilities_orig (np.ndarray): Original probabilities.
        k (int): Number of top tokens to select.

    Returns:
        np.ndarray: Array containing the indices of the top k tokens.
    """
    top_k_token_id = np.argpartition(probabilities_orig, -k, axis=1)[:, -k:]
    sorted_top_k_token_id = np.array([row[np.argsort(-probabilities_orig[i, row])] for i, row in enumerate(top_k_token_id)])
    return sorted_top_k_token_id

# Compute Accuracy at k
def compute_acc_at_k(
    probs: np.ndarray, 
    probs_orig: np.ndarray, 
    k: int = 10
) -> np.ndarray:
    """
    Compute accuracy at k.

    Args:
        probs (np.ndarray): Predicted probabilities.
        probs_orig (np.ndarray): Original probabilities.
        k (int, optional): Number of top tokens.

    Returns:
        np.array: Accuracy at k.
    """
    top_k_token_id_orig = get_top_k_token_id(probs_orig, k)
    top_k_token_id = get_top_k_token_id(probs, k)
    acc_at_k = np.array([np.intersect1d(top_k_token_id_orig[i], top_k_token_id[i]).size / k for i in range(probs.shape[0])])
    return acc_at_k

# Compute Probability Difference at k
def compute_prob_diff_at_k(
    probs: np.ndarray, 
    probs_orig: np.ndarray, 
    k: int = 10
) -> np.ndarray:
    """
    Compute probability difference at k.

    Args:
        probs (np.ndarray): Predicted probabilities.
        probs_orig (np.ndarray): Original probabilities.
        k (int, optional): Number of top tokens.

    Returns:
        np.ndarray: Array containing the probability difference at k.
    """
    top_k_token_id_orig = get_top_k_token_id(probs_orig, k)
    top_k_probs_orig = np.array([probs_orig[enum, item] for enum, item in enumerate(top_k_token_id_orig)])
    top_k_probs = np.array([probs[enum, item] for enum, item in enumerate(top_k_token_id_orig)])
    top_k_prob_diff = np.sum(top_k_probs_orig - top_k_probs, axis=1)
    return top_k_prob_diff

# Run Model
def run_model(
    row_args: List[Union[str, int, float]], 
    mask: Optional[np.ndarray], 
    pipeline
) -> Tuple[int, np.ndarray]:
    """
    Run the model with optional masking.

    Args:
        row_args (List[Union[str, int, float]]): Model arguments.
        mask (Optional[np.ndarray]): Mask for masking.
        pipeline: Model pipeline.

    Returns:
        Tuple[int, np.ndarray]: Prediction and probabilities.
    """
    masker = Text(pipeline.tokenizer)
    fm = MaskedModel(pipeline, masker, links.identity, True, *row_args)
    if mask is None:
        mask = np.ones(len(fm), dtype=bool)
    mask = np.array(mask, dtype=bool)
    pred = fm(mask.reshape(1, -1))[0]
    probs = fm.probs
    return pred, probs
    

# Function to replace words randomly based on a mask
def replace_token_randomly(
    str_input: str, 
    mask: np.ndarray, 
    tokenizer
) -> str:
    """
    Replaces words randomly based on a mask.

    Args:
        str_input (str): The original input string.
        mask (np.ndarray): The mask indicating which words to replace.
        tokenizer: Tokenizer object.

    Returns:
        str: The input string with token ids replaced.
    """
    # Get indices of token ids to replace
    ids_to_replace = np.where(mask == 0)[0].astype(int)
    token_ids = tokenizer.encode(str_input, add_special_tokens=False)
    assert  len(token_ids) == len(mask)

    # Replace token ids
    for i in ids_to_replace:
        L = 0
        while(L != len(token_ids)):
            # Replace with a random word from tokenizer vocabulary
            token_ids[i] = random.choice(list(tokenizer.vocab.values()))
            new_str_input = tokenizer.decode(token_ids)
            # Check if the new string length is within token length limits
            L = len(tokenizer.encode(new_str_input, add_special_tokens=False))
    return new_str_input

# Function to calculate scores for the explanations
def get_scores(
    results,
    pipeline, 
    k: float,
    token_id: int = 0
) -> dict:
    """
    Calculates scores for the explanations.

    Args:
        str_inputs (List[str]): List of input strings.
        input_ids (List[int]): List of input IDs.
        shapley_scores: Shapley scores.
        pipeline: Pipeline object.
        k (float): The percentage of important indices.
        token_id (int, optional): Token ID. Defaults to 0.

    Returns:
        dict: Dictionary containing computed scores.
    """
    # Generate explanatory masks
    masks = generate_explanatory_masks(results["input"], results["explanation"], results["tokens"], k, pipeline.tokenizer, token_id)

    # Initialize lists to store predictions and probabilities
    preds_orig, probs_orig = [], []
    preds_keep, probs_keep = [], []
    preds_rmv, probs_rmv = [], []
    preds_keep_rd, probs_keep_rd = [], []

    # Initialize lists to store valid input ids and inputs
    valid_ids = []
    valid_inputs = []
    valid_tokens = []
    valid_token_ids = []
    
    N = len(results["input"])
    print("Number of explained instances", N)

    # Iterate through all inputs
    for i, str_input in enumerate(results["input"]):
        # Skip if mask is None
        if masks[i] is None:
            print("masks[i] is None for input", str_input, " - skipping...")
            N -= 1
            continue
        else:
            row_args = [str_input]
            mask = np.array(masks[i])

            # Get predictions and probabilities for original, keep, remove, and keep with random replacements
            orig = run_model(row_args, None, pipeline)
            preds_orig.append(orig[0])
            probs_orig.append(orig[1])

            keep = run_model(row_args, mask, pipeline)
            preds_keep.append(keep[0])
            probs_keep.append(keep[1])

            rmv = run_model(row_args, 1-mask, pipeline)
            preds_rmv.append(rmv[0])
            probs_rmv.append(rmv[1])

            new_str_input = replace_token_randomly(str_input, mask, pipeline.tokenizer)
            keep_rd = run_model([new_str_input], None, pipeline)
            preds_keep_rd.append(keep_rd[0])
            probs_keep_rd.append(keep_rd[1])

            valid_ids.append(results["input_id"][i])
            valid_inputs.append(str_input)
            valid_tokens.append(results["tokens"][i])
            valid_token_ids.append(results["token_ids"][i])

    print("Number of explained instances after removing None masks", N)

    # Concatenate predictions and probabilities lists
    preds_orig, probs_orig = np.concatenate(preds_orig).astype(int), np.concatenate(probs_orig)
    preds_keep, probs_keep = np.concatenate(preds_keep).astype(int), np.concatenate(probs_keep)
    preds_rmv, probs_rmv = np.concatenate(preds_rmv).astype(int), np.concatenate(probs_rmv)
    preds_keep_rd, probs_keep_rd = np.concatenate(preds_keep_rd).astype(int), np.concatenate(probs_keep_rd)

    # Calculate fidelity and log-odds
    top_1_probs_orig = torch.Tensor([probs_orig[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()
    top_1_probs_keep = torch.Tensor([probs_keep[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()
    top_1_probs_rmv = torch.Tensor([probs_rmv[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()
    top_1_probs_keep_rd = torch.Tensor([probs_keep_rd[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()

    fid_keep = top_1_probs_orig - top_1_probs_keep
    fid_rmv = top_1_probs_orig - top_1_probs_rmv
    fid_keep_rd = top_1_probs_orig - top_1_probs_keep_rd

    log_odds_keep = np.log(top_1_probs_keep + 1e-6) - np.log(top_1_probs_orig + 1e-6)

    # Calculate accuracy at k and probability difference at k
    acc_at_k = compute_acc_at_k(probs_keep, probs_orig, k=10)
    prob_diff_at_k = compute_prob_diff_at_k(probs_keep, probs_orig, k=10)

    return {
        "fid_keep_rd": fid_keep_rd,
        "fid_keep": fid_keep,
        "fid_rmv": fid_rmv,
        "log_odds_keep": log_odds_keep,
        "acc_at_k": acc_at_k,
        "prob_diff_at_k": prob_diff_at_k,
        "input_id": valid_ids,
        "input": valid_inputs,
        "tokens": valid_tokens,
        "token_ids": valid_token_ids,
    }

# Function to save scores
def save_scores(args, scores):
    """
    Saves the computed scores to a file.

    Args:
        args: Arguments object.
        scores: Dictionary containing computed scores.
    """
    save_dir = os.path.join(args.result_save_dir, f'scores/{args.model_name}/{args.dataset}/{args.algorithm}/seed_{args.seed}/')
    os.makedirs(save_dir, exist_ok=True)
    filename = "scores_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}_{args.threshold}.pkl"
    print(f"Saving scores to {os.path.join(save_dir, filename)}")
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(scores, f)

    