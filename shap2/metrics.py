import numpy as np
import torch
import torch.nn.functional as F
import os
import pickle
from utils import arg_parse, fix_random_seed
from tokenizers import Encoding
import links
from scipy.stats import entropy
from utils import MaskedModel
from maskers import Text
import random

# old_prediction = get_pred_token(text, new_tokens=1)
UNABLE_TO_SWITCH = -1

###### Word deletion / switch point
def get_switch_point_word_deletion(text, words_to_remove_all, old_prediction, pipeline, tokenize):
    """ How many words need to be removed before it is changed? """
    words_to_remove = []
    for i, word in enumerate(words_to_remove_all):
        words_to_remove.append(word)
        if has_prediction_changed(text, words_to_remove, old_prediction, pipeline, tokenize):
            return (i+1)

    return UNABLE_TO_SWITCH


def has_prediction_changed(text, words_to_remove, old_prediction, pipeline, tokenize):
    """ Return True if the prediction has changed after removing the words """
    new_text = remove_word(text, words_to_remove, tokenize)
    return old_prediction != pipeline.get_pred_tokens(new_text)


def remove_word(text, words, tokenize=None):
    """ Remove words from text using the tokenizer provided by the vectorizer"""
    # First, tokenize
    tokens = []
    #if not tokenize:
    tokens = text.split(' ')#re.split(r'(%s)|$' % r'\W+', text) # this comes from LIME code
    #else:
        #tokens = tokenize(text)
    tokens_new = []
    for token in tokens:
        if token not in words and len(token.strip()) > 0:
            tokens_new.append(token.strip())
    return " ".join(tokens_new)


def compute_perturbation_curve(text, words_to_remove_all, old_prediction, pipeline, tokenize, L=10):
    """ Compute AOPC https://arxiv.org/pdf/1509.06321.pdf"""
    # Maximum 10 perturbations
    values = []
    words_to_remove = []
    prob_orig = pipeline.get_probabilities_next_word([text])[old_prediction]

    for i, word in enumerate(words_to_remove_all):
        if i == L:
            break
        words_to_remove.append(word)
        new_text = remove_word(text, words_to_remove, tokenize=tokenize)
        prob = pipeline.get_probabilities_next_word([new_text])[old_prediction]
        values.append((prob_orig - prob).item())
    return np.array(values).sum()/len(values)#(L + 1)

def generate_explanatory_masks(str_inputs, shapley_scores, k, tokenizer, token_id):
    masks = []
    for i, prompt in enumerate(str_inputs):
        # Extract the top k% words based on the shapley value
        n_token = len(tokenizer.tokenize(prompt))
        print(tokenizer.tokenize(prompt))
        print("prompt", prompt)
        print("prompt words", prompt.split())
        shapley_scores_i = shapley_scores[i][:, token_id]
        print("shapley_scores_i", shapley_scores_i)
        if n_token != len(shapley_scores_i):
            print("The scores and the number of tokens are NOT the same")
            masks.append(None)
        else:
            split_point = int(k * n_token)
            important_indices = (-shapley_scores_i).argsort()[:split_point]
            mask = np.zeros(n_token)
            mask[important_indices] = 1
            masks.append(mask)
    return masks

# A boy is not a
# 0 1 1 0 0 
# 1 0 0 1 1
# 1 PAD PAD 1 1


def padleft_mask(masks, max_length):
    att_masks = torch.zeros((len(masks), max_length))
    for i, sub in enumerate(masks):
        att_masks[i][-len(sub):] = torch.Tensor(sub)
    return att_masks

# Tokenization does not necessarily match the word split
# max_length(tokenizer) >= n_words
# aopc corresponds to fidelity+ --> 1 - we remove the explanation


def get_top_k_token_id(probabilities_orig, k):
    top_k_token_id = np.argpartition(probabilities_orig, -k, axis=1)[:, -k:]
    sorted_top_k_token_id = np.array([row[np.argsort(-probabilities_orig[i, row])] for i, row in enumerate(top_k_token_id)])
    return sorted_top_k_token_id

def compute_acc_at_k(probs, probs_orig, k=10):
    top_k_token_id_orig = get_top_k_token_id(probs_orig, k)
    top_k_token_id = get_top_k_token_id(probs, k)
    acc_at_k = np.array([np.intersect1d(top_k_token_id_orig[i], top_k_token_id[i]).size/k for i in range(probs.shape[0])])
    return acc_at_k

def compute_prob_diff_at_k(probs, probs_orig, k=10):
    top_k_token_id_orig = get_top_k_token_id(probs_orig, k)
    top_k_probs_orig = np.array([probs_orig[enum, item] for enum, item in enumerate(top_k_token_id_orig)])
    top_k_probs = np.array([probs[enum, item] for enum, item in enumerate(top_k_token_id_orig)])
    #p = top_k_probs_orig/np.sum(top_k_probs_orig, axis=1, keepdims=True)
    #q = np.nan_to_num(top_k_probs/np.sum(top_k_probs, axis=1, keepdims=True))
    top_k_prob_diff = np.sum(top_k_probs_orig - top_k_probs, axis=1)
    return top_k_prob_diff

def run_model(row_args, mask, pipeline):
    masker = Text(pipeline.tokenizer)
    fm = MaskedModel(pipeline, masker, links.identity, True, *row_args)
    if mask is None:
        mask = np.ones(len(fm), dtype=bool)
    mask = np.array(mask, dtype=bool)
    pred = fm(mask.reshape(1, -1))[0]
    probs = fm.probs
    return pred, probs


def replace_words_randomly(str_input, mask, tokenizer):
    ids_to_replace = np.where(mask == 0)[0].astype(int)
    words = str_input.split(" ")
    assert len(words) == len(mask)
    for i in ids_to_replace:
        L = 0
        while(L != len(words)):
            words[i] = random.choice(list(tokenizer.vocab.keys()))
            new_str_input = " ".join(words)
            L = len(tokenizer.encode(new_str_input, add_special_tokens=False))
    return new_str_input


def get_scores_valid(str_inputs, input_ids, shapley_scores, pipeline, k, token_id=0):

    masks = generate_explanatory_masks(str_inputs, shapley_scores, k, pipeline.tokenizer, token_id)
    # generated masks do not contain prefix and suffix positions!!

    preds_orig, probs_orig = [], []
    preds_keep, probs_keep = [], []
    preds_rmv, probs_rmv = [], []
    preds_keep_rd, probs_keep_rd = [], []
    print("Number of explained instances", len(str_inputs))
    N = len(str_inputs)
    valid_ids = []
    valid_inputs = []
    for i, str_input in enumerate(str_inputs):
        if masks[i] is None:
            print("masks[i] is None")
            N -= 1
            continue
        else:
            row_args = [str_input]
            mask = np.array(masks[i])
            orig = run_model(row_args, None, pipeline)
            preds_orig.append(orig[0])
            probs_orig.append(orig[1])

            keep = run_model(row_args, mask, pipeline)
            preds_keep.append(keep[0])
            probs_keep.append(keep[1])

            rmv = run_model(row_args, 1-mask, pipeline)
            preds_rmv.append(rmv[0])
            probs_rmv.append(rmv[1])

            new_str_input = replace_words_randomly(str_input, mask, pipeline.tokenizer)
            print("new_str_input", new_str_input)
            keep_rd = run_model([new_str_input], None, pipeline)
            preds_keep_rd.append(keep_rd[0])
            probs_keep_rd.append(keep_rd[1])
            valid_ids.append(input_ids[i])
            valid_inputs.append(str_input)

    print("Number of explained instances", N)

    preds_orig, probs_orig = np.concatenate(preds_orig).astype(int), np.concatenate(probs_orig)
    preds_keep, probs_keep = np.concatenate(preds_keep).astype(int), np.concatenate(probs_keep)
    preds_rmv, probs_rmv = np.concatenate(preds_rmv).astype(int), np.concatenate(probs_rmv)
    preds_keep_rd, probs_keep_rd = np.concatenate(preds_keep_rd).astype(int), np.concatenate(probs_keep_rd)


    top_1_probs_orig = torch.Tensor([probs_orig[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()
    top_1_probs_keep = torch.Tensor([probs_keep[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()
    top_1_probs_rmv = torch.Tensor([probs_rmv[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()
    top_1_probs_keep_rd = torch.Tensor([probs_keep_rd[enum, item] for enum, item in enumerate(preds_orig)]).detach().cpu().numpy()

    fid_keep = top_1_probs_orig - top_1_probs_keep
    fid_rmv = top_1_probs_orig - top_1_probs_rmv
    fid_keep_rd = top_1_probs_orig - top_1_probs_keep_rd

    # Calculate log-odds
    log_odds_keep = np.log(top_1_probs_keep + 1e-6) - np.log(top_1_probs_orig + 1e-6)

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
    }


def save_scores(args, scores):
    save_dir = os.path.join(args.result_save_dir, f'scores/{args.model_name}/{args.dataset}/{args.algorithm}/seed_{args.seed}/')
    os.makedirs(save_dir, exist_ok=True)
    filename = "scores_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}_{args.threshold}.pkl"
    print(f"Saving scores to {os.path.join(save_dir, filename)}")
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(scores, f)


if __name__ == "__main__":
    parser, args = arg_parse()
    fix_random_seed(args.seed)

    save_dir = os.path.join(args.result_save_dir, 'shap_values')
    filename = f"shap_values_{args.dataset}_{args.model_name}_{args.algorithm}_{args.k}.pkl"
    with open(os.path.join(args.result_save_dir, filename), "rb") as f:
        shap_values = pickle.load(f)
    explanations = shap_values.values

    #scores = get_scores(args, data, explanations, lmmodel)
    #save_scores(args, scores)