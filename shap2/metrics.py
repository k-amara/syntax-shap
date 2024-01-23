import numpy as np
import torch
import torch.nn.functional as F
import os
import pickle
from utils import arg_parse, fix_random_seed
from tokenizers import Encoding

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

def generate_explanatory_masks(str_inputs, shapley_scores, k, token_id):
    masks = []
    for i, prompt in enumerate(str_inputs):
        # Extract the top k% words based on the shapley value
        n_words = len(prompt.split())
        shapley_scores_i = shapley_scores[i][:, token_id]
        split_point = int(k * n_words)
        important_indices = (-shapley_scores_i).argsort()[:split_point]

        mask = np.zeros(n_words)
        mask[important_indices] = 1
        masks.append(mask)
    return masks

def padleft_mask(masks, max_length):
    att_masks = torch.zeros((len(masks), max_length))
    for i, sub in enumerate(masks):
        att_masks[i][-len(sub):] = torch.Tensor(sub)
    return att_masks

# Tokenization does not necessarily match the word split
# max_length(tokenizer) >= n_words
# aopc corresponds to fidelity+ --> 1 - we remove the explanation

def run_model(inputs, pipeline):
    inputs = inputs.to(pipeline.device)
    outputs = pipeline.inner_model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    predictions = outputs.sequences[:,-1]
    probabilities = F.softmax(outputs.scores[0], dim=1)
    return predictions, probabilities

def get_scores(str_inputs, shapley_scores, pipeline, k=0.2, token_id=0):
    
    pipeline.tokenizer.padding_side = 'left'
    input_ids = [pipeline.tokenizer.encode(x) for x in str_inputs]
    max_length = max([len(line) for line in input_ids])

    # inputs are token ids (input_ids) and attention masks (attention_mask)
    inputs = pipeline.get_inputs(str_inputs, padding_side='left')
    predictions_orig, probabilities_orig = run_model(inputs, pipeline)
    probs_orig = torch.Tensor([probabilities_orig[enum, item] for enum, item in enumerate(predictions_orig)]).detach().cpu().numpy()

    # initial masks
    masks = generate_explanatory_masks(str_inputs, shapley_scores, k, token_id)

    # mask keep the important words
    new_attention_masks = padleft_mask(masks, max_length = max_length)
    inputs_keep = inputs.copy()
    inputs_keep['attention_mask'] = torch.Tensor(new_attention_masks).to(dtype=torch.long)
    predictions_keep, probabilities_keep = run_model(inputs_keep, pipeline)
    probs_keep = torch.Tensor([probabilities_keep[enum, item] for enum, item in enumerate(predictions_orig)]).detach().cpu().numpy()


    # mask remove the important words
    new_attention_masks = 1-padleft_mask(masks, max_length = max_length)
    inputs_rmv = inputs.copy()
    inputs_rmv['attention_mask'] = torch.Tensor(new_attention_masks).to(dtype=torch.long)
    _, probabilities_rmv = run_model(inputs_rmv, pipeline)
    probs_rmv = torch.Tensor([probabilities_rmv[enum, item] for enum, item in enumerate(predictions_orig)]).detach().cpu().numpy()


    # mask replace the unimportant words with <pad>
    # add <pad> token to the string inputs where masks are 0
    masked_inputs = inputs.copy()
    for i, mask in enumerate(masks):
       masked_inputs['input_ids'][i][-len(mask):][mask == 0] = pipeline.tokenizer.pad_token_id
    _, probabilities_keep_pad = run_model(masked_inputs, pipeline)
    # probs: probabilities of the initial predicted token with the full input sentences
    probs_keep_pad = torch.Tensor([probabilities_keep_pad[enum, item] for enum, item in enumerate(predictions_orig)]).detach().cpu().numpy()

    # Calculate accuracy
    acc = np.sum((predictions_keep == predictions_orig).detach().cpu().numpy())/len(predictions_keep)

    # Calculate fidelity remove important words - AOPC
    fid_rmv = probs_orig - probs_rmv

    # Calculate fidelity keep important words
    fid_keep = probs_orig - probs_keep

    # Calculate KL Divergence
    kl_div_mean = F.kl_div(probabilities_keep.log(), probabilities_orig, reduction='mean')
    kl_div_batchmean = F.kl_div(probabilities_keep.log(), probabilities_orig, reduction='batchmean')
    kl_div_scores = []
    for i in range(len(str_inputs)):
        kl_div = F.kl_div(probabilities_keep[i].log(), probabilities_orig[i], reduction='mean')
        kl_div_scores.append(kl_div.item())

    # Calculate log-odds
    log_odds = np.log(probs_keep_pad + 1e-6) - np.log(probs_orig + 1e-6)

    return {
        "acc": acc,
        "fid_keep": fid_keep,
        "fid_rmv": fid_rmv,
        "kl_div_mean": kl_div_mean.item(),
        "kl_div_batchmean": kl_div_batchmean.item(),
        "kl_div_scores": kl_div_scores,
        "log_odds": log_odds,
    }

def save_scores(args, scores):
    save_dir = os.path.join(args.result_save_dir, 'scores')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"scores_{args.dataset}_{args.model_name}_{args.algorithm}"
    if eval(args.weighted):
        filename += "_weighted"
    filename += ".pkl"
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(scores, f)


if __name__ == "__main__":
    parser, args = arg_parse()
    fix_random_seed(args.seed)

    if args.algorithm.endswith("dtree"):
        weighted = args.weighted
    else:
        weighted = False
    w_str = "_weighted" if weighted else ""
    save_dir = os.path.join(args.result_save_dir, 'shap_values')
    filename = f"shap_values_{args.dataset}_{args.model_name}_{args.algorithm}"
    if eval(args.weighted):
        filename += "_weighted"
    filename += ".pkl"
    with open(os.path.join(args.result_save_dir, filename), "rb") as f:
        shap_values = pickle.load(f)
    explanations = shap_values.values

    #scores = get_scores(args, data, explanations, lmmodel)
    #save_scores(args, scores)