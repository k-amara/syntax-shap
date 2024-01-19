import numpy as np
import torch
import torch.nn.functional as F

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
    print(words)
    # First, tokenize
    tokens = []
    #if not tokenize:
    tokens = text.split(' ')#re.split(r'(%s)|$' % r'\W+', text) # this comes from LIME code
    #else:
        #tokens = tokenize(text)
    print(tokens)
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

        mask = torch.zeros(n_words)
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
def fidelity_rmv(str_inputs, shapley_scores, pipeline, k=0.2, token_id=0):
    pipeline.tokenizer.padding_side = 'left'
    input_ids = [pipeline.tokenizer.encode(x) for x in str_inputs]
    max_length = max([len(line) for line in input_ids])

    inputs = pipeline.get_inputs(str_inputs, padding_side='left')

    outputs_orig = pipeline.get_outputs(str_inputs)
    old_predictions = outputs_orig.sequences[:,-1]
    probabilities_orig = torch.nn.functional.softmax(outputs_orig.scores[0], dim=1)
    probs_orig = torch.Tensor([probabilities_orig[enum, item] for enum, item in enumerate(old_predictions)])

    masks = generate_explanatory_masks(str_inputs, shapley_scores, k, token_id)
    new_attention_masks = padleft_mask(masks, max_length = max_length)

    new_inputs = {'input_ids': inputs['input_ids'], 'attention_mask':new_attention_masks}
    outputs = pipeline.inner_model.generate(**new_inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    new_predictions = outputs.sequences[:,-1]
    probabilities = torch.nn.functional.softmax(outputs.scores[0], dim=1)
    probs = torch.Tensor([probabilities[enum, item] for enum, item in enumerate(old_predictions)])

    fid_scores = probs_orig - probs
    return fid_scores

# fidelity_keep corresponds to fidelity- --> 0 - we keep the explanation
def fidelity_keep(str_inputs, shapley_scores, pipeline, k=0.2, token_id=0):
    pipeline.tokenizer.padding_side = 'left'
    input_ids = [pipeline.tokenizer.encode(x) for x in str_inputs]
    max_length = max([len(line) for line in input_ids])

    inputs = pipeline.get_inputs(str_inputs, padding_side='left')

    outputs_orig = pipeline.get_outputs(str_inputs)
    old_predictions = outputs_orig.sequences[:,-1]
    probabilities_orig = torch.nn.functional.softmax(outputs_orig.scores[0], dim=1)
    probs_orig = torch.Tensor([probabilities_orig[enum, item] for enum, item in enumerate(old_predictions)])

    masks = generate_explanatory_masks(str_inputs, shapley_scores, k, token_id)
    new_attention_masks = 1-padleft_mask(masks, max_length = max_length)

    new_inputs = {'input_ids': inputs['input_ids'], 'attention_mask':new_attention_masks}
    outputs = pipeline.inner_model.generate(**new_inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    new_predictions = outputs.sequences[:,-1]
    probabilities = torch.nn.functional.softmax(outputs.scores[0], dim=1)
    probs = torch.Tensor([probabilities[enum, item] for enum, item in enumerate(old_predictions)])

    fid_scores = probs_orig - probs
    return fid_scores



def kl_fidelity_keep(str_inputs, shapley_scores, pipeline, k=0.2, token_id=0):
    pipeline.tokenizer.padding_side = 'left'
    input_ids = [pipeline.tokenizer.encode(x) for x in str_inputs]
    max_length = max([len(line) for line in input_ids])

    inputs = pipeline.get_inputs(str_inputs, padding_side='left')

    outputs_orig = pipeline.get_outputs(str_inputs)
    probabilities_orig = F.softmax(outputs_orig.scores[0], dim=1)

    masks = generate_explanatory_masks(str_inputs, shapley_scores, k, token_id)
    new_attention_masks = padleft_mask(masks, max_length = max_length)

    new_inputs = {'input_ids': inputs['input_ids'], 'attention_mask':new_attention_masks}
    outputs = pipeline.inner_model.generate(**new_inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    probabilities = F.softmax(outputs.scores[0], dim=1)

    # Calculate KL Divergence
    kl_div_mean = F.kl_div(probabilities.log(), probabilities_orig, reduction='mean')
    kl_div_batchmean = F.kl_div(probabilities.log(), probabilities_orig, reduction='batchmean')
    kl_div_scores = []
    for i in range(len(str_inputs)):
        kl_div = F.kl_div(probabilities[i].log(), probabilities_orig[i], reduction='mean')
        kl_div_scores.append(kl_div.item())

    return kl_div_mean.item(), kl_div_batchmean.item(), kl_div_scores
