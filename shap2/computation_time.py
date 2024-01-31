import csv
import os
from metrics import get_scores, save_scores
import time
import explainers
from explainers.other import LimeTextGeneration
import models
import numpy as np
import shap
from datasets import generics_kb, inconsistent_negation, rocstories
import torch
import pickle
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import arg_parse, fix_random_seed
from utils._exceptions import InvalidAlgorithmError
from utils._filter_data import filter_data
from utils.transformers import parse_prefix_suffix_for_tokenizer

#import shap

MIN_TRANSFORMERS_VERSION = "4.25.1"

# check transformers version
assert (
    transformers.__version__ >= MIN_TRANSFORMERS_VERSION
), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."


def main(args):
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### Load the model ####
    if args.model_name == "gpt2":
        model_load = args.model_name
        tokenizer_load = args.model_name
        model = AutoModelForCausalLM.from_pretrained(model_load)
        model.to(device)
        # set text-generation params under task_specific_params
        model.config.task_specific_params["text-generation"] = {
            "do_sample": True,
            "max_new_tokens": 1,
            "temperature": 0.7,
            "top_k": 50,
            #"no_repeat_ngram_size": 2,
        }
    elif args.model_name == "mistral":
        model_load = "mistralai/Mistral-7B-v0.1"
        tokenizer_load = os.path.join(args.model_save_dir, args.model_name) + "/tokenizer"
        if device.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(model_load, load_in_4bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_load, torch_dtype=torch.float16, device_map="auto")
        
    model.config.is_decoder = True


    # model.save_pretrained(f'/cluster/work/zhang/kamara/syntax-shap/models/{args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    lmmodel = models.TextGeneration(model, tokenizer, device=device)
    parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(lmmodel.tokenizer)
    keep_prefix = parsed_tokenizer_dict['keep_prefix']
    keep_suffix = parsed_tokenizer_dict['keep_suffix']

    #### Prepare the data ####
    if args.dataset == "negation":
        data, _ = inconsistent_negation(args.data_save_dir)
    elif args.dataset == "generics":
        data, _ = generics_kb(args.data_save_dir)
    elif args.dataset == "rocstories":
        data, _ = rocstories(args.data_save_dir)
    filtered_data_init = filter_data(data, lmmodel.tokenizer, args, keep_prefix, keep_suffix)
    random_indices = np.random.choice(len(filtered_data_init), 300, replace=False)
    filtered_data = filtered_data_init[random_indices]


    #### Check if the shap values exist ####
    save_dir = os.path.join(args.result_save_dir, 'shap_values')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"shap_values_{args.dataset}_{args.model_name}_{args.algorithm}.pkl"

    #### Explain the model ####
    if args.algorithm == "partition":
        explainer = explainers.PartitionExplainer(lmmodel, lmmodel.tokenizer)
    elif args.algorithm == "lime":
        explainer = LimeTextGeneration(lmmodel, filtered_data)
    elif args.algorithm == "shap":
        explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, algorithm="shap")
    elif args.algorithm == "syntax":
        explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, algorithm="syntax")
    elif args.algorithm == "syntax-w":
        explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, algorithm="syntax-w")
    else:
        raise InvalidAlgorithmError("Unknown algorithm type passed: %s!" % args.algorithm)
    t0 = time.time()
    shap_values = explainer(filtered_data)
    t1 = time.time()
    print(f"Computation time for {len(filtered_data)} samples: {t1-t0}")
    print(f"Average computation time per sample: {(t1-t0)/len(filtered_data)}")

    print("Done!")

if __name__ == "__main__":
    parser, args = arg_parse()

    print('results directory', args.result_save_dir)

    # Get the absolute path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(parent_dir)

    """# Load the config file
    config_path = os.path.join(parent_dir, "configs", "dataset.yaml")
    # read the configuration file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # loop through the config and add any values to the parser as arguments
    for key, value in config[args.dataset_name].items():
        setattr(args, key, value)
    """
    main(args)