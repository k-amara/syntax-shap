import csv
import os
from metrics import get_scores, save_scores

import explainers
from explainers.other import LimeTextGeneration
import models
import numpy as np
import shap
import torch
import pickle
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import arg_parse, fix_random_seed
from utils._exceptions import InvalidAlgorithmError

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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load, is_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    lmmodel = models.TextGeneration(model, tokenizer, device=device)

    #### Prepare the data ####
    if args.dataset == "negation":
        data_file = "Inconsistent-Dataset-Negation.tsv"
    data_path = os.path.join(args.data_save_dir, data_file)
    tsv_file = open(data_path)
    read_tsv = list(csv.reader(tsv_file, delimiter="\t"))
    data = []
    for row in read_tsv:
        data.append(row[1][:-8])
    data = np.array(data)
    print(f"Inconsistent-Dataset-Negation.tsv: {len(data)}")

    #### Filter invalid data ####
    # Tokenization might split words into multiple tokens, which is not supported by the current implementation
    filter_ids_path = os.path.join(args.result_save_dir, "invalid_data/invalid_ids.npy")
    if os.path.exists(filter_ids_path):
        invalid_ids = np.load(filter_ids_path, allow_pickle=True)
    else:
        invalid_ids = []
    filtered_data = np.delete(data, invalid_ids, axis=0)
    # filtered_explanations = np.delete(explanations, invalid_ids, axis=0)
    # assert len(filtered_data) == len(filtered_explanations)
    print(f"Filtered Inconsistent-Dataset-Negation.tsv: {len(filtered_data)}")


    #### Check if the shap values exist ####
    save_dir = os.path.join(args.result_save_dir, 'shap_values')
    os.makedirs(save_dir, exist_ok=True)
    filename = f"shap_values_{args.dataset}_{args.model_name}_{args.algorithm}.pkl"

    #### Explain the model ####
    if args.algorithm == "partition_init":
        explainer = shap.explainers.PartitionExplainer(lmmodel, lmmodel.tokenizer)
    elif args.algorithm == "partition":
        explainer = explainers.PartitionExplainer(lmmodel, lmmodel.tokenizer)
    elif args.algorithm == "lime":
        explainer = LimeTextGeneration(lmmodel, filtered_data)
    elif args.algorithm == "exact":
        explainer = explainers.DependencyExplainer(lmmodel, lmmodel.tokenizer, algorithm="exact", weighted=False)
    elif args.algorithm == "dtree":
        explainer = explainers.DependencyExplainer(lmmodel, lmmodel.tokenizer, algorithm="dtree", weighted=False)
    elif args.algorithm == "dtreew":
        explainer = explainers.DependencyExplainer(lmmodel, lmmodel.tokenizer, algorithm="dtree", weighted=True)
    elif args.algorithm == "r-dtree":
        explainer = explainers.DependencyExplainer(lmmodel, lmmodel.tokenizer, algorithm="r-dtree", weighted=False)
    else:
        raise InvalidAlgorithmError("Unknown dependency tree algorithm type passed: %s!" % args.algorithm)
    shap_values = explainer(filtered_data[:5])

    #### Save the shap values ####
    if args.algorithm == "lime":
        filtered_explanations = explainer._s
    else: 
        filtered_explanations = shap_values.values

    print("Done!")
    
    #### Evaluate the explanations ####
    scores = get_scores(filtered_data[:5], filtered_explanations, lmmodel, args.threshold)
    print("scores", scores)
    save_scores(args, scores)


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
