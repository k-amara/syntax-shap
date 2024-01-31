import csv
import os
from metrics import get_scores, save_scores

import explainers
from explainers.other import LimeTextGeneration
import models
import numpy as np
import shap
from datasets import generics_kb, generics_kb_large, inconsistent_negation, rocstories
import torch
import pickle
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from utils import arg_parse, fix_random_seed
from utils._exceptions import InvalidAlgorithmError
from utils._filter_data import filter_data
from utils.transformers import parse_prefix_suffix_for_tokenizer
from tqdm import tqdm


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
    elif args.dataset == "generics-0":
        data, _ = generics_kb(args.data_save_dir)
    elif args.dataset == "generics":
        data, _ = generics_kb_large(args.data_save_dir)
    elif args.dataset == "rocstories":
        data, _ = rocstories(args.data_save_dir)
    filtered_data = filter_data(data, lmmodel.tokenizer, args, keep_prefix, keep_suffix)
    if args.num_batch is not None:
        assert args.num_batch * args.batch_size < len(filtered_data), "Batch number is too large!"
        n_min = args.batch_size * args.num_batch
        n_max = args.batch_size * (args.num_batch + 1) if args.num_batch < len(filtered_data) // args.batch_size else len(filtered_data)
        print(f"Batch number {args.num_batch} of size {args.batch_size} is being used.")
        filtered_data = filtered_data[n_min:n_max]
    else:
        print(f"Batch number is not specified. Using all {len(filtered_data)} examples.")
    print("Length of filtered_data", len(filtered_data))


    #### Check if the explanations exist ####
    save_dir = os.path.join(args.result_save_dir, f'explanations/{args.model_name}/{args.dataset}/{args.algorithm}')
    os.makedirs(save_dir, exist_ok=True)
    filename = "explanations_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}.pkl"
    if os.path.exists(os.path.join(save_dir, filename)):
        print("Loading explanations...")
        filtered_explanations = pickle.load(open(os.path.join(save_dir, filename), "rb"))
        if args.algorithm != "lime":
            filtered_explanations = filtered_explanations.values
    else:
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
        
        explanations = explainer(filtered_data)

        #### Save the shap values ####
        if args.algorithm == "lime":
            explainer._save(os.path.join(save_dir, filename))
            filtered_explanations = explainer._s
        else: 
            explanations._save(os.path.join(save_dir, filename))
            filtered_explanations = explanations.values

    print("Done!")
    
    #### Evaluate the explanations ####
    scores = get_scores(filtered_data, filtered_explanations, lmmodel, args.threshold)
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
