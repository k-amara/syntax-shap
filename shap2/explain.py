import csv
import os
from metrics import get_scores, save_scores

import explainers
import models
import numpy as np
import shap
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    print("args", args)

    #### Load the model ####
    if args.model_name == "gpt2":
        model_load = args.model_name
        tokenizer_load = args.model_name
    elif args.model_name == "mistral":
        model_load = "mistralai/Mistral-7B-v0.1"
        tokenizer_load = os.path.join(args.model_save_dir, args.model_name) + "/tokenizer"

    model = AutoModelForCausalLM.from_pretrained(model_load)
    model.to(device)
    # model.save_pretrained(f'/cluster/work/zhang/kamara/syntax-shap/models/{args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load)
    tokenizer.pad_token = tokenizer.eos_token
    if args.model_name == "gpt2":
        # set model decoder to true
        model.config.is_decoder = True
        # set text-generation params under task_specific_params
        model.config.task_specific_params["text-generation"] = {
            "do_sample": True,
            "max_new_tokens": 1,
            "temperature": 0.7,
            "top_k": 50,
            "no_repeat_ngram_size": 2,
        }
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

    #### Explain the model ####
    if args.algorithm == "partition_init":
        explainer = shap.explainers.PartitionExplainer(lmmodel, lmmodel.tokenizer)
    elif args.algorithm == "partition":
        explainer = explainers.PartitionExplainer(lmmodel, lmmodel.tokenizer)
    elif args.algorithm == "exact":
        explainer = explainers.DependencyExplainer(lmmodel, lmmodel.tokenizer, algorithm="exact", weighted=eval(args.weighted))
    elif args.algorithm == "dtree":
        explainer = explainers.DependencyExplainer(lmmodel, lmmodel.tokenizer, algorithm="dtree", weighted=eval(args.weighted))
    elif args.algorithm == "r-dtree":
        explainer = explainers.DependencyExplainer(lmmodel, lmmodel.tokenizer, algorithm="r-dtree", weighted=eval(args.weighted))
    else:
        raise InvalidAlgorithmError("Unknown dependency tree algorithm type passed: %s!" % args.algorithm)
    shap_values = explainer(data)

    #### Save the shap values ####
    save_dir = os.path.join(args.result_save_dir, 'shap_values')
    filename = f"shap_values_{args.dataset}_{args.model_name}_{args.algorithm}"
    if eval(args.weighted):
        filename += "_weighted"
    filename += ".pkl"
    shap_values._save(os.path.join(save_dir, filename))
    print("Done!")
    
    #### Evaluate the explanations ####
    explanations = shap_values.values
    
    #### Filter invalid data ####
    # Tokenization might split words into multiple tokens, which is not supported by the current implementation
    filter_ids_path = os.path.join(args.result_save_dir, "data/invalid_ids.npy")
    if os.path.exists(filter_ids_path):
        invalid_ids = np.load(filter_ids_path, allow_pickle=True)
    else:
        invalid_ids = []
    filtered_data = np.delete(data, invalid_ids, axis=0)
    filtered_explanations = np.delete(explanations, invalid_ids, axis=0)
    assert len(filtered_data) == len(filtered_explanations)

    scores = get_scores(args, filtered_data, filtered_explanations, lmmodel)
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
