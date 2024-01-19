import csv
import os

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


    #### Define the LM model ####
    model = AutoModelForCausalLM.from_pretrained(args.model) # AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
    tsv_file = open("./data/Inconsistent-Dataset-Negation.tsv")
    read_tsv = list(csv.reader(tsv_file, delimiter="\t"))
    data = []
    for row in read_tsv:
        data.append(row[1][:-8])
    data = np.array(data)
    print(f"Inconsistent-Dataset-Negation.tsv: {len(data)}")

    # build the right subclass
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
    filename = f"loss_dist/shap_values_{args.algorithm}"
    if eval(args.weighted):
        filename += "_weighted"
    filename += ".pkl"
    shap_values._save(os.path.join(args.result_save_dir, filename))
    print("Done!")

    #### Save the shap values ####


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
