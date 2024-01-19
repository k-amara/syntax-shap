import csv
import os

import explainers
import models
import numpy as np
import shap
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, MistralForCausalLM
from utils import arg_parse, fix_random_seed


def main(args):
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("args", args)

    #### Define the LM model ####

    if args.model_name == "gpt2":
        model_load = args.model_name
        tokenizer_load = args.model_name
    elif args.model_name == "mistral":
        model_load = f'/cluster/work/zhang/kamara/syntax-shap/models/{args.model_name}'#"mistralai/Mistral-7B-v0.1"
        tokenizer_load = os.path.join(args.model_save_dir, args.model_name)

    model = AutoModelForCausalLM.from_pretrained(model_load)
    print('model', model)
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

    prompt = "My favourite condiment is"
    print(f"Prompt: {prompt}")

    model_inputs = tokenizer([prompt], return_tensors="pt")#.to(device)
    # model.to(device)
    print("model starts generating...")
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    print("generated_ids", generated_ids)
    decoded_str = tokenizer.batch_decode(generated_ids)[0]
    print(f"Generated text: {decoded_str}")

    lmmodel = models.TextGeneration(model, tokenizer, device=device)
    print("lmmodel.inner_model", lmmodel.inner_model)
    input = (np.array(["Peter is a father with a"], dtype="<U21"),)
    print('input', input)
    print("lmmodel(*input)", lmmodel(*input))



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
