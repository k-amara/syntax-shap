import os
import pandas as pd
import models
import numpy as np
from datasets import generics_kb, inconsistent_negation, rocstories
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import arg_parse, fix_random_seed

# Minimum required version of Transformers library
MIN_TRANSFORMERS_VERSION = "4.25.1"

# Check if the installed Transformers version meets the minimum requirement
assert (
    transformers.__version__ >= MIN_TRANSFORMERS_VERSION
), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."

def save_predictions(args):
    """
    Generates and saves predictions for a given dataset using a specified model.

    Args:
        args: Parsed command-line arguments.

    """
    # Fix random seed for reproducibility
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### Load the model ####
    if args.model_name == "gpt2":
        model_load = args.model_name
        tokenizer_load = args.model_name
        model = AutoModelForCausalLM.from_pretrained(model_load)
        model.to(device)
        # Set text-generation parameters under task_specific_params
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

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Create the language model
    lmmodel = models.TextGeneration(model, tokenizer, device=device)

    #### Prepare the data ####
    if args.dataset == "negation":
        data, _ = inconsistent_negation(args.data_save_dir)
    elif args.dataset == "generics":
        data, _ = generics_kb(args.data_save_dir)
    elif args.dataset == "rocstories":
        data, _ = rocstories(args.data_save_dir)
    
    # Generate predictions and save them to a CSV file
    input_ids = np.arange(len(data))
    df_pred = pd.DataFrame(columns=['input_id', 'input', 'y'])
    df_pred['input'] = data
    df_pred['input_id'] = input_ids
    df_pred['y'] = df_pred['input'].apply(lambda x: lmmodel.tokenizer.decode(lmmodel(x)[0]))
    print(df_pred.head(10))

    # Save predictions to a CSV file
    output_file = f'{args.data_save_dir}/{args.dataset}/seed_{args.seed}/{args.dataset}_{args.model_name}_{args.seed}_predictions.csv'
    df_pred.to_csv(output_file, index=False)
    print("Predictions saved to:", output_file)

if __name__ == "__main__":
    # Parse command-line arguments and save predictions
    parser, args = arg_parse()
    save_predictions(args)
