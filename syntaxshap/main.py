# Import necessary libraries and modules
import os
import numpy as np
import torch
import pickle
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import dill
import pandas as pd

# Import custom modules and functions
from metrics import get_scores, save_scores
import explainers
from explainers.other import LimeTextGeneration, Random, SVSampling, SVSamplingProb, Ablation
import models
from datasets import generics_kb, inconsistent_negation, rocstories
from utils import arg_parse, fix_random_seed
from utils._exceptions import InvalidAlgorithmError
from utils._filter_data import filter_data
from utils.transformers import parse_prefix_suffix_for_tokenizer

#from huggingface_hub import login
#login(token="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")

# Define minimum required version of transformers library
MIN_TRANSFORMERS_VERSION = "4.25.1"

# Check if the transformers library meets the minimum version requirement
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."

def main(args):
    # Set random seed
    fix_random_seed(args.seed)
    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### Load the model ####
    if args.model_name == "gpt2":
        model_load = args.model_name
        tokenizer_load = args.model_name
        # Load GPT-2 model
        model = AutoModelForCausalLM.from_pretrained(model_load)
        model.to(device)
        # Set text generation parameters
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
        # Load Mistral model
        if device.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(model_load, load_in_4bit=True, cache_dir="/cluster/scratch/kamara/huggingface")#, device_map='cuda')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_load, torch_dtype=torch.float16, device_map="auto")
    else:
        raise ValueError("Unknown model type passed: %s!" % args.model_name)
    print("Model loaded")
    model.config.is_decoder = True

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Initialize TextGeneration model
    lmmodel = models.TextGeneration(model, tokenizer, device=device)
    parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(lmmodel.tokenizer)
    keep_prefix = parsed_tokenizer_dict['keep_prefix'] # check that keep_prefix is not None, value 0 or 1
    keep_suffix = parsed_tokenizer_dict['keep_suffix'] # check that keep_prefix is not None, value 0 or 1

    #### Prepare the data ####
    # Load dataset based on argument
    if args.dataset == "negation":
        data, _ = inconsistent_negation(args.data_save_dir)
    elif args.dataset == "generics":
        data, _ = generics_kb(args.data_save_dir)
    elif args.dataset == "rocstories":
        data, _ = rocstories(args.data_save_dir)
    else:
        raise ValueError("Unknown dataset type passed: %s!" % args.dataset)
    
    # Filter data based on tokenizer and specified prefixes/suffixes
    # filtered_data, filtered_ids = filter_data(data, lmmodel.tokenizer, args, keep_prefix, keep_suffix)
    filtered_data, filtered_ids = filter_data(data, lmmodel.tokenizer, args, keep_prefix, keep_suffix)
    # Get permutation indices
    if eval(args.shuffle):
        permutation_indices = np.random.permutation(len(filtered_data))
    else:
        permutation_indices = np.arange(len(filtered_data))

    # Shuffle both arrays using the same permutation indices
    data = filtered_data[permutation_indices] # check that permutation indices are in the range of 0 and len(data)
    data_ids = filtered_ids[permutation_indices]

    # Limit data to specified batch size and number
    if args.num_batch is not None:
        assert args.num_batch * args.batch_size < len(data), "Batch number is too large!"
        n_min = args.batch_size * args.num_batch
        n_max = args.batch_size * (args.num_batch + 1) if args.num_batch < len(data) // args.batch_size else len(data)
        print(f"Batch number {args.num_batch} of size {args.batch_size} is being used.")
        data = data[n_min:n_max]
        data_ids = data_ids[n_min:n_max]
    else:
        print(f"Batch number is not specified. Using all {len(data)} examples.")
    print("Length of data", len(data))

    #### Check if the explanations exist ####
    save_dir = os.path.join(args.result_save_dir, f'explanations/{args.model_name}/{args.dataset}/{args.algorithm}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = "explanations_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}.pkl"
    if os.path.exists(os.path.join(save_dir, filename)):
        print("Loading explanations...")
        results = pickle.load(open(os.path.join(save_dir, filename), "rb"))
    else:
        #### Explain the model ####
        # Choose appropriate explainer based on specified algorithm
        if args.algorithm == "random":
            explainer = Random(lmmodel, lmmodel.tokenizer)
        elif args.algorithm == "partition":
            explainer = explainers.PartitionExplainer(lmmodel, lmmodel.tokenizer)
        elif args.algorithm == "hedge":
            explainer = explainers.HEDGE(lmmodel, lmmodel.tokenizer, model)
        elif args.algorithm == "lime":
            explainer_save_dir = os.path.join(args.result_save_dir, f"explainer/seed_{args.seed}")
            os.makedirs(explainer_save_dir, exist_ok=True)
            if os.path.exists(os.path.join(explainer_save_dir, f"{args.dataset}_{args.model_name}_lime.pkl")):
                print("Loading LIME explainer...")
                explainer = dill.load(open(os.path.join(explainer_save_dir, f"{args.dataset}_{args.model_name}_lime.pkl"), "rb"))
            else:
                explainer = LimeTextGeneration(lmmodel, filtered_data[:1000])
                with open(os.path.join(explainer_save_dir, f"{args.dataset}_{args.model_name}_lime.pkl"), "wb") as file:
                    dill.dump(explainer, file)
        elif args.algorithm == "shap":
            explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, model, algorithm="shap")
        elif args.algorithm == "syntax":
            explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, model, algorithm="syntax")
        elif args.algorithm == "syntax-w":
            explainer = explainers.SyntaxExplainer(lmmodel, lmmodel.tokenizer, model, algorithm="syntax-w")
        elif args.algorithm == "svsampling":
            explainer = SVSampling(lmmodel, lmmodel.tokenizer, model)
        elif args.algorithm == "svsamplingprob":
            explainer = SVSamplingProb(lmmodel, lmmodel.tokenizer, model)
        elif args.algorithm == "ablation":
            explainer = Ablation(lmmodel, lmmodel.tokenizer, model)
        else:
            raise InvalidAlgorithmError("Unknown algorithm type passed: %s!" % args.algorithm)
        
        explanations = explainer(data)

        #### Save the shap values ####
        if args.algorithm == "lime":
            explanations = explainer._s
        else: 
            explanations = explanations.values

        results = []
        for i in range(len(explanations)):
            token_ids = lmmodel.tokenizer.encode(data[i])
            tokens = [lmmodel.tokenizer.decode(token_id) for token_id in token_ids]
            results.append({'input_id': data_ids[i], 'input': data[i], 'tokens': tokens, 'token_ids': token_ids, 'explanation': explanations[i]})
        with open(os.path.join(save_dir, filename), "wb") as f:
            pickle.dump(results, f)

    print("Done!")
    
    #### Evaluate the explanations ####
    # Calculate scores for explanations
    results = pd.DataFrame(results)
    print(f"Calculating scores for {len(results['input_id'])} explained instances...")
    scores = get_scores(results, lmmodel, args.threshold)
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
    # Execute main function
    main(args)
