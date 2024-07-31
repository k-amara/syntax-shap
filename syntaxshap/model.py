import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define minimum required version of transformers library
MIN_TRANSFORMERS_VERSION = "4.25.1"

# Check if the transformers library meets the minimum version requirement
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."


def load_model(device, args):
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
    return model, tokenizer
