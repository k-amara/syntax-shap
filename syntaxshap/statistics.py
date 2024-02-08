
import numpy as np
from datasets import generics_kb, generics_kb_large, inconsistent_negation, rocstories
import pickle
import transformers
import os 


#import shap

MIN_TRANSFORMERS_VERSION = "4.25.1"

# check transformers version
assert (
    transformers.__version__ >= MIN_TRANSFORMERS_VERSION
), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."

data_save_dir = "/cluster/home/kamara/syntax-shap/data"

datasets = ["negation", "generics", "rocstories"]
seeds = [0,1,2,3]
model_names = ["gpt2", "mistral"]

for seed in seeds:
    for dataset in datasets:
        for model_name in model_names:
            #### Prepare the data ####
            if dataset == "negation":
                data, _ = inconsistent_negation(data_save_dir)
            elif dataset == "generics":
                data, _ = generics_kb_large(data_save_dir)
            elif dataset == "rocstories":
                data, _ = rocstories(data_save_dir)


            filter_ids_path = os.path.join(data_save_dir, f"{dataset}/seed_{seed}")
            filename = os.path.join(filter_ids_path, f"{dataset}_{model_name}_{seed}_invalid_ids.npy")
            invalid_ids = np.load(filename, allow_pickle=True)
            filtered_data = np.delete(data, invalid_ids, axis=0)

            size = []
            for prompt in filtered_data:
                n_words = len(prompt.split(" "))
                size.append(n_words)

            stats = {
                "dataset": dataset,
                "model_name": model_name,
                "n_data": len(data),
                "seed": seed,
                "n_filtered": len(filtered_data),
                "n_tokens_list": size,
            }
            #with open(os.path.join(filter_ids_path, f"{dataset}_{model_name}_{seed}_stats.pkl"), "wb") as f:
                #pickle.dump(stats, f)