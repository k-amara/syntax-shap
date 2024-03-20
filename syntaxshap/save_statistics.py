import numpy as np
from datasets import generics_kb, inconsistent_negation, rocstories
import pickle
import transformers
import os 

# Minimum required version of Transformers library
MIN_TRANSFORMERS_VERSION = "4.25.1"

# Check if the installed Transformers version meets the minimum requirement
assert (
    transformers.__version__ >= MIN_TRANSFORMERS_VERSION
), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."

def save_dataset_statistics(seed: int, dataset: str, model_name: str):
    """
    Saves statistics about the dataset filtered by invalid IDs for a given seed, dataset, and model.

    Args:
        seed (int): Seed value for random number generation.
        dataset (str): Name of the dataset.
        model_name (str): Name of the model.

    Returns:
        None
    """
    data_save_dir = "/cluster/home/kamara/syntax-shap/data"

    # Prepare the data based on the dataset name
    if dataset == "negation":
        data, _ = inconsistent_negation(data_save_dir)
    elif dataset == "generics":
        data, _ = generics_kb(data_save_dir)
    elif dataset == "rocstories":
        data, _ = rocstories(data_save_dir)

    # Load invalid IDs for the dataset, seed, and model
    filter_ids_path = os.path.join(data_save_dir, f"{dataset}/seed_{seed}")
    filename = os.path.join(filter_ids_path, f"{dataset}_{model_name}_{seed}_invalid_ids.npy")
    invalid_ids = np.load(filename, allow_pickle=True)

    # Filter the dataset based on invalid IDs
    filtered_data = np.delete(data, invalid_ids, axis=0)

    # Compute statistics
    size = []
    for prompt in filtered_data:
        n_words = len(prompt.split(" "))
        size.append(n_words)

    # Save statistics to a pickle file
    stats = {
        "dataset": dataset,
        "model_name": model_name,
        "n_data": len(data),
        "seed": seed,
        "n_filtered": len(filtered_data),
        "n_tokens_list": size,
    }
    with open(os.path.join(filter_ids_path, f"{dataset}_{model_name}_{seed}_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

if __name__ == "__main__":
    # Define datasets, seeds, and model names
    datasets = ["negation", "generics", "rocstories"]
    seeds = [0, 1, 2, 3]
    model_names = ["gpt2", "mistral"]

    # Loop through combinations of seed, dataset, and model name to save statistics
    for seed in seeds:
        for dataset in datasets:
            for model_name in model_names:
                save_dataset_statistics(seed=seed, dataset=dataset, model_name=model_name)
