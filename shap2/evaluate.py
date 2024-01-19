import os
import pickle
import numpy as np
from utils import arg_parse, fix_random_seed
from metrics import fidelity_keep, fidelity_rmv, kl_fidelity_keep



def get_scores(args, data, explanations, lmmodel):

    #### Prepare the data ####
    filter_ids_path = os.path.join(args.result_save_dir, "data/invalid_ids.npy")
    if os.path.exists(filter_ids_path):
        invalid_ids = np.load(filter_ids_path, allow_pickle=True)
    else:
        invalid_ids = []
    ## filter the invalid inputs
    filtered_data = np.delete(data, invalid_ids, axis=0)
    filtered_explanations = np.delete(explanations, invalid_ids, axis=0)
    assert len(filtered_data) == len(filtered_explanations)

    #### Evaluate the fidelity ####
    fid_keep_scores = fidelity_keep(filtered_data, filtered_explanations, lmmodel)
    fid_rmv_scores = fidelity_rmv(filtered_data, filtered_explanations, lmmodel)
    kl_fid_mean, kl_fid_scores = kl_fidelity_keep(filtered_data, filtered_explanations, lmmodel)
    scores = {
        "fid_keep": fid_keep_scores,
        "fid_rmv": fid_rmv_scores,
        "kl_fid_mean": kl_fid_mean,
        "kl_fid_scores": kl_fid_scores,
    }
    return scores

def save_scores(args, scores):
    save_dir = os.path.join(args.result_save_dir, 'scores')
    filename = f"scores_{args.dataset}_{args.model_name}_{args.algorithm}"
    if eval(args.weighted):
        filename += "_weighted"
    filename += ".pkl"
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(scores, f)



if __name__ == "__main__":
    parser, args = arg_parse()
    fix_random_seed(args.seed)

    if args.algorithm.endswith("dtree"):
        weighted = args.weighted
    else:
        weighted = False
    w_str = "_weighted" if weighted else ""
    print("w_str", w_str)
    save_dir = os.path.join(args.result_save_dir, 'shap_values')
    filename = f"shap_values_{args.dataset}_{args.model_name}_{args.algorithm}"
    if eval(args.weighted):
        filename += "_weighted"
    filename += ".pkl"
    with open(os.path.join(args.result_save_dir, filename), "rb") as f:
        shap_values = pickle.load(f)
    explanations = shap_values.values

    #scores = get_scores(args, data, explanations, lmmodel)
    #save_scores(args, scores)
