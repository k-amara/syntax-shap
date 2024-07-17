import os
import dill

import explainers
from explainers.other import LimeTextGeneration, Random, SVSampling, Ablation, HEDGEOrig
from utils._exceptions import InvalidAlgorithmError
from utils._general import convert_to_token_expl




def compute_explanations(lmmodel, model, data, data_ids, filtered_data, args):
    #### Explain the model ####
    # Choose appropriate explainer based on specified algorithm
    if args.algorithm == "random":
        explainer = Random(lmmodel, lmmodel.tokenizer)
    elif args.algorithm == "partition":
        explainer = explainers.PartitionExplainer(lmmodel, lmmodel.tokenizer)
    elif args.algorithm == "hedge":
        explainer = explainers.HEDGE(lmmodel, lmmodel.tokenizer, model)
    elif args.algorithm == "hedge_orig":
        explainer = HEDGEOrig(lmmodel, lmmodel.tokenizer)
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
        if args.algorithm == "lime":
            token_explanation = convert_to_token_expl(data[i], explanations[i], lmmodel.tokenizer, keep_prefix=keep_prefix)
        else:
            token_explanation = explanations[i]
        assert len(token_explanation) + args.keep_prefix == len(token_ids), "Length of explanations and data do not match!"
        results.append({'input_id': data_ids[i], 'input': data[i], 'tokens': tokens, 'token_ids': token_ids, 'explanation': token_explanation})
    return results

def save_path(args):
    save_dir = os.path.join(args.result_save_dir, f'explanations/{args.model_name}/{args.dataset}/{args.algorithm}/seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    filename = "explanations_"
    filename += f"batch_{args.num_batch}_" if args.num_batch is not None else ""
    filename += f"{args.dataset}_{args.model_name}_{args.algorithm}_{args.seed}.pkl"
    return os.path.join(save_dir, filename)
    