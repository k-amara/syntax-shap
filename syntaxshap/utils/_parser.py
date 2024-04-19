import argparse
import random

import numpy as np
import torch
from transformers import set_seed

# Define the path to the data, model, logs, results, and colors
#

CKPT_ROOT = "/cluster/home/kamara/syntax-shap/"
STORAGE = "/cluster/work/zhang/kamara/syntax-shap/"
#CKPT_ROOT = "/Users/kenzaamara/GithubProjects/syntax-shap/"
#STORAGE = "/Users/kenzaamara/GithubProjects/syntax-shap/"
DATA_DIR = CKPT_ROOT + "data/"
MODEL_DIR = STORAGE + "models/"
FIG_DIR = CKPT_ROOT + "figures/"
RESULT_DIR = STORAGE + "results/"



def fix_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dest", help="dest", type=str, default="/cluster/home/kamara/"
    )

    parser.add_argument(
        "--data_save_dir",
        help="Directory where benchmark is located",
        type=str,
        default=DATA_DIR,
    )

    parser.add_argument(
        "--model_save_dir",
        help="Directory where figures are saved",
        type=str,
        default=MODEL_DIR,
    )

    parser.add_argument(
        "--result_save_dir",
        help="Directory where results are saved",
        type=str,
        default=RESULT_DIR,
    )
    parser.add_argument(
        "--fig_save_dir",
        help="Directory where figures are saved",
        type=str,
        default="figures",
    )

    parser.add_argument(
        "--seed", help="random seed", type=int, default=0
    )

    parser.add_argument(
        "--batch_size", type=int, default=64
    )

    parser.add_argument(
        "--shuffle", type=str, default='True', help="shuffle the data if 'True' else 'False'", choices=["True", "False"]
    )

    parser.add_argument(
        "--num_batch", type=int, default=None
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="negation", choices=["negation", "rocstories", "generics"]
    )

    parser.add_argument(
        "--model_name",
        help="The type of shapley value algorithm",
        type=str,
        default="gpt2", choices=["gpt2", "mistral"]
    )

    parser.add_argument(
        "--algorithm",
        help="The type of shapley value algorithm",
        type=str,
        default="syntax", choices=["random", "lime", "partition", "shap", "syntax", "syntax-w"]
    )

    parser.add_argument(
        "--threshold",
        help="The percentage of important indices ",
        type=float,
        default=0.2
    )

    args, unknown = parser.parse_known_args()
    return parser, args


def create_args_group(parser, args):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = group_dict
    return arg_groups
