import argparse
import random

import numpy as np
import torch
from transformers import set_seed

# Define the path to the data, model, logs, results, and colors
#

CKPT_ROOT = "./" #"/cluster/work/zhang/kamara/syntax-shap/"
DATA_DIR = CKPT_ROOT + "data/"
MODEL_DIR = CKPT_ROOT + "model/"
LOG_DIR = CKPT_ROOT + "logs/"
RESULT_DIR = CKPT_ROOT + "results/"



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
        "--logs_save_dir",
        help="Directory where logs are saved",
        type=str,
        default=LOG_DIR,
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
        "--model",
        help="The type of shapley value algorithm",
        type=str,
        default="gpt2", # ["mistralai/Mistral-7B-v0.1"]
    )

    parser.add_argument(
        "--algorithm",
        help="The type of shapley value algorithm",
        type=str,
        default="dtree", # ["partition", "exact", "dtree", "r-dtree"]
    )


    parser.add_argument(
        "--weighted",
        help="Weight on the contribution of each word based on its position in the tree (only for dtree and r-dtree)",
        type=str,
        default='False', # False or True
    )

    args, unknown = parser.parse_known_args()
    return parser, args


def create_args_group(parser, args):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = group_dict
    return arg_groups
