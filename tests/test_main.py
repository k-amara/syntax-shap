
"""
This Python test file contains test functions for the `main.py` script. 

These test functions cover various scenarios such as valid input arguments, invalid input arguments, 
edge cases, and exceptions that the main function might encounter. 
You can run these tests using pytest to ensure that the main function behaves correctly under different conditions.

"""

import pytest
import os
from main import main, arg_parse
from utils._exceptions import InvalidAlgorithmError

@pytest.fixture
def sample_args():
    # Define sample arguments for testing
    args = arg_parse()[1]
    args.model_name = "gpt2"  # Set model name
    args.dataset = "negations"  # Set dataset
    args.algorithm = "random"  # Set algorithm
    args.seed = 0  # Set seed
    args.result_save_dir = "results"  # Set result save directory
    args.data_save_dir = "data"  # Set data save directory
    args.batch_size = 32  # Set batch size
    args.num_batch = 2  # Set number of batches
    args.shuffle = "True"  # Set shuffle to True
    args.threshold = 0.5  # Set threshold
    return args

def test_main_function(sample_args):
    # Test the main function with sample arguments
    main(sample_args)
    assert os.path.exists(os.path.join(sample_args.result_save_dir, "explanations/gpt2/rocstories/random/seed_42/explanations_batch_2_rocstories_gpt2_random_42.pkl"))

def test_main_function_invalid_model():
    # Test the main function with invalid model name
    invalid_args = sample_args()
    invalid_args.model_name = "invalid_model"
    with pytest.raises(ValueError):
        main(invalid_args)

def test_main_function_invalid_dataset():
    # Test the main function with invalid dataset name
    invalid_args = sample_args()
    invalid_args.dataset = "invalid_dataset"
    with pytest.raises(ValueError):
        main(invalid_args)

def test_main_function_invalid_algorithm():
    # Test the main function with invalid algorithm name
    invalid_args = sample_args()
    invalid_args.algorithm = "invalid_algorithm"
    with pytest.raises(InvalidAlgorithmError):
        main(invalid_args)

def test_main_function_invalid_shuffle():
    # Test the main function with invalid shuffle argument
    invalid_args = sample_args()
    invalid_args.shuffle = "invalid_shuffle"
    with pytest.raises(ValueError):
        main(invalid_args)

def test_main_function_no_batches():
    # Test the main function without specifying batch number
    no_batch_args = sample_args()
    no_batch_args.num_batch = None
    main(no_batch_args)

def test_main_function_large_batch():
    # Test the main function with a large batch number
    large_batch_args = sample_args()
    large_batch_args.num_batch = 1000  # Set an unrealistically large batch number
    with pytest.raises(AssertionError):
        main(large_batch_args)

def test_main_function_invalid_threshold():
    # Test the main function with an invalid threshold
    invalid_args = sample_args()
    invalid_args.threshold = "invalid_threshold"
    with pytest.raises(AssertionError):
        main(invalid_args)
