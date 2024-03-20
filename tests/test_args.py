import pytest
import argparse
from main import arg_parse

@pytest.fixture
def parser():
    return arg_parse()[0]

@pytest.fixture
def args(parser):
    return arg_parse()[1]

def test_arguments_type(args):
    # Check the type of each argument
    assert isinstance(args.dest, str)
    assert isinstance(args.data_save_dir, str)
    assert isinstance(args.model_save_dir, str)
    assert isinstance(args.result_save_dir, str)
    assert isinstance(args.fig_save_dir, str)
    assert isinstance(args.seed, int)
    assert isinstance(args.batch_size, int)
    assert isinstance(args.shuffle, str)  # Should be either "True" or "False"
    assert isinstance(args.num_batch, (int, type(None)))
    assert isinstance(args.dataset, str)  # Should be one of "negation", "rocstories", "generics_kb"
    assert isinstance(args.model_name, str)  # Should be one of "gpt2", "mistral"
    assert isinstance(args.algorithm, str)  # Should be one of "random", "partition", "shap", "syntax", "syntax-w"
    assert isinstance(args.threshold, float)

def test_arguments_range(args):
    # Check the range of threshold argument
    assert 0 <= args.threshold <= 1
    assert args.shuffle in ["True", "False"]
    assert args.dataset in ["negation", "rocstories", "generics_kb"]
    assert args.model_name in ["gpt2", "mistral"]
    assert args.algorithm in ["random", "partition", "shap", "syntax", "syntax-w"]
    assert args.num_batch is None or args.num_batch > 0
    assert args.batch_size > 0
    assert args.seed >= 0

if __name__ == "__main__":
    pytest.main([__file__])
