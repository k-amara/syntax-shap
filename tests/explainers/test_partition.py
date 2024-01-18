""" This file contains tests for partition explainer.
"""

import pickle

import shap2

from . import common


def test_translation(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_additivity(shap2.explainers.PartitionExplainer, model, tokenizer, data)


def test_translation_auto(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_additivity(shap2.Explainer, model, tokenizer, data)


def test_translation_algorithm_arg(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_additivity(shap2.Explainer, model, tokenizer, data, algorithm="partition")

def test_tabular_single_output():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap2.explainers.PartitionExplainer, model.predict, shap2.maskers.Partition(data), data)

def test_tabular_multi_output():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap2.explainers.PartitionExplainer, model.predict_proba, shap2.maskers.Partition(data), data)


def test_serialization(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_serialization(shap2.explainers.PartitionExplainer, model, tokenizer, data)


def test_serialization_no_model_or_masker(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_serialization(
        shap2.explainers.Partition, model, tokenizer, data, model_saver=None, masker_saver=None,
        model_loader=lambda _: model, masker_loader=lambda _: tokenizer
    )


def test_serialization_custom_model_save(basic_translation_scenario):
    model, tokenizer, data = basic_translation_scenario
    common.test_serialization(shap2.explainers.PartitionExplainer, model, tokenizer, data, model_saver=pickle.dump, model_loader=pickle.load)
