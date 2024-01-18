""" This file contains tests for custom (user supplied) maskers.
"""

import numpy as np

import shap2


def test_raw_function():
    """ Make sure passing a simple masking function works.
    """

    X, _ = shap2.datasets.california(n_points=500)

    def test(X):
        return np.sum(X, 1)

    def custom_masker(mask, x):
        return (x * mask).reshape(1, len(x)) # just zero out the features we are masking

    explainer = shap2.Explainer(test, custom_masker)
    shap_values = explainer(X[:100])

    assert np.var(shap_values.values - shap_values.data) < 1e-6
