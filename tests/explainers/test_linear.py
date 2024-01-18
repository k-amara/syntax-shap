""" Unit tests for the Linear explainer.
"""

import numpy as np
import pytest
import scipy.special

import shap2

# Ignore expected internal shap warnings about deprecated syntax in LinearExplainer
# In future when the deprecated syntax is fully removed, the tests must be updated
pytestmark = [
    pytest.mark.filterwarnings('ignore:The option feature.* has been renamed'),
    pytest.mark.filterwarnings('ignore:The feature_perturbation option is now deprecated'),
]

def test_tied_pair():
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    explainer = shap2.LinearExplainer((beta, 0), (mu, Sigma), feature_dependence="correlation")
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05

def test_tied_pair_independent():
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    explainer = shap2.LinearExplainer((beta, 0), (mu, Sigma), feature_dependence="independent")
    assert np.abs(explainer.shap_values(X) - np.array([1, 0, 0])).max() < 0.05

def test_tied_pair_new():
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    explainer = shap2.explainers.LinearExplainer((beta, 0), shap2.maskers.Impute({"mean": mu, "cov": Sigma}))
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05

def test_wrong_masker():
    with pytest.raises(NotImplementedError):
        shap2.explainers.LinearExplainer((0, 0), shap2.maskers.Fixed())

def test_tied_triple():
    beta = np.array([0, 1, 0, 0])
    mu = 1*np.ones(4)
    Sigma = np.array([[1, 0.999999, 0.999999, 0], [0.999999, 1, 0.999999, 0], [0.999999, 0.999999, 1, 0], [0, 0, 0, 1]])
    X = 2*np.ones((1, 4))
    explainer = shap2.LinearExplainer((beta, 0), (mu, Sigma), feature_dependence="correlation")
    assert explainer.expected_value == 1
    assert np.abs(explainer.shap_values(X) - np.array([0.33333, 0.33333, 0.33333, 0])).max() < 0.05

def test_sklearn_linear():
    Ridge = pytest.importorskip('sklearn.linear_model').Ridge

    # train linear model
    X, y = shap2.datasets.california(n_points=100)
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap2.LinearExplainer(model, X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)

def test_sklearn_linear_old_style():
    Ridge = pytest.importorskip('sklearn.linear_model').Ridge

    # train linear model
    X, y = shap2.datasets.california(n_points=100)
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap2.LinearExplainer(model, X, feature_perturbation="independent")
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)

def test_sklearn_linear_new():
    Ridge = pytest.importorskip('sklearn.linear_model').Ridge

    # train linear model
    X, y = shap2.datasets.california(n_points=100)
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap2.explainers.LinearExplainer(model, X)
    shap_values = explainer(X)
    assert np.abs(shap_values.values.sum(1) + shap_values.base_values - model.predict(X)).max() < 1e-6
    assert np.abs(shap_values.base_values[0] - model.predict(X).mean()) < 1e-6

def test_sklearn_multiclass_no_intercept():
    Ridge = pytest.importorskip('sklearn.linear_model').Ridge

    # train linear model
    X, y = shap2.datasets.california(n_points=100)

    # make y multiclass
    multiclass_y = np.expand_dims(y, axis=-1)
    model = Ridge(fit_intercept=False)
    model.fit(X, multiclass_y)

    # explain the model's predictions using SHAP values
    explainer = shap2.LinearExplainer(model, X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)

def test_perfect_colinear():
    LinearRegression = pytest.importorskip('sklearn.linear_model').LinearRegression

    X, y = shap2.datasets.california(n_points=100)
    X.iloc[:, 0] = X.iloc[:, 4] # test duplicated features
    X.iloc[:, 5] = X.iloc[:, 6] - X.iloc[:, 6] # test multiple colinear features
    X.iloc[:, 3] = 0 # test null features
    model = LinearRegression()
    model.fit(X, y)
    explainer = shap2.LinearExplainer(model, X, feature_dependence="correlation")
    shap_values = explainer.shap_values(X)
    assert np.abs(shap_values.sum(1) - model.predict(X) + model.predict(X).mean()).sum() < 1e-7

def test_shape_values_linear_many_features():

    Ridge = pytest.importorskip('sklearn.linear_model').Ridge

    coef = np.array([1, 2]).T

    # FIXME: this test should ideally pass with any random seed. See #2960
    random_seed = 0
    rs = np.random.RandomState(random_seed)
    # generate linear data
    X = rs.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(X, coef) + 1 + rs.normal(scale=0.1, size=1000)

    # train linear model
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap2.LinearExplainer(model, X.mean(0).reshape(1, -1))

    values = explainer.shap_values(X)

    assert values.shape == (1000, 2)

    expected = (X - X.mean(0)) * coef
    np.testing.assert_allclose(expected - values, 0, atol=0.01)

def test_single_feature(random_seed):
    """ Make sure things work with a univariate linear regression.
    """
    Ridge = pytest.importorskip('sklearn.linear_model').Ridge

    # generate linear data
    rs = np.random.RandomState(random_seed)
    X = rs.normal(1, 10, size=(100, 1))
    y = 2 * X[:, 0] + 1 + rs.normal(scale=0.1, size=100)

    # train linear model
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = shap2.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    assert np.max(np.abs(explainer.expected_value + shap_values.sum(1) - model.predict(X))) < 1e-6

def test_sparse():
    """ Validate running LinearExplainer on scipy sparse data
    """
    make_multilabel_classification = pytest.importorskip('sklearn.datasets').make_multilabel_classification
    LogisticRegression = pytest.importorskip('sklearn.linear_model').LogisticRegression

    n_features = 20
    X, y = make_multilabel_classification(n_samples=100,
                                          sparse=True,
                                          n_features=n_features,
                                          n_classes=1,
                                          n_labels=2)

    # train linear model
    model = LogisticRegression()
    model.fit(X, y.squeeze())

    # explain the model's predictions using SHAP values
    explainer = shap2.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert np.max(np.abs(scipy.special.expit(explainer.expected_value + shap_values.sum(1)) - model.predict_proba(X)[:, 1])) < 1e-6


@pytest.mark.parametrize("feature_pertubation,masker", [
    (None, shap2.maskers.Independent),
    ("interventional", shap2.maskers.Independent),
    ("independent", shap2.maskers.Independent),
    ("correlation_dependent", shap2.maskers.Impute),
    ("correlation", shap2.maskers.Impute)
])
def test_feature_perturbation_sets_correct_masker(feature_pertubation, masker):
    Ridge = pytest.importorskip('sklearn.linear_model').Ridge

    # train linear model
    X, y = shap2.datasets.california(n_points=100)
    model = Ridge(0.1)
    model.fit(X, y)

    explainer = shap2.explainers.LinearExplainer(model, X, feature_perturbation=feature_pertubation)
    assert isinstance(explainer.masker, masker)
