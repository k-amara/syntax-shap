import pandas as pd
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

from .._explainer import Explainer

class LimeTextGeneration(Explainer):
    """
    Wrapper for LimeTextExplainer into the common SHAP interface for text generation models.

    Parameters:
        model (function or iml.Model): 
            User-supplied function that takes a matrix of samples (# samples x # features) and
            computes the output of the model for those samples. The output can be a vector
            (# samples) or a matrix (# samples x # model outputs).

        data (numpy.array): 
            The background dataset.

        batch_size (int): 
            Size of batches for batch prediction.

        mode (str): 
            Mode of LimeTextGeneration, either "classification" or "regression".
    """

    def __init__(self, model, data, batch_size=100, mode="classification"):
        self.model = model
        if mode not in ["classification", "regression"]:
            raise ValueError(f"Invalid mode {mode!r}, must be one of 'classification' or 'regression'")
        self.mode = mode

        # Get vocabulary for text generation
        voc = model.tokenizer.get_vocab()
        inv_voc = {v: k for k, v in voc.items()}
        self.vocab = dict(sorted(inv_voc.items()))

        if isinstance(data, pd.DataFrame):
            data = data.values
        self.data = data
        self.explainer = LimeTextExplainer(class_names=self.vocab)

        # Compute predictions
        # If the data is too large, we need to split it into batches
        if len(data) > 300:
            predictions = []
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                batch_predictions = self.model(batch_data).reshape(-1)
                predictions.append(batch_predictions)
            self.predictions = np.concatenate(predictions)
        else:
            self.predictions = self.model(data).reshape(-1)

        # Check output dimension
        out = self.model(data[0:1])
        if len(out.shape) == 1:
            self.out_dim = 1
            self.flat_out = True
            if mode == "classification":
                def pred(X):  # Assume 1D outputs are probabilities
                    preds = self.model(X).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))
                self.model = pred
        else:
            self.out_dim = self.model(data[0:1]).shape[1]
            self.flat_out = False

        # Fit linear model
        self.fit_linear_model()
        self._s = None

    def fit_linear_model(self):
        # Vectorize to tf-idf vectors
        self.tfidf_vc = TfidfVectorizer(min_df=10, max_features=100000, analyzer="word", ngram_range=(1, 2), stop_words='english', lowercase=True)
        data_vc = self.tfidf_vc.fit_transform(self.data)

        # Fit logistic regression model
        linear_model = LogisticRegression(C=0.5, solver="sag")
        self.linear_model = linear_model.fit(data_vc, self.predictions)        

    def __call__(self, X):
        # Create pipeline
        c = make_pipeline(self.tfidf_vc, self.linear_model)
        explainer = LimeTextExplainer(class_names=self.vocab)
        attributions = []
        for i in range(X.shape[0]):
            # Explain instance
            tokens = self.model.tokenizer.encode(X[i])
            exp = explainer.explain_instance(X[i], c.predict_proba, num_features=20)
            words_order = X[i].split()
            exp_dict = dict(exp.as_list())
            explanation = [exp_dict[k] for k in words_order if k in exp_dict]
            attributions.append(np.array(explanation)[:, np.newaxis])
        self._s = attributions
        return self._s
    
    def _save(self, filename):
        if self._s is None:
            raise Exception("You must run the explainer before saving it!")
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self._s, outp, pickle.HIGHEST_PROTOCOL)
