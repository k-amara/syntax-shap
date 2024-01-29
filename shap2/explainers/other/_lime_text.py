import pandas as pd
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

from .._explainer import Explainer

class LimeTextGeneration(Explainer):
    """ Simply wrap of lime.lime_tabular.LimeTabularExplainer into the common shap interface.

    Parameters
    ----------
    model : function or iml.Model
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array
        The background dataset.

    mode : "classification" or "regression"
        Control the mode of LIME tabular.
    """

    def __init__(self, model, data, batch_size = 100, mode="classification"):
        self.model = model
        if mode not in ["classification", "regression"]:
            emsg = f"Invalid mode {mode!r}, must be one of 'classification' or 'regression'"
            raise ValueError(emsg)
        self.mode = mode

        # get all vocabulary for text generation
        voc = model.tokenizer.get_vocab()
        inv_voc = {v: k for k, v in voc.items()}
        self.vocab = dict(sorted(inv_voc.items()))


        if isinstance(data, pd.DataFrame):
            data = data.values
        self.data = data
        self.explainer = LimeTextExplainer(class_names = self.vocab)
        self.predictions = self.model(data).reshape(-1)
        

        out = self.model(data[0:1])
        if len(out.shape) == 1:
            self.out_dim = 1
            self.flat_out = True
            if mode == "classification":
                def pred(X): # assume that 1d outputs are probabilities
                    preds = self.model(X).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))
                self.model = pred
        else:
            self.out_dim = self.model(data[0:1]).shape[1]
            self.flat_out = False

        self.fit_linear_model()
        self._s = None

    def fit_linear_model(self):
        # vectorize to tf-idf vectors
        self.tfidf_vc = TfidfVectorizer(min_df = 10, max_features = 100000, analyzer = "word", ngram_range = (1, 2), stop_words = 'english', lowercase = True)
        data_vc = self.tfidf_vc.fit_transform(self.data)

        linear_model = LogisticRegression(C = 0.5, solver = "sag")
        self.linear_model = linear_model.fit(data_vc, self.predictions)        

    def __call__(self, X):
        c = make_pipeline(self.tfidf_vc, self.linear_model)
        explainer = LimeTextExplainer(class_names = self.vocab)
        attributions = []
        for i in range(X.shape[0]):
            exp = explainer.explain_instance(X[i], c.predict_proba, num_features = 20)
            print("lime explanation", exp.as_list())
            words_order = X[i].split(" ")
            print("words_order", words_order)
            exp_dict = dict(exp.as_list())
            sorted_dict = {k: exp_dict[k] for k in words_order if k in exp_dict}
            attributions.append(np.array(list(sorted_dict.values()))[:, np.newaxis])
        self._s = attributions
        return self._s
    
    def _save(self, filename):
        if self._s is None:
            raise Exception("You must run the explainer before saving it!")
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self._s, outp, pickle.HIGHEST_PROTOCOL)


