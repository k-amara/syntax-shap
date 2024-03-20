import math
from itertools import chain, combinations

import links
import numpy as np
import pandas as pd
import spacy
from _explanation import Explanation
from explainers._explainer import Explainer
from models import Model
from utils import (
    MaskedModel,
    OpChain,
    create_dataframe_from_tree,
    make_masks,
    safe_isinstance,
    spacy_doc_to_tree,
)


def convert_feat_to_mask(feature, m):
    mask = np.zeros(m)
    mask[feature] = 1
    return np.array(mask, dtype=bool)

def weight_matrix_subset(subsets, m, n, w):
    """_summary_

    Args:
        subsets (List[int]): List of integers representing one combination of words/features
        m (int): Number of features
        n (int): Number of combinations
        w (List[float]): Shapley weight of the combination

    Returns:
        _type_: Weighted matrix
    """
    Z = np.zeros((n, m + 1))
    X = np.zeros((n, m + 1))
    R = np.zeros((m + 1, n))
    for i in range(n):
        Z[i, 0] = 1
        subset_vec = subsets[i]
        n_elements = len(subset_vec)
        if n_elements > 0:
            for j in range(n_elements):
                Z[i, subset_vec[j]] = 1
    for i in range(n):
        for j in range(Z.shape[1]):
            X[i, j] = w[i] * Z[i, j]
    R = np.linalg.solve(np.dot(X.T, Z), X.T)
    return R

def weight_matrix(X, normalize_W_weights=True):
    w = X["shapley_weight"]
    if normalize_W_weights:
        w[1:-1] /= np.sum(w[1:-1])
    W = weight_matrix_subset(subsets=X["features"], m=X.iloc[-1]["n_features"], n=X.shape[0], w=w)
    return W

def shapley_weights_approx(M, N, n_features, weight_zero_m=10**6):
    x = (M - 1) / (N * n_features * (M - n_features))
    x[~np.isfinite(x)] = weight_zero_m
    return x

def shapley_weights_exact(M, n_features):
    if n_features == M:
        return 0
    else:
        return math.factorial(n_features) * math.factorial(M - n_features-1) / math.factorial(M)


def respects_order(index, causal_ordering):
    for i in index:
        # Find the position of i in causal_ordering
        idx_position = next((pos for pos, sublist in enumerate(causal_ordering) if i in sublist), -1)

        # Check if i is in causal_ordering
        if idx_position == -1:
            raise ValueError("Element not found in causal_ordering")

        # Check for precedents if not in the root set (first element)
        if idx_position > 0:
            # Get precedents
            precedents = [item for sublist in causal_ordering[:idx_position] for item in sublist]

            # Check if all precedents are in index
            if not set(precedents).issubset(set(index)):
                return False

    return True

def feature_exact(M, asymmetric=False, causal_ordering=None):
    features = id_combination = n_features = shapley_weight = N = None
    dt = pd.DataFrame({'id_combination': range(2**M)})
    list_combinations = [list(x) for x in chain(*[combinations(range(M), i) for i in range(M+1)])]
    dt['features'] = list_combinations
    dt['n_features'] = dt['features'].apply(len)
    dt['N'] = dt.groupby('n_features')['n_features'].transform('count')
    # dt['weight_approx'] = shapley_weights_approx(M, dt['N'], dt['n_features'])
    # dt['weight'] = dt['n_features'].apply(lambda row: shapley_weights_exact(M, row))

    if asymmetric:
        if causal_ordering is None:
            causal_ordering = [list(range(m))]

        dt = dt[dt['features'].apply(lambda x: respects_order(x, causal_ordering))]
        dt['N'] = dt.groupby('n_features')['n_features'].transform('count')
        # dt['weight_approx'] = shapley_weights_approx(M, dt['N'], dt['n_features'])
        # dt['weight'] = dt['n_features'].apply(lambda row: shapley_weights_exact(M, row))

    dt['mask'] = dt['features'].apply(lambda x: convert_feat_to_mask(x, M))

    return dt

class SyntaxExplainer(Explainer):

    def __init__(self, model, masker, algorithm='syntax', confounding=True, output_names=None, link=links.identity, linearize_link=True,
                 feature_names=None, **call_args):
        """ Uses the Syntax SHAP method to explain the output of any function.

        Syntax SHAP


        Parameters
        ----------
        model : function
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes the output of the model for those samples.

        masker : function or numpy.array or pandas.DataFrame or tokenizer
            The function used to "mask" out hidden features of the form `masker(mask, x)`. It takes a
            single input sample and a binary mask and returns a matrix of masked samples. These
            masked samples will then be evaluated using the model function and the outputs averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. Domain specific masking
            functions are available in shap such as syntaxshap.maskers.Text
            for text.

        type: str
            The type of algo to use. The options are 'shap', 'syntax', 'syntax_w'.


        Examples
        --------
       """

        super().__init__(model, masker, link=link, linearize_link=linearize_link, \
                         output_names = output_names, feature_names=feature_names)

        # convert dataframes
        # if isinstance(masker, pd.DataFrame):
        #     masker = TabularMasker(masker)
        # elif isinstance(masker, np.ndarray) and len(masker.shape) == 2:
        #     masker = TabularMasker(masker)
        # elif safe_isinstance(masker, "transformers.PreTrainedTokenizer"):
        #     masker = TextMasker(masker)
        # self.masker = masker

        # TODO: maybe? if we have a tabular masker then we build a PermutationExplainer that we
        # will use for sampling
        self.input_shape = masker.shape[1:] if hasattr(masker, "shape") and not callable(masker.shape) else None
        # self.output_names = output_names
        if not safe_isinstance(self.model, "models.Model"):
            self.model = Model(self.model)#lambda *args: np.array(model(*args))
        self.expected_value = None
        self._curr_base_value = None

        # Define the type of Syntax SHAP
        self.algorithm = algorithm
        self.weighted = True if algorithm == 'syntax-w' else False
        self.confounding = confounding

        # handle higher dimensional tensor inputs
        if self.input_shape is not None and len(self.input_shape) > 1:
            self._reshaped_model = lambda x: self.model(x.reshape(x.shape[0], *self.input_shape))
        else:
            self._reshaped_model = self.model

        # if we have gotten default arguments for the call function we need to wrap ourselves in a new class that
        # has a call function with those new default arguments
        if len(call_args) > 0:
            class SyntaxExplainer(self.__class__):
                # this signature should match the __call__ signature of the class defined below
                def __call__(self, *args, max_evals=500, outputs=None):
                    return super().__call__(
                        *args, max_evals=max_evals, outputs=outputs
                    )
            SyntaxExplainer.__call__.__doc__ = self.__class__.__call__.__doc__
            self.__class__ = SyntaxExplainer
            for k, v in call_args.items():
                self.__call__.__kwdefaults__[k] = v

    # note that changes to this function signature should be copied to the default call argument wrapper above
    def __call__(self, *args, max_evals=500, main_effects=False, error_bounds=False, batch_size="auto",
                 outputs=None, silent=False):
        """ Explain the output of the model on the given arguments.
        """
        return super().__call__(
            *args, max_evals=max_evals, main_effects=main_effects, error_bounds=error_bounds, batch_size=batch_size,
            outputs=outputs, silent=silent
        )

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).
        """

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)
        # make sure we have the base value and current value outputs
        mask_size_with_prefix_and_suffix = len(fm)
        M = len(row_args[0].split(' '))
        m00 = np.zeros(mask_size_with_prefix_and_suffix, dtype=bool)
        # if not fixed background or no base value assigned then compute base value for a row
        if self._curr_base_value is None or not getattr(self.masker, "fixed_background", False):
            self._curr_base_value = fm(m00.reshape(1, -1), zero_index=0)[0] # the zero index param tells the masked model what the baseline is
        f11 = fm(~m00.reshape(1, -1))[0]

        if callable(self.masker.clustering):
            self._clustering = self.masker.clustering(*row_args)
            self._mask_matrix = make_masks(self._clustering)

        if hasattr(self._curr_base_value, 'shape') and len(self._curr_base_value.shape) > 0:
            if outputs is None:
                outputs = np.arange(len(self._curr_base_value))
            elif isinstance(outputs, OpChain):
                outputs = outputs.apply(Explanation(f11)).values

            out_shape = (2*self._clustering.shape[0]+1, len(outputs))
        else:
            out_shape = (2*self._clustering.shape[0]+1,)

        if max_evals == "auto":
            max_evals = 500

        self.values = np.zeros(out_shape)
        self.dvalues = np.zeros(out_shape)

        if 'syntax' in self.algorithm:
            # Build the dependency dataframe
            dependency_dt = self.dependency_dt(*row_args)
        else:
            dependency_dt = None
        self.compute_shapley_values(fm, self._curr_base_value, M, algorithm=self.algorithm, weighted=self.weighted, dependency_dt=dependency_dt, confounding=self.confounding)
        
        mask_shapes = []
        for s in fm.mask_shapes:
            s = list(s)
            s[0] -= self.keep_prefix + self.keep_suffix
            mask_shapes.append(tuple(s))
        return {
            "values": self.values[:M].copy(),
            "expected_values": self._curr_base_value if outputs is None else self._curr_base_value[outputs],
            "mask_shapes": [s + out_shape[1:] for s in mask_shapes],
            "main_effects": None,
            "output_indices": outputs,
            "output_names": getattr(self.model, "output_names", None)
        }

    def __str__(self):
        return "explainers.DependencyExplainer()"

    def dependency_dt(self, *args):# Example usage:
        nlp = spacy.load("en_core_web_sm")
        text = args[0]
        doc = nlp(text+' MASK')

        # Convert Spacy dependency tree to a Tree object
        tree_root = spacy_doc_to_tree(doc)

        # Example usage with the Tree structure
        tree_df = create_dataframe_from_tree(tree_root)
        tree_df = tree_df[tree_df['token'] != 'MASK']
        return tree_df

    def compute_shapley_values(self, fm, f00, M, algorithm='syntax', dependency_dt=None, weighted=False):
        if (dependency_dt is not None) and (dependency_dt['position'].max() + 1 != M):
            return ValueError("The tokenizer divided words into subwords and the dependency tree only consider full words")
        dt_exact = feature_exact(M)

        count_updates = np.zeros(M, dtype=int)
        
        if algorithm=='syntax' or algorithm=='syntax-w':
            causal_ordering = []
            # Find unique levels
            unique_levels = dependency_dt['level'].unique()

            # Loop over unique levels
            for level in unique_levels:
                # Print the positions of rows for each level
                positions = dependency_dt[dependency_dt['level'] == level]['position'].tolist()
                causal_ordering.append(positions)

            dt = feature_exact(M, asymmetric=True, causal_ordering=causal_ordering)

        elif algorithm=='shap':
            dt = dt_exact

        else:
            return ValueError("The algorithm must be either 'syntax', 'syntax-w' or 'shap'")

        dt = dt.reset_index(drop=True)
        max_id_combination = dt.index.max()

        for i in range(max_id_combination):
            combination = dt['features'][i]
            m00 = convert_feat_to_mask(combination, M)
            remaining_indices = list(set(range(M)) - set(combination))
            for ind in remaining_indices:
                m10 = m00.copy()
                m10[ind] = 1
                f10 = fm(m10.reshape(1,-1))[0]
                weight = dependency_dt[dependency_dt['position'] == ind]['level_weight'] if weighted else 1
                self.dvalues[ind] += (f10-f00) * weight
                count_updates[ind] += 1

        self.values = self.dvalues[:M]/count_updates[:, np.newaxis]
        self.values = self.values/np.sum(self.values)
