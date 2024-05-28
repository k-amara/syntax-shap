import numpy as np

import links
from models import Model
from utils import MaskedModel
from captum.attr import (
    ShapleyValueSampling,
    TextTokenInput,
    LLMAttribution
)

from explainers._explainer import Explainer


class SVSampling(Explainer):
    """ Simply returns random (normally distributed) feature attributions.

    This is only for benchmark comparisons. It supports both fully random attributions and random
    attributions that are constant across all explanations.
    """
    def __init__(self, model, masker, model_init, link=links.identity, feature_names=None, linearize_link=True, constant=False, **call_args):
        self.tokenizer = masker
        self.model_init = model_init
        self.device = model_init.device
        super().__init__(model, masker, link=link, linearize_link=linearize_link, feature_names=feature_names)

        if not isinstance(model, Model):
            self.model = Model(model)

        for arg in call_args:
            self.__call__.__kwdefaults__[arg] = call_args[arg]

        sv = ShapleyValueSampling(self.model_init) 
        self.llm_attr = LLMAttribution(sv, self.tokenizer)


    def _format_model_input(self, model_input):
        """
        Convert str to tokenized tensor
        to make LLMAttribution work with model inputs of both
        raw text and text token tensors
        """
        # return tensor(1, n_tokens)
        if isinstance(model_input, str):
            return self.tokenizer.encode(model_input, return_tensors="pt").to(
                self.device
            )
        return model_input.to(self.device)
    
    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """ Explains a single row.
        """

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # compute any custom clustering for this row
        row_clustering = None
        if getattr(self.masker, "clustering", None) is not None:
            if isinstance(self.masker.clustering, np.ndarray):
                row_clustering = self.masker.clustering
            elif callable(self.masker.clustering):
                row_clustering = self.masker.clustering(*row_args)
            else:
                raise NotImplementedError("The masker passed has a .clustering attribute that is not yet supported by the Permutation explainer!")

        inp = TextTokenInput(
            row_args[0], 
            self.tokenizer,
            skip_tokens=[1],  # skip the special token for the start of the text <s>
        )
        
        target = ''
        k = 0
        while target == '':
            model_inp = self._format_model_input(inp.to_model_input())
            output_tokens = self.model_init.generate(model_inp, max_new_tokens = k+1, do_sample = False)
            target_tokens = output_tokens[0][model_inp.size(1)+k :]
            print('target_tokens:', target_tokens)
            expected_value = target_tokens.detach().cpu().numpy()[0]
            target= self.tokenizer.decode(expected_value.astype(int), skip_special_tokens=True)
            if k>1:
                print('input:', row_args[0])
                print('target_tokens:', target_tokens)
            print('target:', target)
            k += 1
            
        attr_res = self.llm_attr.attribute(inp, target=target)
        row_values = attr_res.seq_attr.detach().cpu().numpy()   

        mask_shapes = []
        for s in fm.mask_shapes:
            s = list(s)
            s[0] -= self.keep_prefix + self.keep_suffix
            mask_shapes.append(tuple(s))

        return {
            "values": row_values,
            "expected_values": expected_value,
            "mask_shapes": [(s[0],1) for s in mask_shapes],
            "main_effects": None,
            "clustering": row_clustering,
            "error_std": None,
            "output_names": self.model.output_names if hasattr(self.model, "output_names") else None
        }
