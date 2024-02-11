import warnings

from ._coefficient import Coefficient
from ._lime_text import LimeTextGeneration
from ._random import Random

__all__ = [
    "LimeTextGeneration",
    "Random",
]


# Deprecated class alias with incorrect spelling
def Coefficent(*args, **kwargs):  # noqa
    warnings.warn(
        "Coefficent has been renamed to Coefficient. "
        "The former is deprecated and will be removed in shap 0.45.",
        DeprecationWarning
    )
    return Coefficient(*args, **kwargs)
