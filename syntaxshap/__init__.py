from ._explanation import Cohorts, Explanation

# explainers
from .explainers import other
from .explainers._syntax import SyntaxExplainer
from .explainers._explainer import Explainer
from .explainers._partition import PartitionExplainer

try:
    # Version from setuptools-scm
    from ._version import version as __version__
except ImportError:
    # Expected when running locally without build
    __version__ = "0.0.0-not-built"


# other stuff :)
from . import datasets, links, utils  # noqa: E402
from .utils import approximate_interactions, sample  # noqa: E402

#from . import benchmark
from .utils._legacy import kmeans  # noqa: E402

# Use __all__ to let type checkers know what is part of the public API.
__all__ = [
    "Cohorts",
    "Explanation",

    # Explainers
    "other",
    "Explainer",
    "PartitionExplainer",
    "SyntaxExplainer",

    # Other stuff
    "datasets",
    "links",
    "utils",
    "approximate_interactions",
    "sample",
    "kmeans",
]
