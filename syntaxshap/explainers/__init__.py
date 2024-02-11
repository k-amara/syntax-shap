from ._syntax import SyntaxExplainer
from ._partition import PartitionExplainer

# Alternative legacy "short-form" aliases, which are kept here for backwards-compatibility
Partition = PartitionExplainer
Syntax = SyntaxExplainer

__all__ = [
    "PartitionExplainer",
    "SyntaxExplainer",
]
