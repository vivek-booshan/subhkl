from ._sinkhorn import sinkhorn_indexer
from ._cosine import cosine_indexer
from ._soft import soft_indexer
from ._binary import binary_indexer

__all__ = [
    "sinkhorn_indexer",
    "cosine_indexer",
    "soft_indexer",
    "binary_indexer",
]
