"""Optional local-model adapters.

Importing this package never imports Torch or Transformers. Those dependencies
are resolved only when a concrete factory is called through LazyBackendAdapter.
"""

from .ar import create_ar_transformers
from .llada import create_llada

__all__ = [
    "create_ar_transformers",
    "create_llada",
]
