"""MemPalace — standalone AI memory system.

The highest-scoring AI memory system on GitHub (96.6% LongMemEval R@5).
Stores verbatim text in ChromaDB, temporal knowledge in SQLite.

Local-only: no API key, no network calls, all storage on-machine.
"""

from mempalace._compat import MemoryProvider, tool_error
from mempalace.core import MemPalaceMemoryProvider, _store_triple_with_inverse, register

__all__ = [
    "MemPalaceMemoryProvider",
    "MemoryProvider",
    "register",
    "tool_error",
    "_store_triple_with_inverse",
]

__version__ = "1.0.0"
