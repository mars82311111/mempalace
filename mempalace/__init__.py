"""MemPalace — standalone AI memory system.

The highest-scoring AI memory system on GitHub (96.6% LongMemEval R@5).
Stores verbatim text in ChromaDB, temporal knowledge in SQLite.

Local-only: no API key, no network calls, all storage on-machine.
"""

from mempalace._compat import MemoryProvider, tool_error
from mempalace.core import (
    MemPalaceMemoryProvider,
    _store_triple_with_inverse,
    _bulk_add_episodes_to_chromadb,
    _read_wal,
    _truncate_wal,
    _update_health,
    _wal_path,
    _health_path,
    _start_wal_batcher,
    _stop_wal_batcher,
    register,
)
from mempalace.backup import (
    run_full_backup,
    start_backup_worker,
    stop_backup_worker,
    enqueue_incremental,
    BackupQueue,
    BackupWorker,
)

__all__ = [
    "MemPalaceMemoryProvider",
    "MemoryProvider",
    "register",
    "tool_error",
    "_store_triple_with_inverse",
    "_bulk_add_episodes_to_chromadb",
    "_read_wal",
    "_truncate_wal",
    "_update_health",
    "_wal_path",
    "_health_path",
    "_start_wal_batcher",
    "_stop_wal_batcher",
    "run_full_backup",
    "start_backup_worker",
    "stop_backup_worker",
    "enqueue_incremental",
    "BackupQueue",
    "BackupWorker",
]

__version__ = "1.1.0"
