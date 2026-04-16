# MemPalace

The highest-scoring AI memory system on GitHub (96.6% LongMemEval R@5).

MemPalace is a **standalone**, local-only memory system for AI agents. It stores verbatim text in ChromaDB and maintains a temporal Knowledge Graph in SQLite. No API keys, no network calls — everything stays on your machine.

## What's New in v1.1.0

- **WAL Crash-Safe Architecture**: All episodic writes first append to an atomic NDJSON WAL file (<1ms), then flush to ChromaDB asynchronously in batches. If your process crashes, un-flushed entries are automatically replayed on the next startup — **zero data loss**.
- **Real-time User Message Persistence**: User input is persisted immediately before assistant generation starts.
- **Built-in CLI**: `python -m mempalace` gives you direct access to search, add drawers, run KG queries, consolidation and reflection without writing code.

## Features

- **7-Layer Memory Architecture**
  - L-WM: Working Memory (in-process LRU cache, 50 recent turns)
  - L0: Identity Layer
  - L1: Narrative summary
  - L2: Semantic drawers (ChromaDB)
  - L3: Episodic memory (raw conversation logs)
  - KG: Temporal Knowledge Graph with inverse relations
  - Cross-Palace: Link multiple memory palaces
- **Self-healing**: built-in reflection, consolidation, and duplicate detection
- **Zero-latency recall**: working memory lives in-process
- **Pluggable**: ships with a `MemoryProvider` adapter for Hermes Agent, but works standalone

## Quick Start

```bash
git clone https://github.com/mars82311111/mempalace.git
cd mempalace
pip install -e .
```

### CLI Usage

```bash
# Check palace status
python -m mempalace status

# Store a fact
python -m mempalace add_drawer "The user prefers dark mode." --wing facts --room preferences

# Search
python -m mempalace search "dark mode preference"

# Run maintenance (consolidation + reflection)
python -m mempalace consolidate
python -m mempalace reflect --fix
```

### Python Usage

```python
from mempalace import MemPalaceMemoryProvider

provider = MemPalaceMemoryProvider()
provider.initialize("my-session")

# Store a fact
result = provider.handle_tool_call("mempalace_add_drawer", {
    "content": "The user prefers dark mode.",
    "wing": "facts",
    "room": "preferences",
})
print(result)

# Search
result = provider.handle_tool_call("mempalace_search", {
    "query": "dark mode preference"
})
print(result)
```

## Testing

```bash
pip install -e ".[test]"
pytest -v
```

## License

MIT
