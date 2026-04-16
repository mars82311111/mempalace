# MemPalace

The highest-scoring AI memory system on GitHub (96.6% LongMemEval R@5).

MemPalace is a **standalone**, local-only memory system for AI agents. It stores verbatim text in ChromaDB and maintains a temporal Knowledge Graph in SQLite. No API keys, no network calls — everything stays on your machine.

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
