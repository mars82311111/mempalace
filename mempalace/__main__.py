"""MemPalace CLI entry point for standalone usage."""

import argparse
import json
import sys
from pathlib import Path

from mempalace import MemPalaceMemoryProvider, run_full_backup


def cmd_status(provider):
    palace_path = Path(provider._palace_path)
    return {
        "palace_path": str(palace_path),
        "palace_exists": palace_path.exists(),
        "provider": provider.name,
        "available": provider.is_available(),
    }


def main():
    parser = argparse.ArgumentParser(prog="mempalace", description="MemPalace standalone AI memory system")
    parser.add_argument("--palace", default=str(Path.home() / ".mempalace_hermes"), help="Path to palace directory")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show palace status")
    wakeup = subparsers.add_parser("wakeup", help="Run wake-up sequence")
    wakeup.add_argument("--wing", default=None)

    search = subparsers.add_parser("search", help="Search drawers")
    search.add_argument("query")
    search.add_argument("--n_results", type=int, default=8)

    add_drawer = subparsers.add_parser("add_drawer", help="Add a memory drawer")
    add_drawer.add_argument("content")
    add_drawer.add_argument("--wing", default="facts")
    add_drawer.add_argument("--room", default="general")

    kg_query = subparsers.add_parser("kg_query", help="Query knowledge graph")
    kg_query.add_argument("entity")

    kg_add = subparsers.add_parser("kg_add", help="Add KG triple")
    kg_add.add_argument("subject")
    kg_add.add_argument("predicate")
    kg_add.add_argument("obj")

    consolidate = subparsers.add_parser("consolidate", help="Run consolidation")
    consolidate.add_argument("--dry_run", action="store_true")

    reflect = subparsers.add_parser("reflect", help="Run memory reflection")
    reflect.add_argument("--fix", action="store_true")

    episodes = subparsers.add_parser("episodes", help="List recent episodes")
    episodes.add_argument("--limit", type=int, default=10)

    backup = subparsers.add_parser("backup", help="Trigger an incremental cloud backup")
    backup.add_argument("--files", nargs="*", help="Specific files to back up")

    backup_full = subparsers.add_parser("backup_full", help="Run a full encrypted export and upload to GitHub")
    backup_full.add_argument("--output", default=None, help="Local export filepath (optional)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    provider = MemPalaceMemoryProvider()
    provider._palace_path = args.palace
    provider.initialize(session_id="mempalace_cli")

    if args.command == "status":
        result = cmd_status(provider)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "wakeup":
        print(provider.handle_tool_call("mempalace_wakeup", {"wing": args.wing}))

    elif args.command == "search":
        print(provider.handle_tool_call("mempalace_search", {
            "query": args.query,
            "n_results": args.n_results,
        }))

    elif args.command == "add_drawer":
        print(provider.handle_tool_call("mempalace_add_drawer", {
            "content": args.content,
            "wing": args.wing,
            "room": args.room,
        }))

    elif args.command == "kg_query":
        print(provider.handle_tool_call("mempalace_kg_query", {"entity": args.entity}))

    elif args.command == "kg_add":
        print(provider.handle_tool_call("mempalace_kg_add", {
            "subject": args.subject,
            "predicate": args.predicate,
            "object": args.obj,
        }))

    elif args.command == "consolidate":
        print(provider.handle_tool_call("mempalace_consolidate", {"dry_run": args.dry_run}))

    elif args.command == "reflect":
        print(provider.handle_tool_call("mempalace_memory_reflection", {"fix": args.fix}))

    elif args.command == "episodes":
        print(provider.handle_tool_call("mempalace_episodes", {"limit": args.limit}))

    elif args.command == "backup":
        from mempalace import enqueue_incremental
        files = args.files or []
        if not files:
            base = Path(args.palace)
            for name in ("knowledge_graph.sqlite3", "episodes.wal.ndjson", "health.json", "config.json", "identity.txt"):
                p = base / name
                if p.exists():
                    files.append(str(p))
        enqueue_incremental(files)
        print(json.dumps({"result": "Incremental backup enqueued", "files": files}, indent=2, ensure_ascii=False))

    elif args.command == "backup_full":
        from mempalace import _stop_wal_batcher, _start_wal_batcher, stop_backup_worker, start_backup_worker

        def _stop_all():
            _stop_wal_batcher(timeout=10.0)
            stop_backup_worker(timeout=10.0)

        def _start_all():
            _start_wal_batcher()
            start_backup_worker()

        result = run_full_backup(
            export_filepath=args.output,
            palace_path=Path(args.palace),
            feishu_alert=True,
            stop_workers_fn=_stop_all,
            start_workers_fn=_start_all,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    provider.shutdown()


if __name__ == "__main__":
    main()
