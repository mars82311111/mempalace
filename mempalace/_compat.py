"""Compatibility shims so MemPalace can run standalone without Hermes Agent."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class MemoryProvider(ABC):
    """Abstract base class for memory providers (standalone copy)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider."""

    @abstractmethod
    def is_available(self) -> bool:
        return False

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        return tool_error(f"Tool {tool_name} not implemented")

    def shutdown(self) -> None:
        pass


def tool_error(message, **extra) -> str:
    """Return a JSON error string for tool handlers."""
    result = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)
