"""MCP server: code repository ingestion, annotation, and retrieval."""

from .app import INSTRUCTIONS, main, mcp  # noqa: F401  (re-export for consumers)

__all__ = ["main", "mcp", "INSTRUCTIONS"]
__version__ = "0.1.0"
