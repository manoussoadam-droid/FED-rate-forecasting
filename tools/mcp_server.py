#!/usr/bin/env python3
"""
MCP stdio server for corpus + analysis tools.
Run from project root with venv activated; point your MCP client at this script.

Example Cursor MCP config (conceptual):
  "command": "python",
  "args": ["tools/mcp_server.py"],
  "cwd": "/path/to/fomc_tools"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Install mcp: pip install mcp", file=sys.stderr)
    sys.exit(1)

from core.analysis_pipeline import analyze_text
from core.ingest import load_fomc, load_speaker, random_corpus_row

mcp = FastMCP("FOMC text tools")


@mcp.tool()
def analyze_fed_text(text: str) -> str:
    """Run full analysis pipeline on raw text; returns JSON string."""
    return json.dumps(analyze_text(text), indent=2, default=str)


@mcp.tool()
def random_corpus_document() -> str:
    """Return one random row from the combined FOMC + speaker pickle corpus."""
    fomc = load_fomc()
    speaker = load_speaker()
    return json.dumps(random_corpus_row(fomc, speaker), indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
