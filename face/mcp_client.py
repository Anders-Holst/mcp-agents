"""
MCP server configuration for the face agent.

Gives the LLM access to external tools (smart home, web search, etc.)
via MCP (Model Context Protocol) servers.

--- How to add a new MCP server ---

Option A: Config file (mcp_servers.json)

    {
        "servers": [
            {
                "name": "lights",
                "type": "sse",
                "url": "http://localhost:8000/sse"
            },
            {
                "name": "search",
                "type": "stdio",
                "command": "uv",
                "args": ["--directory", "../my_mcp", "run", "server.py"]
            }
        ]
    }

Option B: CLI flag

    python agent.py --mcp-server http://localhost:8000/sse

Server types:
  sse   — connect to an already-running MCP server over HTTP
  stdio — launch an MCP server as a subprocess (stdin/stdout)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio

logger = logging.getLogger("mcp_client")

DEFAULT_CONFIG = "mcp_servers.json"


def load_servers(
    config_path: Optional[str] = None,
    server_urls: Optional[list[str]] = None,
) -> list:
    """Load MCP servers from a config file and/or CLI URLs.

    Returns a list of pydantic-ai toolset objects, ready to pass
    to ``Agent(toolsets=[...])``.
    """
    servers = []

    # --- Config file ---
    path = Path(config_path or DEFAULT_CONFIG)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        for entry in data.get("servers", []):
            server = _create_server(entry)
            if server:
                servers.append(server)
        logger.info(f"Loaded {len(servers)} MCP server(s) from {path}")
    elif config_path:
        logger.warning(f"MCP config not found: {path}")

    # --- CLI URLs (SSE) ---
    for url in server_urls or []:
        logger.info(f"MCP server (CLI): SSE -> {url}")
        servers.append(MCPServerSSE(url=url))

    if servers:
        logger.info(f"Total MCP servers: {len(servers)}")
    return servers


def _create_server(entry: dict):
    """Create a single MCP server from a config dict."""
    name = entry.get("name", "unnamed")
    server_type = entry.get("type", "sse")

    if server_type == "sse":
        url = entry["url"]
        logger.info(f"MCP '{name}': SSE -> {url}")
        return MCPServerSSE(url=url)

    if server_type == "stdio":
        command = entry["command"]
        args = entry.get("args", [])
        env = entry.get("env")
        logger.info(f"MCP '{name}': stdio -> {command} {' '.join(args)}")
        return MCPServerStdio(command=command, args=args, env=env)

    logger.warning(f"Unknown MCP server type '{server_type}' for '{name}', skipping")
    return None
