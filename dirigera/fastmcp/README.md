# Dirigera MCP Server (FastMCP version)

The Dirigera MCP Server is an IoT device management server that interfaces with the IKEA DIRIGERA platform. It allows users to manage and interact with various IoT devices, such as environment sensors, lights and outlets.

## Transports

| Transport | Use Case | Command |
|-----------|----------|---------|
| `stdio` | Local (Claude Desktop, MCP CLI) | `uv run dirigeramcp.py` |
| `streamable-http` | Remote (recommended) | `uv run dirigeramcp.py --transport streamable-http --host 0.0.0.0` |
| `sse` | Remote (legacy) | `uv run dirigeramcp.py --transport sse --host 0.0.0.0` |

## How to Run

### Local (stdio)
```bash
npx @modelcontextprotocol/inspector uv --directory . run dirigeramcp.py
```

### Remote Server (streamable-http - recommended)
```bash
uv run dirigeramcp.py --transport streamable-http --host 0.0.0.0 --port 8000
# Server available at http://<your-ip>:8000/mcp
```

### Remote Server (sse)
```bash
uv run dirigeramcp.py --transport sse --host 0.0.0.0 --port 8000
# Server available at http://<your-ip>:8000/sse
```

## Configuration

You will need a config file named dirigera_mcp_server_config.toml in the following format:

```toml
[dirigera]
host = '<your dirigera host>'
token = "<your dirigera token>"

```

In Claude and other tools you will need something like this in your config.json:

```json
{
  "mcpServers": {
    "dirigera": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/joakimeriksson/work/mcp-agents/dirigera/fastmcp",
        "run",
        "dirigeramcp.py"
      ]
    }
  }
}
```
