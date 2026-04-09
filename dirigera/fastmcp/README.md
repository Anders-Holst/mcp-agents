# Dirigera MCP Server

An MCP server for controlling IKEA Dirigera smart home devices -- outlets, lights, and environment sensors.

Built with [FastMCP](https://github.com/jlowin/fastmcp) and the [dirigera](https://github.com/Leggin/dirigera) Python library.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- IKEA Dirigera Hub on your local network

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Get a Dirigera token

You need a JWT token to authenticate with your Dirigera Hub. Use the included helper:

```bash
uv run gen_token.py 192.168.1.200
```

This will send an authentication challenge to the hub. Press the action button on your Dirigera Hub when prompted, and the token will be printed.

### 3. Configure

Copy the example config and fill in your values:

```bash
cp config.example.toml config.toml
```

```toml
[dirigera]
host = "192.168.1.200"
token = "your-dirigera-token-here"
```

The server looks for `../config.toml` by default (one directory up). Override with `--config-path`:

```bash
uv run dirigeramcp.py --config-path ./config.toml
```

## Running the server

### Local (stdio) -- for Claude Desktop, Claude Code, MCP CLI

```bash
uv run dirigeramcp.py
```

### Remote (streamable-http) -- recommended for remote clients

```bash
uv run dirigeramcp.py --transport streamable-http --host 0.0.0.0 --port 8000
```

Server will be available at `http://<your-ip>:8000/mcp`.

### Remote (SSE) -- legacy

```bash
uv run dirigeramcp.py --transport sse --host 0.0.0.0 --port 8000
```

Server will be available at `http://<your-ip>:8000/sse`.

### With JWT authentication

For remote transports, you can enable JWT authentication:

```bash
uv run dirigeramcp.py --transport streamable-http --host 0.0.0.0 --auth
```

On first run with `--auth`, if no `[auth]` section exists in your config, the server auto-generates secrets and saves them. You can also set them manually in `config.toml`:

```toml
[auth]
jwt_secret = "your-hex-secret"
api_key = "your-hex-api-key"
expiry_hours = 24
```

To get a JWT token, POST to the token endpoint:

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-api-key"}'
```

To refresh an expiring token:

```bash
curl -X POST http://localhost:8000/auth/refresh \
  -H "Authorization: Bearer <your-jwt-token>"
```

## Available tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_environment_sensors` | List sensors with temperature, humidity, PM2.5, VOC, CO2 | -- |
| `get_outlets` | List outlets with power, voltage, current | -- |
| `get_lights` | List lights with brightness, color temp, hue, saturation, on/off | -- |
| `set_onoff` | Turn an outlet or light on/off by name | `name`, `is_on` |
| `set_light_level` | Set light brightness | `name`, `light_level` (int) |
| `set_light_color` | Set light color (if supported) | `name`, `color_saturation` (0.0-1.0), `color_hue` (0-360) |

## Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv --directory . run dirigeramcp.py
```

## Claude Desktop configuration

Add this to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dirigera": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-agents/dirigera/fastmcp",
        "run",
        "dirigeramcp.py"
      ]
    }
  }
}
```

## Command-line options

```
--config-path   Path to config TOML file (default: ../config.toml)
--host          Host to bind to (default: 127.0.0.1, use 0.0.0.0 for remote)
--port          Port to bind to (default: 8000)
--transport     stdio | sse | streamable-http (default: stdio)
--auth          Enable JWT authentication (remote transports only)
```
