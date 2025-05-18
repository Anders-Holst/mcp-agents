# Screenshot MCP Server
A FastMCP-based server that captures and serves screenshots of the primary monitor. T

### Starting the Server as SSE

You will need to use uv to run the example code.

```bash
uv run screenshot.py --transport sse
[05/18/25 21:36:06] INFO     Starting server "ScreenshotServer"...                                                                                           server.py:214
INFO:     Started server process [69692]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Test the server

```bash
uv run mcpclient.py
```

This will save the screenshot on disk.
