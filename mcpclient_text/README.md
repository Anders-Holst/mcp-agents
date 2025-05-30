# Generic LLM-based MCP-client
A generic MCP-client which can connect to any Fastmcp server and then provides access to its tools via text-based communication with an LLM.

### Prerequisites

You need to have Ollama installed, and to have downloaded a suitable LLM model.
Near the top of mcpclient_text.py you can insert the correct model name.

Before running this client, you need to start a selected MCP-server, with the tools to control some gadget. See its README for how.

### Run the server through this client 

Start this client like this, and then you can start chatting with your gadget.
```bash
uv run mcpclient_text.py
```

