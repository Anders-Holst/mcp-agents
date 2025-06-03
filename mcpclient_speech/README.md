# Generic speech-to-speech LLM-based MCP-client
A generic MCP-client which can connect to any Fastmcp server and then provides access to its tools via speech-based communication with an LLM.

### Prerequisites

You need to have Ollama installed, and to have downloaded a suitable LLM model.
Near the top of mcpclient_speech.py you can insert the correct model name.

You also need to install whisper.cpp and piper. Near the top of record.py is a path to the whisper.cpp directory which may be adjusted. In piperscript are the paths to piper and the piper models. Right now the code assumes that there are available piper voices for languages en, sv, de, fr, and es.

Before running this client, you need to start a selected MCP-server, with the tools to control some gadget. See its README for how.

### Run the server through this client 

Start this client like this, and then you can start chatting with your gadget.
```bash
uv run mcpclient_text.py
```

