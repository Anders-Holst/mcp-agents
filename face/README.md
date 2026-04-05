# Face Agent

A camera-based agent that sees, recognizes, and talks to people. It greets
known faces, asks strangers their name, responds to speech, and says goodbye
when someone leaves. An LLM (Ollama) drives the conversation, and MCP servers
can give it extra capabilities like controlling smart home devices.

## Architecture

```
 Camera frames              Microphone audio
      |                          |
      v                          v
 +--------------+        +-------------+
 | face_tracker |        | voice_input |
 |  detection   |        |  VAD        |
 |  recognition |        |  Whisper    |
 |  emotion     |        |  transcribe |
 |  tracking    |        +------+------+
 +------+-------+               |
        |                       |
        |  FaceEvents           |  speech text
        v                       v
 +------+-----------------------+------+
 |              agent                  |
 |                                     |
 |  _on_face_event    _on_heard_speech |
 |       |                  |          |
 |       v                  v          |
 |   [decide]           [respond]      |
 |       |                  |          |
 |       v                  v          |
 |  ConversationLLM  (Ollama + MCP)    |
 |       |                             |
 |       v                             |
 |  voice_output  (piper TTS)          |
 +-------------------------------------+
        |
        v
 +--------------+     +----------------+
 | people_memory|     | mcp_client     |
 | per-person   |     | MCP server     |
 | dialogues,   |     | config/loader  |
 | facts, prefs |     +----------------+
 +--------------+
```

## Agent decision logic

The agent is event-driven. Two inputs trigger decisions:

### Face events (from face_tracker)

```
_on_face_event(event)
  |
  |-- FACE_DISAPPEARED (known person)?
  |     -> say "Goodbye, <name>!"           (always, even when busy)
  |
  |-- busy?
  |     -> drop event                       (one interaction at a time)
  |
  |-- IDENTITY_CONFIRMED (recognized face)?
  |     -> greet if auto_greet and cooldown expired
  |
  |-- FACE_APPEARED (new track)?
  |     -> if known: greet (same cooldown logic)
  |     -> if unknown: tracked, later check_unknown_faces() asks their name
```

### Speech events (from voice_input via ContinuousListener)

```
_on_heard_speech(text)
  |
  |-- find who is in focus (primary face)
  |-- log dialogue to people_memory
  |-- LLM generates response (with person context + MCP tools)
  |-- speak response
```

### Busy gate

The agent does one thing at a time. A `_busy` flag prevents overlapping
interactions. Goodbye is the exception -- it runs even when busy since
it's a quick speak with no LLM call.

## Modules

| File | Purpose | Run standalone |
|------|---------|----------------|
| `face_tracker.py` | Detection, recognition, emotion, tracking, events | `pixi run vision` |
| `voice_input.py` | Mic monitoring, VAD, Whisper transcription | `pixi run listen` |
| `voice_output.py` | Piper TTS, speak text aloud | `pixi run speak` |
| `people_memory.py` | Per-person storage (dialogues, facts, prefs) | `pixi run people` |
| `mcp_client.py` | Load MCP server configs for the LLM | -- |
| `agent.py` | Decision loop tying it all together | `pixi run agent` |
| `main.py` | Legacy all-in-one UI (pre-agent refactor) | `pixi run run` |

## Quick start

```bash
# Install dependencies
pixi install

# Start Ollama with a model
ollama pull qwen3:8b

# Run the agent (camera + mic + speaker)
pixi run agent
```

## Adding MCP servers

MCP servers give the LLM access to external tools (lights, search, calendar, etc.).

### Option A: Config file

Create `mcp_servers.json`:

```json
{
    "servers": [
        {
            "name": "lights",
            "type": "sse",
            "url": "http://localhost:8000/sse"
        },
        {
            "name": "dirigera",
            "type": "stdio",
            "command": "uv",
            "args": ["--directory", "../dirigera/fastmcp", "run", "dirigeramcp.py"]
        }
    ]
}
```

The agent picks up `mcp_servers.json` automatically.

### Option B: CLI flags

```bash
pixi run agent --mcp-server http://localhost:8000/sse
pixi run agent --mcp-server http://localhost:8000/sse --mcp-server http://localhost:9000/sse
pixi run agent --mcp-config path/to/config.json
```

### Server types

- **sse** -- connect to an already-running MCP server over HTTP
- **stdio** -- launch an MCP server as a subprocess

## CLI options

```
--camera N           Camera index (default: 0)
--fps N              Target frame rate (default: 15)
--llm-model NAME     Ollama model (default: qwen3:8b)
--ollama-url URL     Ollama API URL (default: http://localhost:11434/v1)
--mcp-config PATH    MCP servers JSON config file
--mcp-server URL     MCP server SSE URL (repeatable)
--no-auto-greet      Don't auto-greet known faces
--no-auto-ask        Don't auto-ask unknown faces for their name
--en-voice NAME      Piper TTS voice (default: en_US-lessac-medium)
--db-dir DIR         Face database directory (default: known_faces)
--people-dir DIR     People memory directory (default: people)
```
