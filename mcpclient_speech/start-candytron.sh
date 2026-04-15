#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SESSION="candytron"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# Create new session in candytron_mcp dir with the server command pre-filled
tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR/../candytron_mcp"
tmux send-keys -t "$SESSION" "uv run python candytron_mcp.py --port 7999" ""

# Split horizontally, bottom pane in mcpclient_speech dir with client command pre-filled
tmux split-window -v -t "$SESSION" -c "$SCRIPT_DIR"
tmux send-keys -t "$SESSION" "uv run python mcpclient_speech_face.py" ""

# Focus top pane (candytron must start first)
tmux select-pane -t "$SESSION:.0"

# Attach
tmux attach -t "$SESSION"
