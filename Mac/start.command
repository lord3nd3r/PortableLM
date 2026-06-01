#!/bin/bash
# ===================================================
#  Portable AI - Fast Web Chat (Mac)
# ===================================================

echo "==================================================="
echo "    Portable AI - Fast Web Chat Mode (Mac)"
echo "==================================================="
echo ""
echo "  Launches the AI engine + browser chat UI."
echo "  All chats auto-save to the USB drive."
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
USB_ROOT="$(dirname "$SCRIPT_DIR")"
SHARED_DIR="$USB_ROOT/Shared"
OLLAMA_RUNTIME="$SHARED_DIR/.ollama-runtime"
mkdir -p "$OLLAMA_RUNTIME"

# ---- Full portability: keep EVERYTHING on the USB ----
export OLLAMA_MODELS="$SHARED_DIR/models/ollama_data"
export OLLAMA_HOME="$OLLAMA_RUNTIME"
export OLLAMA_RUNNERS_DIR="$OLLAMA_RUNTIME/runners"
export OLLAMA_TMPDIR="$OLLAMA_RUNTIME/tmp"
export OLLAMA_ORIGINS="*"
export OLLAMA_HOST="127.0.0.1:11434"
mkdir -p "$OLLAMA_RUNTIME/runners" "$OLLAMA_RUNTIME/tmp"
# -------------------------------------------------------

PORTABLE_OLLAMA="$SHARED_DIR/bin/ollama-darwin"
PORTABLE_LLAMA="$SHARED_DIR/bin/llama-mac/llama-server"
SETTINGS_FILE="$SHARED_DIR/chat_data/settings.json"
OLLAMA_PID=""

# Read chatEngine setting (default: ollama)
CHAT_ENGINE="ollama"
if command -v python3 >/dev/null 2>&1 && [ -f "$SETTINGS_FILE" ]; then
    CHAT_ENGINE=$(python3 -c "import json,sys; d=json.load(open('$SETTINGS_FILE')); print(d.get('chatEngine','ollama'))" 2>/dev/null || echo "ollama")
fi

if [ "$CHAT_ENGINE" = "llama" ] && [ -f "$PORTABLE_LLAMA" ]; then
    chmod +x "$PORTABLE_LLAMA" 2>/dev/null
    xattr -d com.apple.quarantine "$PORTABLE_LLAMA" 2>/dev/null || true
    echo "[OK] Chat backend: llama.cpp (managed by chat server)"
else
    CHAT_ENGINE="ollama"
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "[OK] Ollama is already running - using existing instance."
    elif [ -f "$PORTABLE_OLLAMA" ]; then
        echo "Starting portable AI engine..."
        HOME="$OLLAMA_RUNTIME" "$PORTABLE_OLLAMA" serve &
        OLLAMA_PID=$!
        echo "Waiting for engine to initialize..."
        until curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
            sleep 1
        done
        echo "[OK] Engine is online!"
    elif command -v ollama > /dev/null 2>&1; then
        echo "Portable engine not found - starting system Ollama..."
        ollama serve &
        OLLAMA_PID=$!
        echo "Waiting for engine to initialize..."
        until curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
            sleep 1
        done
        echo "[OK] System Ollama is online!"
    else
        echo "==================================================="
        echo "  ERROR: No Ollama engine found!"
        echo "==================================================="
        echo ""
        echo "  No Ollama engine is running and none was found on"
        echo "  this system. To fix this, either:"
        echo "    1. Double-click 'Mac/install.command' to download"
        echo "       the portable engine, OR"
        echo "    2. Install Ollama system-wide from https://ollama.com"
        echo "       and make sure it is running before starting."
        echo ""
        read -n 1 -s -r -p "Press any key to exit..."
        exit 1
    fi
fi

echo ""
echo "==================================================="
echo "  AI ENGINE IS RUNNING"
echo "  Chat UI will open automatically."
echo "  Press Ctrl+C to shut down."
echo "==================================================="
echo ""

# Launch Python chat server using system Python (comes pre-installed on Mac)
if command -v python3 &> /dev/null; then
    python3 "$SHARED_DIR/chat_server.py"
elif command -v python &> /dev/null; then
    python "$SHARED_DIR/chat_server.py"
else
    echo "ERROR: Python not found. Please type 'brew install python' in terminal."
    exit 1
fi

# Cleanup
if [ -n "$OLLAMA_PID" ]; then
    kill -9 $OLLAMA_PID 2>/dev/null
fi
echo "Goodbye!"
