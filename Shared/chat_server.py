#!/usr/bin/env python3
"""
Portable AI Chat Server
=======================
A zero-dependency Python HTTP server that:
  1. Serves the FastChatUI.html web interface
  2. Saves/loads chat history as JSON files on the USB drive
  3. Proxies all Ollama API requests (eliminates CORS issues)

Works on Windows, macOS, and Linux without installing anything.
"""

import http.server
import json
import os
import sys
import urllib.request
import urllib.error
import threading
import webbrowser
import time
import platform
import ctypes
import logging
import logging.handlers
import queue
import uuid
import subprocess
import base64
import re
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# ── Image Generation Job Tracking ──────────────────────────────
IMAGE_JOBS = {}
IMAGE_JOBS_LOCK = threading.RLock()
IMAGE_JOB_MAX_AGE_SECS = 3600  # Cleanup old jobs after 1 hour

def _register_image_job(job_id):
    with IMAGE_JOBS_LOCK:
        IMAGE_JOBS[job_id] = {
            "status": "starting",
            "step": 0,
            "total_steps": 0,
            "elapsed_ms": 0,
            "eta_ms": None,
            "started_at": time.time(),
            "updated_at": time.time(),
            "error": None,
            "image_b64": None,
            "params": {},
        }

def _update_image_job(job_id, **kwargs):
    with IMAGE_JOBS_LOCK:
        if job_id in IMAGE_JOBS:
            IMAGE_JOBS[job_id].update(kwargs)
            IMAGE_JOBS[job_id]["updated_at"] = time.time()

def _get_image_job(job_id):
    with IMAGE_JOBS_LOCK:
        return IMAGE_JOBS.get(job_id, {}).copy()

def _cleanup_old_image_jobs():
    now = time.time()
    with IMAGE_JOBS_LOCK:
        stale = [jid for jid, j in IMAGE_JOBS.items() if now - j["updated_at"] > IMAGE_JOB_MAX_AGE_SECS]
        for jid in stale:
            del IMAGE_JOBS[jid]

# ── Engine Startup Job Tracking ────────────────────────────────
ENGINE_STARTUP_JOBS = {}
ENGINE_STARTUP_JOBS_LOCK = threading.RLock()

def _register_engine_startup_job(job_id, engine, model=""):
    with ENGINE_STARTUP_JOBS_LOCK:
        ENGINE_STARTUP_JOBS[job_id] = {
            "status": "starting",
            "engine": engine,
            "model": model,
            "phase": "initializing",
            "progress_pct": 0,
            "message": "Starting engine...",
            "started_at": time.time(),
            "updated_at": time.time(),
            "error": None,
        }

def _update_engine_startup_job(job_id, **kwargs):
    with ENGINE_STARTUP_JOBS_LOCK:
        if job_id in ENGINE_STARTUP_JOBS:
            ENGINE_STARTUP_JOBS[job_id].update(kwargs)
            ENGINE_STARTUP_JOBS[job_id]["updated_at"] = time.time()

def _get_engine_startup_job(job_id):
    with ENGINE_STARTUP_JOBS_LOCK:
        return ENGINE_STARTUP_JOBS.get(job_id, {}).copy()

def _cleanup_old_engine_startup_jobs():
    now = time.time()
    with ENGINE_STARTUP_JOBS_LOCK:
        stale = [jid for jid, j in ENGINE_STARTUP_JOBS.items() if now - j["updated_at"] > 300]
        for jid in stale:
            del ENGINE_STARTUP_JOBS[jid]

# ── Model Pull Job Tracking ────────────────────────────────────
MODEL_PULL_JOBS = {}
MODEL_PULL_JOBS_LOCK = threading.RLock()

def _register_model_pull_job(job_id, model_name):
    with MODEL_PULL_JOBS_LOCK:
        MODEL_PULL_JOBS[job_id] = {
            "status": "starting",
            "model": model_name,
            "progress_pct": 0,
            "message": "Starting download...",
            "downloaded": "",
            "total": "",
            "started_at": time.time(),
            "updated_at": time.time(),
            "error": None,
        }

def _update_model_pull_job(job_id, **kwargs):
    with MODEL_PULL_JOBS_LOCK:
        if job_id in MODEL_PULL_JOBS:
            MODEL_PULL_JOBS[job_id].update(kwargs)
            MODEL_PULL_JOBS[job_id]["updated_at"] = time.time()

def _get_model_pull_job(job_id):
    with MODEL_PULL_JOBS_LOCK:
        return MODEL_PULL_JOBS.get(job_id, {}).copy()

def _cleanup_old_model_pull_jobs():
    now = time.time()
    with MODEL_PULL_JOBS_LOCK:
        stale = [jid for jid, j in MODEL_PULL_JOBS.items() if now - j["updated_at"] > 600]
        for jid in stale:
            del MODEL_PULL_JOBS[jid]

# ── HuggingFace Download Job Tracking ──────────────────────────
HF_DOWNLOAD_JOBS = {}
HF_DOWNLOAD_JOBS_LOCK = threading.RLock()

def _register_hf_download_job(job_id, url, filename, model_type):
    with HF_DOWNLOAD_JOBS_LOCK:
        HF_DOWNLOAD_JOBS[job_id] = {
            "status": "starting",
            "url": url,
            "filename": filename,
            "model_type": model_type,  # "gguf" or "safetensors"
            "progress_pct": 0,
            "message": "Starting download...",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "started_at": time.time(),
            "updated_at": time.time(),
            "error": None,
        }

def _update_hf_download_job(job_id, **kwargs):
    with HF_DOWNLOAD_JOBS_LOCK:
        if job_id in HF_DOWNLOAD_JOBS:
            HF_DOWNLOAD_JOBS[job_id].update(kwargs)
            HF_DOWNLOAD_JOBS[job_id]["updated_at"] = time.time()

def _get_hf_download_job(job_id):
    with HF_DOWNLOAD_JOBS_LOCK:
        return HF_DOWNLOAD_JOBS.get(job_id, {}).copy()

def _format_bytes(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"

def _run_hf_download(job_id, url, dest_path):
    """Download a file from HuggingFace with progress tracking."""
    try:
        _update_hf_download_job(job_id, message="Connecting...")
        
        req = urllib.request.Request(url, headers={"User-Agent": "PortableLM/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            total = int(response.headers.get('Content-Length', 0))
            _update_hf_download_job(job_id, total_bytes=total, 
                                    message=f"Downloading {_format_bytes(total)}...")
            
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            last_update = time.time()
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress every 0.5s to avoid spam
                    now = time.time()
                    if now - last_update > 0.5:
                        pct = int((downloaded / total) * 100) if total else 0
                        _update_hf_download_job(job_id, 
                            progress_pct=pct,
                            downloaded_bytes=downloaded,
                            message=f"{_format_bytes(downloaded)} / {_format_bytes(total)}")
                        last_update = now
        
        _update_hf_download_job(job_id, status="complete", progress_pct=100,
                                message="Download complete!")
    except Exception as e:
        # Clean up partial file
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except:
                pass
        _update_hf_download_job(job_id, status="failed", error=str(e))

# Optional: psutil for hardware stats (graceful fallback to native APIs if not installed)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Configuration ──────────────────────────────────────────────
CHAT_SERVER_PORT = 3333
OLLAMA_HOST = "http://127.0.0.1:11434"

def _find_free_port(start=8080, end=8200):
    """Return the first TCP port in [start, end] not already bound."""
    import socket
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    # Whole range is occupied (e.g. start=8080 is already bound on this host).
    # Let the OS assign an ephemeral port rather than returning a port we
    # already know is taken.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

LLAMA_PORT = _find_free_port()
LLAMA_HOST = f"http://127.0.0.1:{LLAMA_PORT}"

# Active engine is loaded from settings after SETTINGS_FILE is defined.
# Default here; overwritten in main() once settings are read.
ACTIVE_ENGINE = "ollama"   # "ollama" | "llama"
ACTIVE_ENGINE_LOCK = threading.RLock()

def _get_chat_host():
    """Return the base URL for the active chat engine."""
    with ACTIVE_ENGINE_LOCK:
        return LLAMA_HOST if ACTIVE_ENGINE == "llama" else OLLAMA_HOST

def _set_active_engine(engine):
    global ACTIVE_ENGINE
    with ACTIVE_ENGINE_LOCK:
        ACTIVE_ENGINE = engine if engine in ("ollama", "llama") else "ollama"

# Always resolve paths relative to THIS script's location (the USB drive)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Text-to-Speech (piper) ─────────────────────────────────────
def _find_piper_binary():
    """Locate the piper text-to-speech binary for this platform."""
    plat = platform.system()
    bin_dir = os.path.join(SCRIPT_DIR, "bin")
    if plat == "Windows":
        candidates = [os.path.join(bin_dir, "piper-windows", "piper.exe")]
    elif plat == "Linux":
        candidates = [os.path.join(bin_dir, "piper-linux", "piper")]
    else:  # Darwin / macOS
        candidates = [os.path.join(bin_dir, "piper-mac", "piper")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

PIPER_BINARY = _find_piper_binary()
PIPER_VOICES_DIR = os.path.join(SCRIPT_DIR, "models", "voices")
# Piper is roughly 10x realtime on CPU, so synthesis is done inline rather than
# through the background-job pattern used for image generation.
TTS_MAX_CHARS = 4000

VOICE_CATALOG_FILE = os.path.join(SCRIPT_DIR, "config", "voices.json")

# Voice download job tracking, same shape as the other background jobs.
VOICE_JOBS = {}
VOICE_JOBS_LOCK = threading.RLock()

def _register_voice_job(job_id, voice_id):
    with VOICE_JOBS_LOCK:
        VOICE_JOBS[job_id] = {
            "status": "starting", "voice": voice_id, "progress_pct": 0,
            "message": "Starting download...", "downloaded_bytes": 0,
            "total_bytes": 0, "started_at": time.time(),
            "updated_at": time.time(), "error": None,
        }

def _update_voice_job(job_id, **kwargs):
    with VOICE_JOBS_LOCK:
        if job_id in VOICE_JOBS:
            VOICE_JOBS[job_id].update(kwargs)
            VOICE_JOBS[job_id]["updated_at"] = time.time()

def _get_voice_job(job_id):
    with VOICE_JOBS_LOCK:
        return VOICE_JOBS.get(job_id, {}).copy()

def _load_voice_catalog():
    """Curated list of downloadable voices, or empty if the catalog is missing."""
    try:
        with open(VOICE_CATALOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = data.get("base_url", "")
        out = []
        for v in data.get("voices", []):
            if not v.get("id") or not v.get("onnx_path"):
                continue
            out.append({
                "id": v["id"],
                "label": v.get("label", ""),
                "description": v.get("description", ""),
                "quality": v.get("quality", ""),
                "size_mb": v.get("size_mb", 0),
                "onnx_url": base + v["onnx_path"],
                "config_url": base + v.get("config_path", v["onnx_path"] + ".json"),
                "onnx_bytes": v.get("onnx_bytes", 0),
            })
        return out
    except Exception:
        return []

def _run_voice_download(job_id, entry):
    """Fetch a voice's .onnx and .onnx.json into the voices folder."""
    os.makedirs(PIPER_VOICES_DIR, exist_ok=True)
    onnx_path = os.path.join(PIPER_VOICES_DIR, entry["id"] + ".onnx")
    cfg_path = onnx_path + ".json"
    try:
        # Config first: it is tiny, and a voice whose .onnx exists without it is
        # skipped by _find_piper_voices(), so this ordering never leaves a
        # half-installed voice visible in the picker.
        _update_voice_job(job_id, message="Fetching voice config...")
        req = urllib.request.Request(entry["config_url"], headers={"User-Agent": "PortableLM/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(cfg_path, "wb") as f:
            f.write(r.read())

        _update_voice_job(job_id, message="Downloading voice...")
        req = urllib.request.Request(entry["onnx_url"], headers={"User-Agent": "PortableLM/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            total = int(r.headers.get("Content-Length", 0)) or entry.get("onnx_bytes", 0)
            _update_voice_job(job_id, total_bytes=total)
            done = 0
            last = 0.0
            with open(onnx_path, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 0.5:
                        pct = int(done / total * 100) if total else 0
                        _update_voice_job(
                            job_id, progress_pct=pct, downloaded_bytes=done,
                            message=f"{_format_bytes(done)} / {_format_bytes(total)}",
                        )
                        last = now
        if entry.get("onnx_bytes") and os.path.getsize(onnx_path) < entry["onnx_bytes"] * 0.9:
            raise RuntimeError("Downloaded voice looks incomplete.")
        _update_voice_job(job_id, status="complete", progress_pct=100, message="Voice installed!")
    except Exception as e:
        # Remove both halves so a failed download never leaves a broken voice.
        for p in (onnx_path, cfg_path):
            try:
                os.remove(p)
            except Exception:
                pass
        _update_voice_job(job_id, status="failed", error=str(e))

def _find_piper_voices():
    """Return every installed piper voice (.onnx paired with its .onnx.json)."""
    found = []
    if os.path.isdir(PIPER_VOICES_DIR):
        for fname in sorted(os.listdir(PIPER_VOICES_DIR)):
            if not fname.lower().endswith(".onnx"):
                continue
            path = os.path.join(PIPER_VOICES_DIR, fname)
            # A voice without its config cannot be loaded, so don't offer it.
            if not os.path.isfile(path + ".json"):
                continue
            stem = os.path.splitext(fname)[0]
            found.append({
                "file": fname,
                "path": path,
                "name": stem.replace("_", " ").replace("-", " "),
                "id": stem,
            })
    return found

# ── Speech-to-Text (whisper.cpp) ───────────────────────────────
def _find_whisper_binary():
    """Locate the whisper.cpp CLI for this platform.

    Upstream publishes prebuilt binaries for Linux and Windows only; on macOS
    this stays None unless the user drops a self-built whisper-cli into
    Shared/bin/whisper-mac/, which is picked up here automatically.
    """
    plat = platform.system()
    bin_dir = os.path.join(SCRIPT_DIR, "bin")
    if plat == "Windows":
        candidates = [os.path.join(bin_dir, "whisper-windows", "whisper-cli.exe"),
                      os.path.join(bin_dir, "whisper-windows", "main.exe")]
    elif plat == "Linux":
        candidates = [os.path.join(bin_dir, "whisper-linux", "whisper-cli"),
                      os.path.join(bin_dir, "whisper-linux", "main")]
    else:  # Darwin / macOS
        candidates = [os.path.join(bin_dir, "whisper-mac", "whisper-cli"),
                      os.path.join(bin_dir, "whisper-mac", "main")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

WHISPER_BINARY = _find_whisper_binary()
WHISPER_MODELS_DIR = os.path.join(SCRIPT_DIR, "models", "whisper")
# Uploaded audio is short spoken input, not a file transfer. 25 MB is roughly
# 13 minutes of 16 kHz mono PCM — far more than any single utterance needs.
STT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

def _find_whisper_models():
    """Return installed whisper ggml model files, smallest (fastest) first."""
    found = []
    if os.path.isdir(WHISPER_MODELS_DIR):
        for fname in sorted(os.listdir(WHISPER_MODELS_DIR)):
            if not fname.lower().endswith(".bin"):
                continue
            path = os.path.join(WHISPER_MODELS_DIR, fname)
            try:
                size = os.path.getsize(path)
            except Exception:
                continue
            found.append({"file": fname, "path": path, "size": size})
    found.sort(key=lambda m: m["size"])
    return found

def _transcribe_audio(wav_bytes, model_path):
    """Run whisper.cpp over WAV bytes and return the transcript. Raises on failure."""
    if not WHISPER_BINARY or not os.path.isfile(WHISPER_BINARY):
        raise RuntimeError("Speech recognition engine not installed. Please run the installer.")
    if not model_path or not os.path.isfile(model_path):
        raise RuntimeError("No speech recognition model found. Please run the installer.")

    bin_dir = os.path.dirname(os.path.abspath(WHISPER_BINARY))
    env = os.environ.copy()
    if platform.system() != "Windows":
        key = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
        existing = env.get(key, "")
        env[key] = bin_dir + ((os.pathsep + existing) if existing else "")

    tmp_dir = os.path.join(SCRIPT_DIR, "chat_data", "stt_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    in_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.wav")
    try:
        with open(in_path, "wb") as f:
            f.write(wav_bytes)
        cmd = [
            WHISPER_BINARY,
            "--model", model_path,
            "--file", in_path,
            "--language", "en",
            "--no-timestamps",
            "--no-prints",
            "--threads", str(max(1, min(8, (os.cpu_count() or 4)))),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, timeout=300)
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="ignore").strip()[:200]
            raise RuntimeError(f"Transcription failed. {detail}")
        # With --no-prints, whisper writes only the transcript to stdout;
        # progress and backend chatter go to stderr.
        text = proc.stdout.decode("utf-8", errors="ignore").strip()
        # Whisper marks non-speech with bracketed all-caps tokens such as
        # [BLANK_AUDIO], [ SILENCE ] or [MUSIC]. Only all-caps bracketed runs
        # are stripped: matching loosely (or inside parentheses) would delete
        # real speech like "(that sounds good)", and leaving a stray marker in
        # is far less damaging than dropping words the user actually said.
        text = re.sub(r"\[\s*[A-Z][A-Z0-9_ ]*\s*\]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    finally:
        try:
            os.remove(in_path)
        except Exception:
            pass

# Emoji and decorative symbol ranges. espeak-ng pronounces these by their
# Unicode names ("smiling face with smiling eyes"), which roughly doubles the
# length of a short reply and sounds absurd. Deliberately surgical: degree
# signs, dashes, curly quotes, ellipses and (c)/(r)/(tm) are all left alone
# because they either matter ("20°C") or already read correctly.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoji blocks: pictographs, faces, transport, flags
    "←-⇿"          # arrows
    "☀-⛿"          # miscellaneous symbols
    "✀-➿"          # dingbats: check marks, crosses, stars
    "⬀-⯿"          # misc symbols and arrows
    "︀-️"          # variation selectors
    "‍"                 # zero-width joiner (emoji sequences)
    "⃣"                 # combining enclosing keycap
    "•"                 # bullet
    "]"
)

# Text emoticons. Anchored to whitespace on the left and whitespace or
# sentence punctuation on the right so timestamps ("9:30"), ratios ("3:4")
# and code are never mangled.
# Note "8" is deliberately absent from the leading character class: "8)" is a
# sunglasses emoticon but far more often a numbered list item, and silently
# eating list numbers is worse than leaving one rare emoticon in.
_EMOTICON_RE = re.compile(
    r"(?<![^\s])"                       # start of string or preceded by space
    r"(?:[:;=][-~^]?[)(\]\[DdPpOo3><|/\\]{1,2}|<3|\^_\^|>:\(|:'\()"
    r"(?=[\s.,!?;:]|$)"                 # followed by space, punctuation, or end
)

def _strip_unspeakable_symbols(text):
    """Remove emoji and text emoticons before synthesis."""
    text = _EMOJI_RE.sub(" ", text)
    return _EMOTICON_RE.sub(" ", text)

def _text_for_speech(text):
    """Strip markup from model output so it reads naturally aloud.

    Spoken output has different needs than rendered output: literal asterisks,
    backtick fences and raw URLs are unbearable read out, and reasoning traces
    are not part of the answer at all. Anything unspeakable is either dropped
    or replaced with a short spoken stand-in.
    """
    if not text:
        return ""
    s = str(text)
    # Reasoning traces are internal, not the reply. Drop closed and unclosed alike.
    s = re.sub(r"<think>[\s\S]*?</think>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<think>[\s\S]*$", " ", s, flags=re.IGNORECASE)
    # Fenced code: announce rather than read out every brace and semicolon.
    s = re.sub(r"```[\s\S]*?```", " (code block omitted) ", s)
    s = re.sub(r"```[\s\S]*$", " (code block omitted) ", s)
    # Images before links, so alt text doesn't survive as a stray phrase.
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", " ", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)   # keep link text, drop target
    s = re.sub(r"<[^>]+>", " ", s)                    # any remaining HTML
    s = re.sub(r"https?://\S+", " (link) ", s)        # bare URLs
    s = re.sub(r"`([^`]*)`", r"\1", s)                # inline code: keep content
    s = re.sub(r"^\s{0,3}#{1,6}\s*", "", s, flags=re.MULTILINE)  # heading marks
    s = re.sub(r"^\s{0,3}>\s?", "", s, flags=re.MULTILINE)       # block quotes
    s = re.sub(r"^\s{0,3}[-*_]{3,}\s*$", " ", s, flags=re.MULTILINE)  # rules
    s = re.sub(r"^(\s*)[-*+]\s+", r"\1", s, flags=re.MULTILINE)  # bullets
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)          # bold
    s = re.sub(r"(?<!\w)[*_]([^*_\n]+)[*_](?!\w)", r"\1", s)  # italics
    s = _strip_unspeakable_symbols(s)
    s = re.sub(r"^\s*\|?[\s:|-]*\|[\s:|-]*\|?\s*$", " ", s, flags=re.MULTILINE)  # table separator row
    s = s.replace("|", " ")                            # remaining table pipes
    s = re.sub(r"(?<!\w)-{3,}(?!\w)", " ", s)          # leftover dash runs
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def _synthesize_speech(text, voice_path, length_scale=1.0):
    """Run piper and return WAV bytes. Raises on failure."""
    if not PIPER_BINARY or not os.path.isfile(PIPER_BINARY):
        raise RuntimeError("Text-to-speech engine not installed. Please run the installer.")
    if not voice_path or not os.path.isfile(voice_path):
        raise RuntimeError("No voice model found. Please run the installer.")

    bin_dir = os.path.dirname(os.path.abspath(PIPER_BINARY))
    env = os.environ.copy()
    if platform.system() != "Windows":
        # piper ships its own onnxruntime/espeak libs next to the binary
        key = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
        existing = env.get(key, "")
        env[key] = bin_dir + ((os.pathsep + existing) if existing else "")

    # piper writes the WAV to a file; '-' (stdout) mixes with its logging on
    # some builds, so use a temp file and read it back.
    tmp_dir = os.path.join(SCRIPT_DIR, "chat_data", "tts_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.wav")
    cmd = [
        PIPER_BINARY,
        "--model", voice_path,
        "--output_file", out_path,
        "--length_scale", str(length_scale),
        "--quiet",
    ]
    try:
        # Text goes over stdin, never the command line: no shell, no injection,
        # and no length limit from the OS argument cap.
        proc = subprocess.run(
            cmd, input=text.encode("utf-8"), env=env,
            capture_output=True, timeout=120,
        )
        if proc.returncode != 0 or not os.path.isfile(out_path):
            detail = proc.stderr.decode("utf-8", errors="ignore").strip()[:200]
            raise RuntimeError(f"Speech synthesis failed. {detail}")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(out_path)
        except Exception:
            pass

# ── Stable Diffusion Engine Paths ──────────────────────────────
def _find_sd_binary():
    """Locate the stable-diffusion.cpp CLI binary for this platform."""
    plat = platform.system()
    bin_dir = os.path.join(SCRIPT_DIR, "bin")
    candidates = []
    if plat == "Windows":
        sd_dir = os.path.join(bin_dir, "sd-windows")
        candidates = [os.path.join(sd_dir, "sd.exe"), os.path.join(sd_dir, "sd-cli.exe")]
    elif plat == "Linux":
        sd_dir = os.path.join(bin_dir, "sd-linux")
        candidates = [os.path.join(sd_dir, "sd"), os.path.join(sd_dir, "sd-cli")]
    else:  # Darwin / macOS
        sd_dir = os.path.join(bin_dir, "sd-mac")
        candidates = [os.path.join(sd_dir, "sd"), os.path.join(sd_dir, "sd-cli")]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

SD_BINARY = _find_sd_binary()

def _find_sd_models():
    """Return a list of dicts for every .safetensors file found in the models folder."""
    models_dir = os.path.join(SCRIPT_DIR, "models")
    found = []
    if os.path.isdir(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if fname.lower().endswith(".safetensors"):
                found.append({
                    "file": fname,
                    "path": os.path.join(models_dir, fname),
                    "name": os.path.splitext(fname)[0].replace("_", " ").replace("-", " "),
                })
    return found

SD_MODELS = _find_sd_models()
SD_MODEL = SD_MODELS[0]["path"] if SD_MODELS else None
SD_ENABLED = SD_BINARY is not None and len(SD_MODELS) > 0

_SD_BINARY_TEST_CACHE = {"result": None, "checked_at": 0.0}
_SD_BINARY_TEST_LOCK = threading.Lock()
_SD_BINARY_TEST_TTL = 30  # seconds

def _test_sd_binary():
    """Test if the SD binary can actually execute (catches missing VC++ runtime on Windows).

    Cached briefly since this spawns a subprocess and engine status is polled
    frequently, while the binary's ability to execute rarely changes.
    """
    now = time.time()
    with _SD_BINARY_TEST_LOCK:
        cached = _SD_BINARY_TEST_CACHE["result"]
        if cached is not None and now - _SD_BINARY_TEST_CACHE["checked_at"] < _SD_BINARY_TEST_TTL:
            return cached

    if not SD_BINARY or not os.path.isfile(SD_BINARY):
        result = False
    else:
        try:
            proc_result = subprocess.run([SD_BINARY, "-h"], capture_output=True, timeout=10)
            # sd -h returns non-zero but prints help; just check it doesn't crash with DLL errors
            stderr = proc_result.stderr.decode('utf-8', errors='ignore').lower()
            result = not ('dll' in stderr or 'vcruntime' in stderr or 'msvcp' in stderr)
        except Exception:
            result = False

    with _SD_BINARY_TEST_LOCK:
        _SD_BINARY_TEST_CACHE["result"] = result
        _SD_BINARY_TEST_CACHE["checked_at"] = now
    return result

def _detect_sd_backend():
    """Probe which GPU backend this sd-cli binary was compiled with.

    Tries each candidate by running the binary with --backend <name> against a
    dummy path. The backend validity check happens before model loading, so if
    the binary returns 'was not found' the backend is not compiled in. Any other
    exit (model error, etc.) means the backend is supported.

    Returns (backend, rng) where backend is None for CPU-only.
    """
    if not SD_BINARY or not os.path.isfile(SD_BINARY):
        return None, 'cpu'

    sd_dir = os.path.dirname(os.path.abspath(SD_BINARY))
    env = os.environ.copy()
    existing_ldpath = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = sd_dir + (":" + existing_ldpath if existing_ldpath else "")

    # Candidate order: prefer CUDA (fastest), then HIP (AMD), then Vulkan
    # rng=cuda only makes sense with a CUDA build
    candidates = [('cuda0', 'cuda'), ('hip0', 'cpu'), ('vulkan0', 'cpu')]

    for backend, rng in candidates:
        try:
            r = subprocess.run(
                [SD_BINARY, '--backend', backend, '-m', os.devnull,
                 '-p', 'probe', '-o', os.devnull],
                capture_output=True, text=True, env=env, timeout=8
            )
            combined = (r.stdout + r.stderr).lower()
            if 'was not found' in combined or 'backend config failed' in combined:
                continue  # this backend is not compiled in
            # Any other result (model error, etc.) means the backend exists
            return backend, rng
        except Exception:
            continue

    return None, 'cpu'

SD_BACKEND, SD_RNG = _detect_sd_backend()

# Ollama binary path for engine management
OLLAMA_BIN = None
if platform.system() == "Windows":
    OLLAMA_BIN = os.path.join(SCRIPT_DIR, "bin", "ollama-windows.exe")
elif platform.system() == "Linux":
    OLLAMA_BIN = os.path.join(SCRIPT_DIR, "bin", "ollama-linux")
else:
    OLLAMA_BIN = os.path.join(SCRIPT_DIR, "bin", "ollama-darwin")

# llama.cpp server binary
def _find_llama_bin():
    """Locate the llama-server binary for this platform."""
    plat = platform.system()
    bin_dir = os.path.join(SCRIPT_DIR, "bin")
    if plat == "Windows":
        candidates = [
            os.path.join(bin_dir, "llama-windows", "llama-server.exe"),
        ]
    elif plat == "Linux":
        candidates = [
            os.path.join(bin_dir, "llama-linux", "llama-server"),
        ]
    else:
        candidates = [
            os.path.join(bin_dir, "llama-mac", "llama-server"),
        ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

LLAMA_BIN = _find_llama_bin()

def _find_gguf_models():
    """Return list of dicts for every selectable .gguf chat model.

    Skips mmproj projector files: they sit alongside vision models in the same
    folder but are not chat models on their own, and llama-server refuses them
    with "CLIP cannot be used as main model, use it with --mmproj instead".
    """
    models_dir = os.path.join(SCRIPT_DIR, "models")
    found = []
    if os.path.isdir(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if fname.lower().endswith(".gguf") and "mmproj" not in fname.lower():
                found.append({
                    "file": fname,
                    "path": os.path.join(models_dir, fname),
                    "name": os.path.splitext(fname)[0].replace("_", " ").replace("-", " "),
                })
    return found

# Track the llama-server subprocess so we can kill it cleanly
_LLAMA_PROC = None
_LLAMA_PROC_LOCK = threading.RLock()

# Track the ollama serve subprocess so we can kill it cleanly
_OLLAMA_PROC = None
_OLLAMA_PROC_LOCK = threading.RLock()

CHATS_DIR = os.path.join(SCRIPT_DIR, "chat_data")
CHATS_FILE = os.path.join(CHATS_DIR, "chats.json")
SETTINGS_FILE = os.path.join(CHATS_DIR, "settings.json")
HTML_FILE = os.path.join(SCRIPT_DIR, "FastChatUI.html")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "chat_server.log")
LOG_MODE_ERRORS_ONLY = "errors_only"
LOG_MODE_ALL = "all"
DEFAULT_LOG_MODE = LOG_MODE_ERRORS_ONLY

# Lock shared file persistence paths to avoid concurrent write corruption.
DATA_FILE_LOCK = threading.RLock()
LOG_MODE_LOCK = threading.RLock()
ACTIVE_LOG_MODE = DEFAULT_LOG_MODE

# ── Pure-Python Hardware Stats (no psutil needed) ──────────────
_cpu_times_last = None  # (idle, total) from previous sample

def _get_hw_stats():
    """Return (cpu_percent, ram_percent) using only stdlib / ctypes."""
    global _cpu_times_last  # must be at top of function, before any branch uses it

    if HAS_PSUTIL:
        cpu = round(psutil.cpu_percent(interval=0.25), 1)
        ram = round(psutil.virtual_memory().percent, 1)
        return cpu, ram

    plat = platform.system()

    # ── Windows ──────────────────────────────────────────────────
    if plat == "Windows":
        # RAM via GlobalMemoryStatusEx
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        msx = MEMORYSTATUSEX()
        msx.dwLength = ctypes.sizeof(msx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(msx))
        ram = float(msx.dwMemoryLoad)

        # CPU via GetSystemTimes (idle/kernel/user tick counts)
        FILETIME = ctypes.c_ulonglong
        idle, kern, user = FILETIME(), FILETIME(), FILETIME()
        ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        idle_v = idle.value
        total_v = kern.value + user.value
        if _cpu_times_last is None:
            # First call — sleep briefly and sample again
            time.sleep(0.25)
            idle2, kern2, user2 = FILETIME(), FILETIME(), FILETIME()
            ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle2), ctypes.byref(kern2), ctypes.byref(user2))
            d_idle  = idle2.value - idle_v
            d_total = (kern2.value + user2.value) - total_v
            _cpu_times_last = (idle2.value, kern2.value + user2.value)
        else:
            prev_idle, prev_total = _cpu_times_last
            d_idle  = idle_v  - prev_idle
            d_total = total_v - prev_total
            _cpu_times_last = (idle_v, total_v)

        cpu = round((1.0 - d_idle / max(d_total, 1)) * 100.0, 1)
        cpu = max(0.0, min(100.0, cpu))
        return cpu, ram

    # ── Linux ─────────────────────────────────────────────────────
    elif plat == "Linux":
        # RAM
        ram = 0.0
        try:
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem[parts[0].rstrip(":")] = int(parts[1])
            total = mem.get("MemTotal", 1)
            avail = mem.get("MemAvailable", total)
            ram = round((1 - avail / total) * 100, 1)
        except Exception:
            pass
        # CPU via /proc/stat delta
        cpu = 0.0
        try:
            def read_cpu():
                with open("/proc/stat") as f:
                    parts = f.readline().split()
                vals = [int(x) for x in parts[1:]]
                idle = vals[3]
                total = sum(vals)
                return idle, total
            if _cpu_times_last is None:
                i1, t1 = read_cpu()
                time.sleep(0.25)
                i2, t2 = read_cpu()
            else:
                i1, t1 = _cpu_times_last
                i2, t2 = read_cpu()
            _cpu_times_last = (i2, t2)
            d_idle  = i2 - i1
            d_total = t2 - t1
            cpu = round((1 - d_idle / max(d_total, 1)) * 100, 1)
        except Exception:
            pass
        return cpu, ram

    # ── macOS ─────────────────────────────────────────────────────
    else:
        # User requested to skip macOS usage to avoid any potential permission/execution issues
        cpu = 0.0
        ram = 0.0
        return cpu, ram

def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _normalize_log_mode(value):
    if value == LOG_MODE_ALL:
        return LOG_MODE_ALL
    return LOG_MODE_ERRORS_ONLY

def _is_log_enabled(level):
    with LOG_MODE_LOCK:
        mode = ACTIVE_LOG_MODE
    if mode == LOG_MODE_ALL:
        return True
    return level >= logging.ERROR

def _load_settings_file():
    default_settings = {
        "globalSystemPrompt": "",
        "temperature": 0.7,
        "logMode": DEFAULT_LOG_MODE,
        "chatEngine": "ollama",
        "llamaModel": "",
    }
    try:
        with DATA_FILE_LOCK:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        if isinstance(loaded, dict):
            merged = dict(default_settings)
            merged.update(loaded)
            merged["logMode"] = _normalize_log_mode(merged.get("logMode"))
            return merged
    except Exception:
        pass
    return default_settings

def _persist_settings_file(settings):
    with DATA_FILE_LOCK:
        temp_file = SETTINGS_FILE + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
            f.flush()
        os.replace(temp_file, SETTINGS_FILE)

def _set_active_log_mode(mode):
    global ACTIVE_LOG_MODE
    with LOG_MODE_LOCK:
        ACTIVE_LOG_MODE = _normalize_log_mode(mode)

def _get_hardware_specs():
    """Collect a stable host hardware snapshot for log enrichment."""
    plat = platform.system()
    specs = {
        "platform": plat,
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count() or 0,
        "has_psutil": HAS_PSUTIL,
        "ram_total_gb": None,
    }
    if HAS_PSUTIL:
        try:
            total_ram = psutil.virtual_memory().total
            specs["ram_total_gb"] = round(total_ram / (1024 ** 3), 2)
        except Exception:
            pass
    elif plat == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            msx = MEMORYSTATUSEX()
            msx.dwLength = ctypes.sizeof(msx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(msx))
            specs["ram_total_gb"] = round(msx.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            pass
    elif plat == "Linux":
        try:
            mem = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        mem[parts[0].rstrip(":")] = int(parts[1])
            total_kb = mem.get("MemTotal")
            if total_kb:
                specs["ram_total_gb"] = round((total_kb * 1024) / (1024 ** 3), 2)
        except Exception:
            pass
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        _, value = line.split(":", 1)
                        model = value.strip()
                        if model:
                            specs["processor"] = model
                            break
        except Exception:
            pass
    else:
        # Keep parity with _get_hw_stats fallback behavior:
        # skip macOS/native probing in the non-psutil path.
        pass
    return specs

# ── Stable Diffusion Helpers ───────────────────────────────────
def _is_ollama_running():
    """Check if Ollama responds on its default port."""
    try:
        req = urllib.request.Request(OLLAMA_HOST + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def _is_ollama_model_loaded():
    """Check if any model is currently loaded in Ollama RAM/VRAM memory."""
    try:
        req = urllib.request.Request(OLLAMA_HOST + "/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                return len(models) > 0
    except Exception:
        pass
    return False

def _unload_ollama_models():
    """Unload all active models from Ollama memory via keep_alive: 0."""
    try:
        req = urllib.request.Request(OLLAMA_HOST + "/api/ps", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                for m in models:
                    model_name = m.get("model") or m.get("name")
                    if model_name:
                        unload_payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
                        unload_req = urllib.request.Request(
                            OLLAMA_HOST + "/api/generate",
                            data=unload_payload,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        try:
                            with urllib.request.urlopen(unload_req, timeout=5) as u_resp:
                                u_resp.read()
                        except Exception:
                            pass
    except Exception:
        pass

def _is_llama_running():
    """Check if llama-server responds on its default port."""
    try:
        req = urllib.request.Request(LLAMA_HOST + "/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def _kill_llama():
    """Stop any running llama-server process."""
    global _LLAMA_PROC
    plat = platform.system()
    with _LLAMA_PROC_LOCK:
        if _LLAMA_PROC is not None:
            try:
                _LLAMA_PROC.terminate()
                _LLAMA_PROC.wait(timeout=5)
            except Exception:
                try:
                    _LLAMA_PROC.kill()
                except Exception:
                    pass
            _LLAMA_PROC = None
    # Also kill any stray processes
    try:
        if plat == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        else:
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    except Exception:
        pass
    for _ in range(10):
        if not _is_llama_running():
            break
        time.sleep(0.5)

def _detect_chat_template(model_path):
    """
    Infer the correct llama-server --chat-template value from the model filename.
    Returns None to let llama-server use the template embedded in the GGUF (preferred),
    or a string override for known models whose GGUFs often lack an embedded template.
    """
    name = os.path.basename(model_path).lower()
    # Llama 2 — uses [INST]/[/INST] with <<SYS>> blocks
    if "llama-2" in name or "llama2" in name:
        return "llama2"
    # Llama 3 — uses <|start_header_id|> format; usually embedded but be safe
    if "llama-3" in name or "llama3" in name:
        return "llama3"
    # Mistral v0.1/v0.2 — [INST] without system slot
    if "mistral" in name and ("instruct" in name or "chat" in name):
        return "mistral"
    # ChatML family (Phi-3, older Qwen, etc.) — usually embedded, skip
    return None

# ── Tool calling ───────────────────────────────────────────────
# Deliberately a fixed allowlist of typed operations rather than anything
# resembling a generic "run this command": the catalog ships abliterated
# models with refusal training removed, and dictated input is frequently
# misheard, so the worst outcome of a bad call must stay harmless.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an image from a text description. Use this whenever the user "
                "asks for a picture, image, drawing, painting, photo or render of something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed visual description of the image to create.",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]

TOOL_CHAT_TEMPLATE = os.path.join(SCRIPT_DIR, "config", "chat-templates", "qwen3-tools.jinja")

def _tools_enabled():
    """Whether tool calling is switched on in settings (off by default)."""
    try:
        return bool(_load_settings_file().get("toolsEnabled", False))
    except Exception:
        return False

def _estimate_tokens(text):
    """Rough token count estimate: ~4 chars per token for English text."""
    if not text:
        return 0
    return max(1, len(text) // 4)

def _b64_image_mime(b64):
    """Infer an image mime type from the leading bytes of its base64 text.

    The UI strips the data: prefix and sends bare base64, so the original mime
    is lost. Decoders sniff the real format from magic bytes anyway, but the
    data URL still needs a plausible type.
    """
    prefix = str(b64 or "")[:16]
    if prefix.startswith("iVBORw0KGgo"):
        return "image/png"
    if prefix.startswith("R0lGOD"):
        return "image/gif"
    if prefix.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"

def _ollama_msg_to_openai(msg):
    """Convert one Ollama-format message to OpenAI chat format.

    Ollama attaches images as a list of bare base64 strings under "images";
    OpenAI expects them inline in a content array as data URLs. Without this
    translation the images key is simply unknown to llama-server and the
    attached picture is silently ignored.

    Text-only messages keep plain string content, which is what the rest of
    the pipeline (truncation, merging) expects.
    """
    role = msg.get("role", "user")
    text = msg.get("content", "") or ""
    images = msg.get("images") or []
    if not images:
        return {"role": role, "content": text}
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    for b64 in images:
        if not b64:
            continue
        b64 = str(b64)
        url = b64 if b64.startswith("data:") else f"data:{_b64_image_mime(b64)};base64,{b64}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": role, "content": parts}

# Markers that identify a multimodal (vision) chat model. Kept in sync with
# VISION_MODELS in FastChatUI.html, including the separator normalization in
# _looks_like_vision_model() / isVision().
VISION_MODEL_MARKERS = (
    "-vl-", "vision", "llava", "bakllava", "moondream",
    "smolvlm", "minicpm-v", "cogvlm", "pixtral", "internvl", "gemma-3",
)

def _looks_like_vision_model(name):
    """Heuristic: does this model name/filename denote a multimodal model?

    Normalizes spaces, underscores and dots to hyphens so the same markers
    match both a raw filename ("Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf") and the
    display name the UI sees ("Qwen2.5 VL 3B Instruct Q4 K M").
    """
    base = os.path.basename(str(name or "")).lower()
    normalized = "-" + re.sub(r"[\s_.]+", "-", base) + "-"
    return any(marker in normalized for marker in VISION_MODEL_MARKERS)

def _find_mmproj_for_model(model_path):
    """Locate the multimodal projector (mmproj) GGUF paired with a model.

    Vision models ship as two files — the language model and a separate mmproj
    projector — and llama-server needs --mmproj to accept images at all. There
    is no metadata link between the pair, so match on filename.

    Returns None for text-only models even when a projector is present in the
    folder: attaching a mismatched projector makes llama-server fail outright
    with "mismatch between text model (n_embd = ...) and mmproj (n_embd = ...)",
    so a wrong guess is far worse than no guess.
    """
    if not model_path:
        return None
    models_dir = os.path.dirname(os.path.abspath(model_path))
    base = os.path.basename(model_path)
    if "mmproj" in base.lower():
        return None  # the projector itself is not a chat model
    # "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" -> "qwen2.5-vl-3b-instruct"
    stem = os.path.splitext(base)[0].lower()
    stem = re.sub(r"[-_.](iq|q)\d[^-_.]*([-_.][km][a-z]?)?$", "", stem)
    stem = re.sub(r"[-_.]f(16|32)$", "", stem)
    try:
        candidates = [f for f in sorted(os.listdir(models_dir))
                      if f.lower().endswith(".gguf") and "mmproj" in f.lower()]
    except Exception:
        return None
    if not candidates:
        return None
    # Prefer a projector whose filename references this specific model
    for fname in candidates:
        if stem and stem in fname.lower():
            return os.path.join(models_dir, fname)
    # Fall back to a lone projector only when this model actually looks
    # multimodal — some publishers name the projector generically
    # (e.g. gemma-3 ships "mmproj-model-f16.gguf" with no model reference).
    if len(candidates) == 1 and _looks_like_vision_model(model_path):
        return os.path.join(models_dir, candidates[0])
    return None

def _truncate_text_middle(text, max_chars):
    """Shorten text to max_chars while keeping both the head and the tail.

    Attachment-bearing messages are built by the UI as
    "Attached document context: <document> --- User: <question>", so the user's
    actual question sits at the very END of the string. Cutting from the tail
    would keep the document and silently discard the question, so cut from the
    middle instead and keep both ends.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n\n[... middle of content truncated to fit context window ...]\n\n"
    if max_chars <= len(marker):
        # No room for both ends plus a marker — the trailing question matters
        # more than the head of the document, so keep the tail.
        return text[-max_chars:]
    budget = max_chars - len(marker)
    head = (budget * 2) // 3
    tail = budget - head
    return text[:head] + marker + (text[-tail:] if tail > 0 else "")

def _truncate_messages_to_context(messages, max_tokens=3500, reserve_for_response=500):
    """
    Truncate a list of chat messages to fit within context window.

    Keeps the most recent messages, dropping older ones from the middle.
    Always preserves the first message (often system/context) AND the newest
    message, which is the user's current turn.

    The newest message is never dropped: if it alone exceeds the budget (the
    usual cause is an attached document), its content is truncated middle-out
    via _truncate_text_middle() instead. Dropping it entirely would send the
    model a conversation whose latest turn is a stale earlier question, which
    it would then answer as if it were current.

    Args:
        messages: List of {"role": str, "content": str} dicts
        max_tokens: Maximum context size (default 3500, leaving room for model overhead)
        reserve_for_response: Tokens to reserve for model response

    Returns:
        Truncated message list, truncation_notice (str or None)
    """
    if not messages:
        return messages, None

    available = max_tokens - reserve_for_response

    # Calculate current token usage
    def msg_tokens(msg):
        return _estimate_tokens(msg.get("content", "")) + 4  # +4 for role/formatting overhead

    if sum(msg_tokens(m) for m in messages) <= available:
        return messages, None  # No truncation needed

    # Budget for the truncation notice we are about to insert, so adding it
    # doesn't push us back over the limit.
    NOTICE_TOKENS = 32
    # Never emit an empty newest message, even if the first message is so large
    # that it consumes the whole budget on its own.
    MIN_NEWEST_CHARS = 400

    # Copy the messages we mutate so we never modify the caller's dicts.
    newest = dict(messages[-1])
    preceding = list(messages[:-1])

    # Always keep the first message too (usually the system prompt) — unless
    # the newest message IS the first message.
    first_msg = dict(preceding[0]) if preceding else None
    first_tokens = msg_tokens(first_msg) if first_msg else 0

    # Give the newest message whatever the first message and notice don't need.
    newest_budget = available - first_tokens - NOTICE_TOKENS
    content_truncated = False
    if msg_tokens(newest) > newest_budget:
        # msg_tokens() adds 4 for overhead; _estimate_tokens() is ~4 chars/token.
        max_chars = max(MIN_NEWEST_CHARS, (newest_budget - 4) * 4)
        original_content = newest.get("content", "") or ""
        shortened = _truncate_text_middle(original_content, max_chars)
        # Only claim a truncation if we actually removed something — the
        # MIN_NEWEST_CHARS floor can leave a short message fully intact even
        # when the budget was already blown by an oversized first message.
        if len(shortened) < len(original_content):
            newest["content"] = shortened
            content_truncated = True

    # Fill the remaining budget with the most recent preceding messages.
    used = first_tokens + msg_tokens(newest) + NOTICE_TOKENS
    kept = []
    for msg in reversed(preceding[1:]):
        tokens = msg_tokens(msg)
        if used + tokens <= available:
            kept.insert(0, msg)
            used += tokens

    dropped_count = (len(preceding) - 1 - len(kept)) if preceding else 0

    result = ([first_msg] if first_msg else []) + kept + [newest]

    # Every notice variant must contain the literal word "truncated" — the
    # Gemma system-folding path in _proxy_ollama() matches on it.
    if dropped_count > 0 and content_truncated:
        truncation_notice = f"[{dropped_count} earlier message(s) and latest message truncated to fit context window]"
    elif dropped_count > 0:
        truncation_notice = f"[{dropped_count} earlier message(s) truncated to fit context window]"
    elif content_truncated:
        truncation_notice = "[latest message truncated to fit context window]"
    else:
        truncation_notice = None

    if truncation_notice:
        # Insert the notice right after the first message (or at the front when
        # there is no preceding message to anchor to).
        result.insert(1 if first_msg else 0, {"role": "system", "content": truncation_notice})

    return result, truncation_notice

def _start_llama(model_path, job_id=None):
    """Start llama-server with the given GGUF model path.
    
    If job_id is provided, updates ENGINE_STARTUP_JOBS with progress.
    """
    global _LLAMA_PROC
    if not LLAMA_BIN or not os.path.isfile(LLAMA_BIN):
        if job_id:
            _update_engine_startup_job(job_id, status="failed", error="llama-server binary not found")
        return False
    if not model_path or not os.path.isfile(model_path):
        if job_id:
            _update_engine_startup_job(job_id, status="failed", error=f"Model file not found: {model_path}")
        return False
    try:
        if job_id:
            _update_engine_startup_job(job_id, phase="loading_model", message="Loading model...", progress_pct=5)
        
        bin_dir = os.path.dirname(os.path.abspath(LLAMA_BIN))
        env = os.environ.copy()
        # Add binary dir to library path for bundled .so/.dll files
        if platform.system() == "Linux":
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = bin_dir + (":" + existing if existing else "")
        cmd = [
            LLAMA_BIN,
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(LLAMA_PORT),
            "--ctx-size", "4096",
            "--n-predict", "-1",
        ]
        # Tool calling needs --jinja plus a template that actually handles
        # tools. Abliterated finetunes routinely ship a stripped ChatML
        # template with no <tools> block, in which case llama-server accepts
        # the tools parameter and silently never emits a tool_call — so the
        # bundled template is passed explicitly. Only applied when tool calling
        # is switched on, since overriding a model's template otherwise risks
        # regressing ordinary chat formatting.
        use_tools = _tools_enabled() and os.path.isfile(TOOL_CHAT_TEMPLATE)
        if use_tools:
            cmd += ["--jinja", "--chat-template-file", TOOL_CHAT_TEMPLATE]
        else:
            chat_template = _detect_chat_template(model_path)
            if chat_template:
                cmd += ["--chat-template", chat_template]

        # Vision models need their paired projector loaded explicitly, otherwise
        # llama-server starts fine but rejects every image with
        # "image input is not supported".
        mmproj_path = _find_mmproj_for_model(model_path)
        if mmproj_path:
            cmd += ["--mmproj", mmproj_path]
        
        # Capture stderr for progress tracking; stdout to devnull
        with _LLAMA_PROC_LOCK:
            _LLAMA_PROC = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE if job_id else subprocess.DEVNULL
            )

        # Pump stderr into a queue from a background thread. A dedicated reader
        # thread (rather than select.select on the pipe) works on Windows too,
        # where select() only supports sockets, not pipe file descriptors.
        stderr_queue = queue.Queue()
        if job_id and _LLAMA_PROC.stderr:
            def _pump_stderr(pipe, q):
                try:
                    for raw_line in iter(pipe.readline, b""):
                        q.put(raw_line)
                except Exception:
                    pass
            threading.Thread(target=_pump_stderr, args=(_LLAMA_PROC.stderr, stderr_queue), daemon=True).start()

        # Parse stderr for progress while waiting for server to be ready
        stderr_lines = []
        start_time = time.time()
        max_wait = 120  # 2 minutes max

        while time.time() - start_time < max_wait:
            if _is_llama_running():
                if job_id:
                    _update_engine_startup_job(job_id, status="complete", phase="ready",
                                                message="Engine ready", progress_pct=100)
                return True

            # Drain any stderr lines the reader thread has picked up
            if job_id:
                while True:
                    try:
                        line = stderr_queue.get_nowait()
                    except queue.Empty:
                        break
                    if line:
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        stderr_lines.append(line_str)
                        # Parse llama-server progress patterns
                        _parse_llama_startup_progress(job_id, line_str, start_time)
            
            # Update progress based on elapsed time if no specific progress
            if job_id:
                elapsed = time.time() - start_time
                # Estimate progress: most of the time is model loading
                # Assume ~60s typical load time, scale to 95% max before "ready"
                estimated_pct = min(95, int(5 + (elapsed / 60) * 85))
                _update_engine_startup_job(job_id, progress_pct=estimated_pct,
                                            message=f"Loading model... ({int(elapsed)}s)")
                time.sleep(0.1)
            else:
                time.sleep(1)
        
        # Timeout - collect any remaining stderr for error message
        error_msg = "llama-server failed to start (timeout)"
        if stderr_lines:
            # Get last few lines that might have error info
            recent = [l for l in stderr_lines[-10:] if l and 'error' in l.lower()]
            if recent:
                error_msg = f"llama-server error: {recent[-1]}"
        
        if job_id:
            _update_engine_startup_job(job_id, status="failed", error=error_msg)
        return False
    except Exception as e:
        if job_id:
            _update_engine_startup_job(job_id, status="failed", error=str(e))
        return False

def _parse_llama_startup_progress(job_id, line, start_time):
    """Parse llama-server stderr line and update progress."""
    line_lower = line.lower()
    
    # Model loading phases from llama.cpp logs
    if "loading model" in line_lower or "llama_model_load" in line_lower:
        _update_engine_startup_job(job_id, phase="loading_model", message="Loading model weights...", progress_pct=15)
    elif "model size" in line_lower or "model params" in line_lower:
        _update_engine_startup_job(job_id, phase="loading_model", message="Model loaded, initializing...", progress_pct=50)
    elif "kv cache" in line_lower or "kv_cache" in line_lower:
        _update_engine_startup_job(job_id, phase="init_context", message="Initializing KV cache...", progress_pct=70)
    elif "warming up" in line_lower or "warmup" in line_lower:
        _update_engine_startup_job(job_id, phase="warmup", message="Warming up model...", progress_pct=85)
    elif "listening" in line_lower or "server listening" in line_lower:
        _update_engine_startup_job(job_id, phase="ready", message="Server ready", progress_pct=95)
    elif "error" in line_lower:
        # Don't mark as failed yet, just update message
        _update_engine_startup_job(job_id, message=f"Warning: {line[:100]}")

def _kill_ollama():
    """Stop any running Ollama process or unload models. Platform-specific."""
    global _OLLAMA_PROC
    # First, attempt to unload models via HTTP API
    _unload_ollama_models()

    # If we started the process ourselves, terminate that handle directly
    with _OLLAMA_PROC_LOCK:
        if _OLLAMA_PROC is not None:
            try:
                _OLLAMA_PROC.terminate()
                _OLLAMA_PROC.wait(timeout=5)
            except Exception:
                try:
                    _OLLAMA_PROC.kill()
                except Exception:
                    pass
            _OLLAMA_PROC = None

    plat = platform.system()
    try:
        if plat == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "ollama-windows.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], capture_output=True)
            # Try to stop Windows service if present (requires admin, may fail silently)
            subprocess.run(["sc", "stop", "Ollama"], capture_output=True)
            # Aggressive fallback: kill any process with 'ollama' in the name
            try:
                result = subprocess.run(["wmic", "process", "where", "name like '%ollama%'", "delete"],
                                        capture_output=True, text=True, timeout=5)
            except Exception:
                pass
        elif plat == "Linux":
            subprocess.run(["pkill", "-9", "-f", "ollama-linux"], capture_output=True)
            subprocess.run(["pkill", "-9", "-f", "ollama serve"], capture_output=True)
            subprocess.run(["pkill", "-9", "-f", "ollama"], capture_output=True)
        else:  # macOS
            subprocess.run(["pkill", "-9", "-f", "ollama-darwin"], capture_output=True)
            subprocess.run(["pkill", "-9", "-f", "ollama serve"], capture_output=True)
            subprocess.run(["pkill", "-9", "-f", "ollama"], capture_output=True)
    except Exception:
        pass
    # Wait for both the port to be free and the model to be unloaded from memory
    for _ in range(10):
        if not _is_ollama_running() and not _is_ollama_model_loaded():
            break
        time.sleep(0.5)

def _start_ollama():
    """Start Ollama in the background if binary exists."""
    global _OLLAMA_PROC
    if not OLLAMA_BIN or not os.path.isfile(OLLAMA_BIN):
        return False
    try:
        env = os.environ.copy()
        env["OLLAMA_MODELS"] = os.path.join(SCRIPT_DIR, "models", "ollama_data")
        env["OLLAMA_ORIGINS"] = "*"
        env["OLLAMA_HOST"] = "127.0.0.1:11434"
        with _OLLAMA_PROC_LOCK:
            _OLLAMA_PROC = subprocess.Popen([OLLAMA_BIN, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Wait for it to be ready
        for _ in range(30):
            if _is_ollama_running():
                return True
            time.sleep(1)
        return False
    except Exception:
        return False

def _run_model_pull(job_id, model_name):
    """Run 'ollama pull' and track progress. Called in background thread."""
    if not OLLAMA_BIN or not os.path.isfile(OLLAMA_BIN):
        _update_model_pull_job(job_id, status="failed", error="Ollama binary not found")
        return
    
    # Ensure Ollama is running first
    if not _is_ollama_running():
        _update_model_pull_job(job_id, message="Starting Ollama...")
        if not _start_ollama():
            _update_model_pull_job(job_id, status="failed", error="Could not start Ollama")
            return
    
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = os.path.join(SCRIPT_DIR, "models", "ollama_data")
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    
    try:
        proc = subprocess.Popen(
            [OLLAMA_BIN, "pull", model_name],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        # Parse progress output: "pulling abc123... 45% ▕██████     ▏ 1.2 GB/2.5 GB"
        pct_pattern = re.compile(r'(\d+)%')
        size_pattern = re.compile(r'([\d.]+\s*[KMGT]?B)\s*/\s*([\d.]+\s*[KMGT]?B)', re.IGNORECASE)
        
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            
            # Check for completion
            if "success" in line.lower():
                _update_model_pull_job(job_id, status="complete", progress_pct=100, message="Download complete!")
                break
            
            # Check for errors
            if "error" in line.lower() or "failed" in line.lower():
                _update_model_pull_job(job_id, status="failed", error=line)
                break
            
            # Parse progress percentage
            pct_match = pct_pattern.search(line)
            size_match = size_pattern.search(line)
            
            update = {"message": line[:80]}
            if pct_match:
                update["progress_pct"] = int(pct_match.group(1))
            if size_match:
                update["downloaded"] = size_match.group(1)
                update["total"] = size_match.group(2)
            
            _update_model_pull_job(job_id, **update)
        
        proc.wait()
        
        # Check final status
        job = _get_model_pull_job(job_id)
        if job.get("status") not in ("complete", "failed"):
            if proc.returncode == 0:
                _update_model_pull_job(job_id, status="complete", progress_pct=100, message="Download complete!")
            else:
                _update_model_pull_job(job_id, status="failed", error=f"Pull failed with exit code {proc.returncode}")
                
    except Exception as e:
        _update_model_pull_job(job_id, status="failed", error=str(e))

def _parse_sd_output(pipe, job_id, total_steps):
    """Read sd stdout+stderr line-by-line and update job progress.
    stable-diffusion.cpp prints step info like 'step 1 sampling completed'.
    We merge stdout+stderr so we catch progress regardless of which stream
    the binary writes it to."""
    # Match: "step 1 sampling completed" or "step 1/20" or "step 1 of 20"
    step_pattern = re.compile(r"step\s+(\d+)", re.IGNORECASE)
    # Only match X/Y when it follows "sampling" to avoid model-loading bars
    sampling_pattern = re.compile(r"sampling.*?\b(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    # Only match % when it follows "sampling"
    pct_pattern = re.compile(r"sampling.*?(\d+)")

    last_step = 0
    step_start_time = None
    step_times = []

    try:
        for raw_line in iter(pipe.readline, b""):
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Try to extract step / total
            current_step = 0
            matched_total = total_steps

            m = step_pattern.search(line)
            if m:
                current_step = int(m.group(1))
            else:
                m = sampling_pattern.search(line)
                if m:
                    current_step = int(m.group(1))
                    matched_total = int(m.group(2))
                else:
                    m = pct_pattern.search(line)
                    if m and total_steps > 0:
                        pct = int(m.group(1))
                        current_step = int(pct / 100.0 * total_steps)

            if current_step > 0:
                now = time.time()
                if step_start_time is not None and current_step > last_step:
                    for _ in range(current_step - last_step):
                        step_times.append(now - step_start_time)
                        if len(step_times) > 10:
                            step_times.pop(0)
                step_start_time = now
                last_step = current_step

                job = _get_image_job(job_id)
                elapsed = now - job.get("started_at", now)
                eta_ms = None
                if current_step < matched_total and step_times:
                    avg_step = sum(step_times) / len(step_times)
                    eta_ms = int(avg_step * (matched_total - current_step) * 1000)

                _update_image_job(
                    job_id,
                    status="generating",
                    step=current_step,
                    total_steps=matched_total,
                    elapsed_ms=int(elapsed * 1000),
                    eta_ms=eta_ms,
                )
    except Exception:
        pass
    finally:
        pipe.close()

def _run_sd_generation(job_id, payload, output_path):
    """Run stable-diffusion.cpp CLI to generate an image asynchronously. Updates job dict."""
    if not SD_BINARY or not os.path.isfile(SD_BINARY):
        _update_image_job(job_id, status="error", error="Stable Diffusion engine not found. Please run the installer.")
        return

    # Resolve the model: use the file from the payload if provided, else fall back to first found.
    models_dir = os.path.join(SCRIPT_DIR, "models")
    requested_file = payload.get("model_file", "")
    if requested_file:
        # Sanitize: only allow a plain filename (no path components)
        requested_file = os.path.basename(requested_file)
        candidate = os.path.join(models_dir, requested_file)
        if os.path.isfile(candidate) and candidate.lower().endswith(".safetensors"):
            model_path = candidate
        else:
            _update_image_job(job_id, status="error", error=f"Image model '{requested_file}' not found in models folder.")
            return
    else:
        available = _find_sd_models()
        if not available:
            _update_image_job(job_id, status="error", error="No .safetensors image model found. Place one in the models folder.")
            return
        model_path = available[0]["path"]

    prompt = payload.get("prompt", "").strip()
    if not prompt:
        _update_image_job(job_id, status="error", error="Prompt is required.")
        return

    # Clamp parameters to safe ranges
    steps = max(1, min(50, _safe_int(payload.get("steps"), 20)))
    cfg = max(1.0, min(15.0, float(payload.get("cfg_scale", 7.0))))
    width = max(256, min(768, _safe_int(payload.get("width"), 512)))
    height = max(256, min(768, _safe_int(payload.get("height"), 512)))
    seed = _safe_int(payload.get("seed"), -1)
    negative = payload.get("negative_prompt", "").strip()
    sampling = payload.get("sampling_method", "euler_a")
    if sampling not in ("euler", "euler_a", "heun", "dpm2", "dpm++2m", "dpm++2mv2"):
        sampling = "euler_a"

    cmd = [
        SD_BINARY,
        "-m", model_path,
        "-p", prompt,
        "-o", output_path,
        "--steps", str(steps),
        "--cfg-scale", str(cfg),
        "--width", str(width),
        "--height", str(height),
        "--seed", str(seed),
        "--sampling-method", sampling,
        "--rng", SD_RNG,
        "-v",
    ]
    if negative:
        cmd += ["-n", negative]
    if SD_BACKEND:
        cmd += ["--backend", SD_BACKEND]

    _update_image_job(
        job_id,
        status="starting",
        step=0,
        total_steps=steps,
        params={
            "prompt": prompt,
            "negative_prompt": negative,
            "steps": steps,
            "cfg_scale": cfg,
            "width": width,
            "height": height,
            "seed": seed,
            "sampling_method": sampling,
        },
    )

    proc = None
    try:
        # Set LD_LIBRARY_PATH so the dynamic linker can find libstable-diffusion.so
        # which lives alongside the sd-cli binary.
        sd_dir = os.path.dirname(os.path.abspath(SD_BINARY))
        env = os.environ.copy()
        existing_ldpath = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = sd_dir + (":" + existing_ldpath if existing_ldpath else "")

        # Merge stdout+stderr so we catch progress regardless of which stream sd uses
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        # Start a thread to parse combined output
        output_thread = threading.Thread(
            target=_parse_sd_output,
            args=(proc.stdout, job_id, steps),
            daemon=True,
        )
        output_thread.start()

        proc.wait()
        output_thread.join()

        if proc.returncode != 0:
            _update_image_job(job_id, status="error", error="Stable Diffusion engine exited with an error. Check server logs.")
            return

        if not os.path.isfile(output_path):
            _update_image_job(job_id, status="error", error="Image generation failed: output file not created.")
            return

        # Read and encode the PNG
        with open(output_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Images placed into the chat transcript are kept on disk and referenced
        # by URL. Embedding the base64 in chats.json instead would add ~1.5 MB
        # per image to a file that is rewritten on every debounced save.
        image_url = None
        if payload.get("persist"):
            image_url = "/chat_data/generated_images/" + os.path.basename(output_path)
        else:
            try:
                os.remove(output_path)
            except Exception:
                pass

        job = _get_image_job(job_id)
        started_at = job.get("started_at", time.time())
        _update_image_job(
            job_id,
            status="done",
            image_b64=img_b64,
            image_url=image_url,
            elapsed_ms=int((time.time() - started_at) * 1000),
        )
    except Exception as e:
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        _update_image_job(job_id, status="error", error=str(e))

class LogFormatter(logging.Formatter):
    """Readable, multi-line formatter with strict spacing and rich context."""

    def format(self, record):
        local_now = datetime.now().astimezone()
        timestamp = local_now.isoformat()
        tz_label = local_now.tzname() or "local"
        exception_text = ""
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
        hardware = getattr(record, "hardware_specs", None) or {}
        request_headers = getattr(record, "request_headers", None) or {}
        lines = [
            "=" * 96,
            f"timestamp     : {timestamp}",
            f"timezone      : {tz_label}",
            f"level         : {record.levelname}",
            f"event         : {record.getMessage()}",
            "",
            "request_context",
            f"  request_id   : {getattr(record, 'request_id', '-')}",
            f"  method       : {getattr(record, 'http_method', '-')}",
            f"  path         : {getattr(record, 'path', '-')}",
            f"  api_source   : {getattr(record, 'api_source', '-')}",
            f"  client_ip    : {getattr(record, 'client_ip', '-')}",
            f"  user_agent   : {request_headers.get('User-Agent', '-')}",
            f"  model_name   : {getattr(record, 'model_name', '-')}",
            f"  model_temp   : {getattr(record, 'model_temperature', '-')}",
            f"  model_stream : {getattr(record, 'model_stream', '-')}",
            "",
            "code_context",
            f"  logger       : {record.name}",
            f"  module       : {record.module}",
            f"  function     : {record.funcName}",
            f"  line         : {record.lineno}",
            f"  thread       : {record.threadName}",
            "",
            "hardware_specs",
            f"  platform     : {hardware.get('platform', '-')}",
            f"  release      : {hardware.get('platform_release', '-')}",
            f"  machine      : {hardware.get('machine', '-')}",
            f"  processor    : {hardware.get('processor', '-')}",
            f"  cpu_logical  : {hardware.get('cpu_count_logical', '-')}",
            f"  ram_total_gb : {hardware.get('ram_total_gb', '-')}",
            f"  python       : {hardware.get('python_version', '-')}",
        ]
        if exception_text:
            lines.extend(["", "exception", exception_text])
        lines.append("=" * 96)
        return "\n".join(lines)

class ImmediateFlushRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Flush each record immediately so logs are visible without waiting."""

    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass

def configure_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("chat_server")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    log_queue = queue.Queue(maxsize=10000)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    file_handler = ImmediateFlushRotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=1, encoding="utf-8"
    )
    file_handler.setFormatter(LogFormatter())

    listener = logging.handlers.QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()
    return logger, listener

def _log_event(level, message, request_context=None, exc_info=False):
    if not _is_log_enabled(level):
        return
    request_context = request_context or {}
    LOGGER.log(
        level,
        message,
        extra={
            "request_id": request_context.get("request_id", "-"),
            "http_method": request_context.get("method", "-"),
            "path": request_context.get("path", "-"),
            "api_source": request_context.get("api_source", "-"),
            "client_ip": request_context.get("client_ip", "-"),
            "request_headers": request_context.get("request_headers", {}),
            "model_name": request_context.get("model_name", "-"),
            "model_temperature": request_context.get("model_temperature", "-"),
            "model_stream": request_context.get("model_stream", "-"),
            "hardware_specs": HOST_HARDWARE_SPECS,
        },
        exc_info=exc_info,
    )

def ensure_data_dir():
    """Create required data folders if they don't exist."""
    os.makedirs(CHATS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    # Hold the shared data-file lock across the check-then-create so a second
    # server process pointed at the same data dir can't race this one.
    with DATA_FILE_LOCK:
        if not os.path.exists(CHATS_FILE):
            with open(CHATS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "globalSystemPrompt": "",
                        "temperature": 0.7,
                        "logMode": DEFAULT_LOG_MODE,
                    },
                    f,
                )

class ChatHandler(http.server.BaseHTTPRequestHandler):
    """Handles all HTTP requests for the Portable AI Chat."""

    def _build_request_context(self, api_source):
        headers = {}
        for key in ("User-Agent", "Authorization"):
            value = self.headers.get(key)
            if value:
                headers[key] = value
        return {
            "request_id": str(uuid.uuid4()),
            "method": getattr(self, "command", "-"),
            "path": getattr(self, "path", "-"),
            "api_source": api_source,
            "client_ip": self.client_address[0] if self.client_address else "-",
            "request_headers": headers,
        }

    def _read_body(self):
        length = _safe_int(self.headers.get("Content-Length"), 0)
        return self.rfile.read(length) if length > 0 else b""

    def log_message(self, format, *args):
        """Print all requests for easy debugging."""
        msg = format % args
        ts = time.strftime("%H:%M:%S")
        # Colour-code by status: errors red, warnings yellow, ok green
        if "404" in msg or "500" in msg or "502" in msg:
            prefix = "  \033[91m[ERR]\033[0m"
        elif "200" in msg or "204" in msg:
            prefix = "  \033[92m[ OK]\033[0m"
        else:
            prefix = "  \033[93m[---]\033[0m"
        print(f"{prefix} {ts}  {msg}")

    # ── CORS headers ───────────────────────────────────────────
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ── Routing ────────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path

        # Serve the main UI
        if path == "/" or path == "/index.html":
            self._serve_html()

        # Chat data API
        elif path == "/api/chats":
            self._get_chats()

        # Settings API
        elif path == "/api/settings":
            self._get_settings()

        # Hardware stats API
        elif path == "/api/stats":
            self._get_stats()

        # Engine status API
        elif path == "/api/engine-status":
            self._get_engine_status()

        # Chat models API (engine-aware)
        elif path == "/api/chat-models":
            self._get_chat_models()

        # Image model info API
        elif path == "/api/image-models":
            self._get_image_models()

        # Text-to-speech voice list
        elif path == "/api/tts-voices":
            self._get_tts_voices()

        elif path == "/api/tts-voice-download-status":
            self._get_voice_download_status()

        # Image generation progress API
        elif path == "/api/image-progress":
            self._get_image_progress()

        # Engine startup progress API
        elif path == "/api/engine-startup-status":
            self._get_engine_startup_status()

        # Model pull progress API
        elif path == "/api/pull-model-status":
            self._get_model_pull_status()

        # HuggingFace download progress API
        elif path == "/api/hf-download-status":
            self._get_hf_download_status()

        # Proxy Ollama API
        elif path.startswith("/ollama/"):
            self._proxy_ollama("GET")

        else:
            # Try serving static files from SCRIPT_DIR
            self._serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/chats":
            self._save_chats()

        elif path == "/api/settings":
            self._save_settings()

        # Image generation API
        elif path == "/api/generate-image":
            self._generate_image()

        # Text-to-speech synthesis
        elif path == "/api/tts":
            self._tts_endpoint()

        # Speech-to-text transcription
        elif path == "/api/stt":
            self._stt_endpoint()

        # Voice library management
        elif path == "/api/tts-voice-download":
            self._voice_download_endpoint()

        elif path == "/api/tts-voice-delete":
            self._delete_voice_endpoint()

        # Engine control APIs
        elif path == "/api/stop-ollama":
            self._stop_ollama_endpoint()

        elif path == "/api/start-ollama":
            self._start_ollama_endpoint()

        elif path == "/api/switch-engine":
            self._switch_engine_endpoint()

        # Model pull API
        elif path == "/api/pull-model":
            self._pull_model_endpoint()

        # HuggingFace download API
        elif path == "/api/hf-download":
            self._hf_download_endpoint()

        # Proxy Ollama API
        elif path.startswith("/ollama/"):
            self._proxy_ollama("POST")

        else:
            self.send_response(404)
            self._cors_headers()
            self.end_headers()

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/ollama/"):
            self._proxy_ollama("DELETE")
        else:
            self.send_response(404)
            self._cors_headers()
            self.end_headers()

    # ── Serve HTML ─────────────────────────────────────────────
    def _serve_html(self):
        try:
            with open(HTML_FILE, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"FastChatUI.html not found.")

    def _serve_static(self, path):
        """Serve static files (CSS, JS, images) from SCRIPT_DIR."""
        safe_path = os.path.normpath(path.lstrip("/"))
        full_path = os.path.join(SCRIPT_DIR, safe_path)

        # Security: don't allow path traversal. Require the trailing separator
        # so a sibling directory sharing SCRIPT_DIR's prefix (e.g. SCRIPT_DIR
        # + "_secret") doesn't pass the check.
        if not (full_path == SCRIPT_DIR or full_path.startswith(SCRIPT_DIR + os.sep)):
            self.send_response(403)
            self.end_headers()
            return

        if os.path.isfile(full_path):
            ext = os.path.splitext(full_path)[1].lower()
            mime_types = {
                ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
                ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
                ".svg": "image/svg+xml", ".ico": "image/x-icon"
            }
            content_type = mime_types.get(ext, "application/octet-stream")
            with open(full_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers()

    # ── Chat Persistence ───────────────────────────────────────
    def _get_chats(self):
        request_context = self._build_request_context("/api/chats")
        try:
            with DATA_FILE_LOCK:
                with open(CHATS_FILE, "r", encoding="utf-8") as f:
                    data = f.read()
        except (FileNotFoundError, json.JSONDecodeError):
            data = "[]"
        except Exception:
            _log_event(logging.ERROR, "Failed to read chats file", request_context=request_context, exc_info=True)
            data = "[]"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data.encode("utf-8"))

    def _save_chats(self):
        request_context = self._build_request_context("/api/chats")
        body = self._read_body()
        try:
            chats = json.loads(body)
            with DATA_FILE_LOCK:
                temp_file = CHATS_FILE + ".tmp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(chats, f, ensure_ascii=False, indent=2)
                    f.flush()
                os.replace(temp_file, CHATS_FILE)
            _log_event(logging.INFO, "Chats saved successfully", request_context=request_context)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())
        except Exception as e:
            _log_event(logging.ERROR, "Failed to save chats", request_context=request_context, exc_info=True)
            self.send_response(500)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _get_settings(self):
        request_context = self._build_request_context("/api/settings")
        try:
            settings = _load_settings_file()
            data = json.dumps(settings)
        except Exception:
            _log_event(logging.ERROR, "Failed to read settings file", request_context=request_context, exc_info=True)
            data = "{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data.encode("utf-8"))

    def _save_settings(self):
        request_context = self._build_request_context("/api/settings")
        body = self._read_body()
        try:
            incoming = json.loads(body)
            if not isinstance(incoming, dict):
                raise ValueError("Settings payload must be a JSON object")

            # Hold the lock across the whole load-update-persist cycle so two
            # concurrent /api/settings requests can't interleave and silently
            # drop one another's changes (lost update).
            with DATA_FILE_LOCK:
                settings = _load_settings_file()
                settings.update(incoming)
                settings["logMode"] = _normalize_log_mode(settings.get("logMode"))
                _persist_settings_file(settings)
            _set_active_log_mode(settings["logMode"])
            _log_event(logging.INFO, "Settings saved successfully", request_context=request_context)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "logMode": settings["logMode"]}).encode())
        except Exception as e:
            _log_event(logging.ERROR, "Failed to save settings", request_context=request_context, exc_info=True)
            self.send_response(500)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    # ── Hardware Stats ─────────────────────────────────────────
    def _get_stats(self):
        """Return CPU % and RAM % as JSON. Works with no external packages."""
        try:
            cpu, ram = _get_hw_stats()
            data = json.dumps({"cpu_percent": cpu, "ram_percent": ram, "has_psutil": HAS_PSUTIL})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(data.encode())
        except Exception as e:
            _log_event(
                logging.ERROR,
                "Failed to collect hardware stats",
                request_context=self._build_request_context("/api/stats"),
                exc_info=True,
            )
            self.send_response(500)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    # ── Image Generation API ───────────────────────────────────
    def _get_image_models(self):
        """Return all .safetensors models found in the models folder."""
        current_models = _find_sd_models()
        models = []
        for m in current_models:
            size_str = ""
            try:
                size_bytes = os.path.getsize(m["path"])
                size_str = f"{size_bytes / 1073741824:.2f} GB"
            except Exception:
                pass
            models.append({
                "name": m["name"],
                "file": m["file"],
                "size": size_str,
            })
        sd_available = SD_BINARY is not None and len(models) > 0
        data = json.dumps({"models": models, "sd_enabled": sd_available})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data.encode())

    # ── Text-to-Speech API ─────────────────────────────────────
    def _get_tts_voices(self):
        """Return installed piper voices, plus the catalog of downloadable ones."""
        voices = _find_piper_voices()
        installed_ids = {v["id"] for v in voices}
        catalog = [
            {
                "id": c["id"], "label": c["label"], "description": c["description"],
                "quality": c["quality"], "size_mb": c["size_mb"],
                "installed": c["id"] in installed_ids,
            }
            for c in _load_voice_catalog()
        ]
        data = json.dumps({
            "tts_enabled": bool(PIPER_BINARY) and len(voices) > 0,
            "has_binary": bool(PIPER_BINARY),
            "voices": [{"id": v["id"], "name": v["name"], "file": v["file"]} for v in voices],
            "catalog": catalog,
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data.encode())

    def _tts_endpoint(self):
        """Synthesize speech from text and return WAV audio.

        Synchronous on purpose: piper runs at roughly 10x realtime, so even a
        long reply is a couple of seconds and the background-job pattern used
        for image generation would only add latency and polling.
        """
        request_context = self._build_request_context("/api/tts")
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        def fail(code, message):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": message}).encode())

        raw_text = payload.get("text", "")
        speech_text = _text_for_speech(raw_text)
        if not speech_text:
            # Nothing speakable (e.g. the reply was only a code block).
            fail(400, "Nothing to speak.")
            return
        if len(speech_text) > TTS_MAX_CHARS:
            speech_text = speech_text[:TTS_MAX_CHARS]

        voices = _find_piper_voices()
        if not voices:
            fail(503, "No voice installed. Please run the installer to add one.")
            return

        # Pick the requested voice, else fall back to the first installed one.
        requested = os.path.basename(str(payload.get("voice", "") or ""))
        voice = next((v for v in voices if v["id"] == requested or v["file"] == requested), voices[0])

        try:
            rate = float(payload.get("length_scale", 1.0))
        except (TypeError, ValueError):
            rate = 1.0
        rate = max(0.5, min(2.0, rate))  # clamp to a sane speaking speed

        try:
            wav = _synthesize_speech(speech_text, voice["path"], length_scale=rate)
        except Exception as e:
            _log_event(logging.ERROR, "Speech synthesis failed",
                       request_context=request_context, exc_info=True)
            fail(500, str(e))
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header("Cache-Control", "no-store")
        self._cors_headers()
        self.end_headers()
        try:
            self.wfile.write(wav)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client navigated away mid-playback

    def _voice_download_endpoint(self):
        """Start downloading a catalog voice in the background."""
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        def fail(code, message):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": message}).encode())

        voice_id = str(payload.get("voice", "") or "")
        # Only ids present in the bundled catalog are accepted, so the id can
        # never steer the download URL or the destination path.
        entry = next((c for c in _load_voice_catalog() if c["id"] == voice_id), None)
        if not entry:
            fail(400, "Unknown voice.")
            return
        if os.path.isfile(os.path.join(PIPER_VOICES_DIR, entry["id"] + ".onnx")):
            fail(409, "That voice is already installed.")
            return

        job_id = str(uuid.uuid4())
        _register_voice_job(job_id, entry["id"])
        threading.Thread(target=_run_voice_download, args=(job_id, entry), daemon=True).start()

        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "job_id": job_id, "voice": entry["id"]}).encode())

    def _get_voice_download_status(self):
        query = parse_qs(urlparse(self.path).query)
        job_id = query.get("job_id", [None])[0]
        job = _get_voice_job(job_id) if job_id else {}
        if not job:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Job not found"}).encode())
            return
        resp = {
            "status": job.get("status"), "voice": job.get("voice"),
            "progress_pct": job.get("progress_pct", 0),
            "message": job.get("message", ""),
        }
        if job.get("status") == "failed":
            resp["error"] = job.get("error", "Unknown error")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def _delete_voice_endpoint(self):
        """Remove an installed voice. Voices are ~63 MB each on a portable drive."""
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        def respond(code, obj):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(obj).encode())

        # basename() plus a match against installed voices keeps this from
        # deleting anything outside the voices folder.
        voice_id = os.path.basename(str(payload.get("voice", "") or ""))
        installed = _find_piper_voices()
        match = next((v for v in installed if v["id"] == voice_id), None)
        if not match:
            respond(404, {"error": "Voice not installed."})
            return
        if len(installed) <= 1:
            respond(409, {"error": "That's the only voice installed — read-aloud would stop working."})
            return
        try:
            os.remove(match["path"])
            cfg = match["path"] + ".json"
            if os.path.isfile(cfg):
                os.remove(cfg)
        except Exception as e:
            respond(500, {"error": str(e)})
            return
        respond(200, {"ok": True, "voice": voice_id})

    # ── Speech-to-Text API ─────────────────────────────────────
    def _stt_endpoint(self):
        """Transcribe uploaded WAV audio and return the recognized text.

        The body is raw audio/wav rather than a multipart form: the stdlib has
        no maintained multipart parser (cgi was removed in 3.13) and a single
        binary body needs none.
        """
        request_context = self._build_request_context("/api/stt")

        def fail(code, message):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": message}).encode())

        length = _safe_int(self.headers.get("Content-Length"), 0)
        if length <= 0:
            fail(400, "No audio received.")
            return
        if length > STT_MAX_UPLOAD_BYTES:
            fail(413, "Recording is too long.")
            return

        audio = self.rfile.read(length)
        # whisper.cpp decodes WAV via miniaudio; browser-native formats such as
        # webm/opus are rejected outright, so catch that here with a clear
        # message instead of a confusing engine error.
        if not (audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"):
            fail(400, "Audio must be WAV format.")
            return

        models = _find_whisper_models()
        if not models:
            fail(503, "No speech recognition model installed. Please run the installer.")
            return

        try:
            text = _transcribe_audio(audio, models[0]["path"])
        except Exception as e:
            _log_event(logging.ERROR, "Transcription failed",
                       request_context=request_context, exc_info=True)
            fail(500, str(e))
            return

        _log_event(logging.INFO, "Audio transcribed", request_context=request_context)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"text": text}).encode())

    def _get_engine_status(self):
        """Return which engines are currently running."""
        ollama_up = _is_ollama_running()
        ollama_model_loaded = _is_ollama_model_loaded() if ollama_up else False
        llama_up = _is_llama_running()
        current_models = _find_sd_models()
        with ACTIVE_ENGINE_LOCK:
            active = ACTIVE_ENGINE
        data = json.dumps({
            "ollama": ollama_up,
            "ollama_model_loaded": ollama_model_loaded,
            "llama": llama_up,
            "active_engine": active,
            "llama_available": bool(LLAMA_BIN and os.path.isfile(LLAMA_BIN)),
            "llama_models": [m["file"] for m in _find_gguf_models()],
            "sd_enabled": SD_BINARY is not None and len(current_models) > 0,
            "sd_binary": bool(SD_BINARY),
            "sd_model": len(current_models) > 0,
            "sd_healthy": _test_sd_binary(),
            "sd_backend": SD_BACKEND or "cpu",
            "tts_enabled": bool(PIPER_BINARY) and len(_find_piper_voices()) > 0,
            "stt_enabled": bool(WHISPER_BINARY) and len(_find_whisper_models()) > 0,
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data.encode())

    def _stop_ollama_endpoint(self):
        request_context = self._build_request_context("/api/stop-ollama")
        try:
            _kill_ollama()
            _kill_llama()
            _log_event(logging.INFO, "Chat engine stopped by user", request_context=request_context)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "message": "Chat engine stopped."}).encode())
        except Exception as e:
            _log_event(logging.ERROR, "Failed to stop Ollama", request_context=request_context, exc_info=True)
            self.send_response(500)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _get_chat_models(self):
        """Return chat models for the active engine."""
        with ACTIVE_ENGINE_LOCK:
            active = ACTIVE_ENGINE
        if active == "llama":
            models = []
            for m in _find_gguf_models():
                size_str = ""
                try:
                    size_str = f"{os.path.getsize(m['path']) / 1073741824:.1f} GB"
                except Exception:
                    pass
                models.append({"name": m["name"], "file": m["file"], "size": size_str})
            data = json.dumps({"engine": "llama", "models": models})
        else:
            # Proxy Ollama tags and reformat
            try:
                req = urllib.request.Request(OLLAMA_HOST + "/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    raw = json.loads(resp.read())
                models = [{"name": m["name"], "file": "",
                           "size": f"{m.get('size', 0) / 1073741824:.1f} GB"}
                          for m in raw.get("models", [])]
            except Exception:
                models = []
            data = json.dumps({"engine": "ollama", "models": models})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data.encode())

    def _switch_engine_endpoint(self):
        """Kill the current chat engine and start the requested one.
        
        For llama.cpp: returns job_id immediately, starts async, poll /api/engine-startup-status
        For Ollama: synchronous (fast startup)
        """
        request_context = self._build_request_context("/api/switch-engine")
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        engine = payload.get("engine", "ollama")
        async_mode = payload.get("async", True)  # Default to async for llama.cpp
        
        if engine not in ("ollama", "llama"):
            self.send_response(400)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "engine must be 'ollama' or 'llama'"}).encode())
            return

        try:
            if engine == "llama":
                model_file = os.path.basename(payload.get("model", ""))
                if not model_file:
                    gguf = _find_gguf_models()
                    if not gguf:
                        raise RuntimeError("No .gguf model found in models folder.")
                    model_file = gguf[0]["file"]
                model_path = os.path.join(SCRIPT_DIR, "models", model_file)
                if not os.path.isfile(model_path) or not model_path.lower().endswith(".gguf"):
                    raise RuntimeError(f"Model '{model_file}' not found.")

                # Persist intent and set active engine BEFORE the slow startup
                # so a page reload during the 60-second wait sees the correct state.
                settings = _load_settings_file()
                settings["chatEngine"] = "llama"
                settings["llamaModel"] = model_file
                _persist_settings_file(settings)
                _set_active_engine("llama")

                # Stop Ollama first
                _kill_ollama()
                _kill_llama()
                
                if async_mode:
                    # Start async and return job_id immediately
                    job_id = str(uuid.uuid4())
                    _register_engine_startup_job(job_id, engine, model_file)
                    
                    def _async_start():
                        try:
                            _update_engine_startup_job(job_id, phase="killing_engines",
                                                       message="Stopping other engines...", progress_pct=5)
                            ok = _start_llama(model_path, job_id=job_id)
                            if not ok:
                                # Revert in-memory state and saved settings. Reload from disk
                                # first so we don't clobber settings the user changed via the
                                # UI while llama-server was starting up.
                                _set_active_engine("ollama")
                                current_settings = _load_settings_file()
                                current_settings["chatEngine"] = "ollama"
                                _persist_settings_file(current_settings)
                                job = _get_engine_startup_job(job_id)
                                if job.get("status") != "failed":
                                    _update_engine_startup_job(job_id, status="failed",
                                                               error="llama-server failed to start")
                        except Exception as e:
                            _set_active_engine("ollama")
                            current_settings = _load_settings_file()
                            current_settings["chatEngine"] = "ollama"
                            _persist_settings_file(current_settings)
                            _update_engine_startup_job(job_id, status="failed", error=str(e))
                    
                    threading.Thread(target=_async_start, daemon=True).start()
                    
                    self.send_response(202)  # Accepted
                    self.send_header("Content-Type", "application/json")
                    self._cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "ok": True, 
                        "async": True,
                        "job_id": job_id, 
                        "engine": engine,
                        "message": "Engine switch started. Poll /api/engine-startup-status?job_id=..."
                    }).encode())
                    return
                else:
                    # Synchronous mode (legacy)
                    ok = _start_llama(model_path)
                    if not ok:
                        _set_active_engine("ollama")
                        settings["chatEngine"] = "ollama"
                        _persist_settings_file(settings)
                        raise RuntimeError("llama-server failed to start. Check that the engine is installed.")
            else:
                # Stop llama-server, start Ollama (fast, always sync)
                _kill_llama()
                if not _is_ollama_running():
                    ok = _start_ollama()
                    if not ok:
                        raise RuntimeError("Ollama failed to start.")
                _set_active_engine("ollama")
                model_file = ""

                # Persist the choice
                settings = _load_settings_file()
                settings["chatEngine"] = engine
                settings["llamaModel"] = model_file if engine == "llama" else settings.get("llamaModel", "")
                _persist_settings_file(settings)

            _log_event(logging.INFO, f"Chat engine switched to '{engine}'", request_context=request_context)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "engine": engine}).encode())
        except Exception as e:
            _log_event(logging.ERROR, "Engine switch failed", request_context=request_context, exc_info=True)
            self.send_response(500)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _start_ollama_endpoint(self):
        """Restart the Ollama chat engine."""
        request_context = self._build_request_context("/api/start-ollama")
        try:
            if _is_ollama_running():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "message": "Chat engine already running."}).encode())
                return
            ok = _start_ollama()
            if ok:
                _log_event(logging.INFO, "Ollama engine started by user", request_context=request_context)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "message": "Chat engine started."}).encode())
            else:
                raise RuntimeError("Ollama failed to start. Check that the engine is installed.")
        except Exception as e:
            _log_event(logging.ERROR, "Failed to start Ollama", request_context=request_context, exc_info=True)
            self.send_response(500)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _generate_image(self):
        """Start an image generation job and return a job_id immediately."""
        request_context = self._build_request_context("/api/generate-image")
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        # Re-scan for models rather than trusting the SD_ENABLED snapshot taken
        # at import time, so a model downloaded after startup (via the model
        # pull / HuggingFace download features) is picked up without a restart.
        if not (SD_BINARY is not None and len(_find_sd_models()) > 0):
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Image generation is not set up. Please run the installer."}).encode())
            return

        # Warn if an Ollama model is loaded in memory (RAM collision risk)
        if _is_ollama_model_loaded():
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Chat engine model is loaded in memory. Please stop/unload it first to free memory for image generation.",
                "needs_stop": True
            }).encode())
            return

        # Clean up stale jobs occasionally
        _cleanup_old_image_jobs()

        job_id = str(uuid.uuid4())
        output_id = str(uuid.uuid4())
        output_dir = os.path.join(SCRIPT_DIR, "chat_data", "generated_images")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_id}.png")

        _register_image_job(job_id)

        # Start generation in a background thread so we can return job_id immediately
        thread = threading.Thread(
            target=_run_sd_generation,
            args=(job_id, payload, output_path),
            daemon=True,
        )
        thread.start()

        _log_event(logging.INFO, "Image generation job started", request_context=request_context)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "job_id": job_id}).encode())

    def _get_image_progress(self):
        """Poll endpoint for image generation progress."""
        query = parse_qs(urlparse(self.path).query)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing job_id parameter."}).encode())
            return

        job = _get_image_job(job_id)
        if not job:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Job not found."}).encode())
            return

        # Build response with progress info
        resp = {
            "status": job.get("status"),
            "step": job.get("step", 0),
            "total_steps": job.get("total_steps", 0),
            "elapsed_ms": job.get("elapsed_ms", 0),
            "eta_ms": job.get("eta_ms"),
            "params": job.get("params", {}),
        }

        if job.get("status") == "done":
            resp["image_b64"] = job.get("image_b64")
            resp["mime_type"] = "image/png"
            # Present only for persisted images (the in-chat tool path), so the
            # transcript can reference a URL instead of storing base64.
            if job.get("image_url"):
                resp["image_url"] = job["image_url"]
        elif job.get("status") == "error":
            resp["error"] = job.get("error", "Unknown error.")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def _get_engine_startup_status(self):
        """Poll endpoint for engine startup progress."""
        query = parse_qs(urlparse(self.path).query)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing job_id parameter."}).encode())
            return

        job = _get_engine_startup_job(job_id)
        if not job:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Job not found."}).encode())
            return

        # Build response with progress info
        elapsed = time.time() - job.get("started_at", time.time())
        resp = {
            "status": job.get("status"),
            "engine": job.get("engine"),
            "model": job.get("model"),
            "phase": job.get("phase"),
            "progress_pct": job.get("progress_pct", 0),
            "message": job.get("message", ""),
            "elapsed_secs": int(elapsed),
        }

        if job.get("status") == "failed":
            resp["error"] = job.get("error", "Unknown error.")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def _pull_model_endpoint(self):
        """Start downloading a model via 'ollama pull'."""
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        
        model_name = payload.get("model", "").strip()
        if not model_name:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Model name required"}).encode())
            return
        
        if not OLLAMA_BIN:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Ollama not installed"}).encode())
            return
        
        _cleanup_old_model_pull_jobs()
        job_id = str(uuid.uuid4())
        _register_model_pull_job(job_id, model_name)
        
        threading.Thread(target=_run_model_pull, args=(job_id, model_name), daemon=True).start()
        
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "job_id": job_id, "model": model_name}).encode())

    def _get_model_pull_status(self):
        """Poll endpoint for model pull progress."""
        query = parse_qs(urlparse(self.path).query)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing job_id parameter"}).encode())
            return

        job = _get_model_pull_job(job_id)
        if not job:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Job not found"}).encode())
            return

        elapsed = time.time() - job.get("started_at", time.time())
        resp = {
            "status": job.get("status"),
            "model": job.get("model"),
            "progress_pct": job.get("progress_pct", 0),
            "message": job.get("message", ""),
            "downloaded": job.get("downloaded", ""),
            "total": job.get("total", ""),
            "elapsed_secs": int(elapsed),
        }
        if job.get("status") == "failed":
            resp["error"] = job.get("error", "Unknown error")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def _hf_download_endpoint(self):
        """Download a model file from HuggingFace."""
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        
        url = payload.get("url", "").strip()
        model_type = payload.get("type", "").lower()  # "gguf" or "safetensors"
        
        if not url:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "URL required"}).encode())
            return
        
        # Validate URL is from HuggingFace. Parse the hostname rather than doing
        # a substring check, which a URL like http://evil.com?x=huggingface.co
        # or http://huggingface.co.evil.com would otherwise slip past.
        parsed_url = urlparse(url)
        hostname = (parsed_url.hostname or "").lower()
        allowed_hosts = ("huggingface.co", "hf.co")
        is_hf_host = any(hostname == h or hostname.endswith("." + h) for h in allowed_hosts)
        if parsed_url.scheme not in ("http", "https") or not is_hf_host:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Only HuggingFace URLs supported"}).encode())
            return
        
        # Extract filename from URL
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = "model.bin"
        
        # Auto-detect type from extension if not provided
        if not model_type:
            if filename.lower().endswith(".gguf"):
                model_type = "gguf"
            elif filename.lower().endswith(".safetensors"):
                model_type = "safetensors"
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Could not detect model type. Use .gguf or .safetensors file"}).encode())
                return
        
        # Determine destination
        models_dir = os.path.join(SCRIPT_DIR, "models")
        dest_path = os.path.join(models_dir, filename)
        
        if os.path.exists(dest_path):
            self.send_response(409)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"File '{filename}' already exists"}).encode())
            return
        
        job_id = str(uuid.uuid4())
        _register_hf_download_job(job_id, url, filename, model_type)
        
        threading.Thread(target=_run_hf_download, args=(job_id, url, dest_path), daemon=True).start()
        
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({
            "ok": True, 
            "job_id": job_id, 
            "filename": filename,
            "type": model_type
        }).encode())

    def _get_hf_download_status(self):
        """Poll endpoint for HuggingFace download progress."""
        query = parse_qs(urlparse(self.path).query)
        job_id = query.get("job_id", [None])[0]
        if not job_id:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing job_id parameter"}).encode())
            return

        job = _get_hf_download_job(job_id)
        if not job:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Job not found"}).encode())
            return

        elapsed = time.time() - job.get("started_at", time.time())
        resp = {
            "status": job.get("status"),
            "filename": job.get("filename"),
            "type": job.get("model_type"),
            "progress_pct": job.get("progress_pct", 0),
            "message": job.get("message", ""),
            "downloaded": _format_bytes(job.get("downloaded_bytes", 0)),
            "total": _format_bytes(job.get("total_bytes", 0)),
            "elapsed_secs": int(elapsed),
        }
        if job.get("status") == "failed":
            resp["error"] = job.get("error", "Unknown error")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    # ── Ollama/llama.cpp Proxy (streaming-aware) ──────────────────────────────
    def _proxy_ollama(self, method):
        """
        Proxy requests from /ollama/* to the active chat engine.
        Transparently translates Ollama API ↔ OpenAI API when llama.cpp is active.
        """
        request_context = self._build_request_context("/ollama")
        ollama_path = self.path[len("/ollama"):]
        with ACTIVE_ENGINE_LOCK:
            active = ACTIVE_ENGINE
        chat_host = _get_chat_host()
        target_url = chat_host + ollama_path

        # Read request body if present
        body = None
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
        payload = {}
        if body:
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}

        if isinstance(payload, dict):
            request_context["model_name"] = payload.get("model", "-")
            request_context["model_stream"] = payload.get("stream", "-")
            request_context["model_temperature"] = payload.get("options", {}).get("temperature", "-")

        try:
            # llama.cpp: synthesise /api/tags from local .gguf files
            if active == "llama" and ollama_path == "/api/tags":
                gguf_models = _find_gguf_models()
                models = [{"name": m["name"], "model": m["name"], "size": 0} for m in gguf_models]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"models": models}).encode())
                return

            # llama.cpp: translate Ollama /api/chat → OpenAI /v1/chat/completions
            if active == "llama" and ollama_path == "/api/chat":
                ollama_req = json.loads(body) if body else {}
                model_name = ollama_req.get("model", "").lower()
                raw_messages = ollama_req.get("messages", [])

                # Models whose chat templates have no system slot — system content
                # must be folded into the first user message instead.
                NO_SYSTEM_SLOT_MODELS = ("gemma",)
                needs_system_fold = any(m in model_name for m in NO_SYSTEM_SLOT_MODELS)

                if needs_system_fold:
                    # Collect and strip all system messages, prepend to first user
                    system_parts = [msg["content"] for msg in raw_messages if msg.get("role") == "system" and msg.get("content")]
                    non_system = [dict(msg) for msg in raw_messages if msg.get("role") != "system"]
                    if system_parts:
                        system_text = "\n\n".join(system_parts)
                        # Find first user message to prepend into
                        for msg in non_system:
                            if msg.get("role") == "user":
                                msg["content"] = system_text + "\n\n" + (msg["content"] or "")
                                break
                        else:
                            # No user message exists yet — insert one
                            non_system.insert(0, {"role": "user", "content": system_text})
                    raw_messages = non_system

                # Merge consecutive same-role messages — gemma (and many other models)
                # require strict user/assistant alternation in their chat templates.
                # Attached images are carried across a merge too, otherwise merging
                # a text turn into an image turn would drop the picture.
                merged_messages = []
                for msg in raw_messages:
                    if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                        merged_messages[-1]["content"] = (merged_messages[-1]["content"] or "") + "\n" + (msg["content"] or "")
                        if msg.get("images"):
                            merged_messages[-1].setdefault("images", [])
                            merged_messages[-1]["images"] = list(merged_messages[-1]["images"]) + list(msg["images"])
                    else:
                        merged_messages.append(dict(msg))

                # Truncate to fit within context window (llama-server uses 4096 by default)
                # Reserve tokens for response and template overhead
                merged_messages, truncation_notice = _truncate_messages_to_context(
                    merged_messages, max_tokens=3500, reserve_for_response=500
                )
                if truncation_notice:
                    _log_event(logging.INFO, f"Context truncated: {truncation_notice}", request_context=request_context)
                    # For Gemma, we need to re-fold the truncation notice into user message
                    if needs_system_fold and len(merged_messages) > 1:
                        # Find and remove the system truncation notice
                        for i, msg in enumerate(merged_messages):
                            if msg.get("role") == "system" and "truncated" in msg.get("content", ""):
                                notice = merged_messages.pop(i)
                                # Prepend to first user message
                                for m in merged_messages:
                                    if m.get("role") == "user":
                                        m["content"] = f"[Note: {notice['content']}]\n\n" + (m["content"] or "")
                                        break
                                break

                # Translate to OpenAI format last, so truncation and merging above
                # both operate on plain string content.
                openai_messages = [_ollama_msg_to_openai(m) for m in merged_messages]

                openai_req = {
                    "model": ollama_req.get("model", "local"),
                    "messages": openai_messages,
                    "stream": True,
                    "temperature": ollama_req.get("options", {}).get("temperature", 0.7)
                }
                if _tools_enabled():
                    openai_req["tools"] = TOOL_DEFINITIONS
                    openai_req["tool_choice"] = "auto"
                target_url = chat_host + "/v1/chat/completions"
                body = json.dumps(openai_req).encode()

            # Ollama speaks tools natively in its own format — arguments arrive
            # as an object in one complete frame, which is already the shape the
            # client expects — so only the request needs enriching, and the
            # response streams through untouched.
            tools_injected = False
            if (active != "llama" and ollama_path == "/api/chat"
                    and _tools_enabled() and isinstance(payload, dict)):
                enriched = dict(payload)
                enriched["tools"] = TOOL_DEFINITIONS
                body = json.dumps(enriched).encode()
                tools_injected = True

            def _open(data):
                r = urllib.request.Request(
                    target_url,
                    data=data,
                    method=method,
                    headers={"Content-Type": self.headers.get("Content-Type", "application/json")}
                )
                # Optional: pass Authorization header if present
                if "Authorization" in self.headers:
                    r.add_header("Authorization", self.headers.get("Authorization"))
                return urllib.request.urlopen(r, timeout=600)

            try:
                response = _open(body)
            except urllib.error.HTTPError as e:
                # Ollama rejects tools outright for models whose template lacks
                # them ("does not support tools"). Retry once without so that
                # enabling the toggle degrades to plain chat instead of breaking
                # it for every model that cannot do tool calls.
                if tools_injected and e.code == 400:
                    _log_event(
                        logging.INFO,
                        "Model rejected tools; retrying without them",
                        request_context=request_context,
                    )
                    response = _open(json.dumps(payload).encode())
                else:
                    raise

            # Send response headers
            self.send_response(response.status)
            is_stream = ("/api/chat" in ollama_path or "/api/generate" in ollama_path)

            for header, value in response.getheaders():
                lower = header.lower()
                if lower not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(header, value)

            self._cors_headers()
            self.end_headers()
            if method == "POST" and ("/api/chat" in ollama_path or "/api/generate" in ollama_path):
                _log_event(
                    logging.INFO,
                    f"Model request succeeded with upstream status {response.status}",
                    request_context=request_context,
                )

            # Buffer partial SSE lines across chunk boundaries. read(4096) can
            # split a frame mid-JSON, and a half-frame silently fails to parse —
            # losing a token of content, or corrupting tool-call arguments,
            # which arrive split across dozens of frames.
            sse_buffer = ""
            # Tool calls stream incrementally: the first frame carries the id and
            # function name, later frames append fragments of the JSON arguments.
            # Accumulate by index and emit once, when the stream signals it's done.
            tool_acc = {}

            def emit(obj):
                """Write one Ollama-format JSONL frame. False if the client is gone."""
                try:
                    self.wfile.write((json.dumps(obj) + "\n").encode())
                    self.wfile.flush()
                    return True
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return False

            def collected_tool_calls():
                calls = []
                for idx in sorted(tool_acc):
                    slot = tool_acc[idx]
                    if not slot.get("name"):
                        continue
                    try:
                        # Ollama represents arguments as an object, OpenAI as a
                        # JSON string, so parse before handing it to the client.
                        args = json.loads(slot.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    calls.append({"function": {"name": slot["name"], "arguments": args}})
                return calls

            def emit_done():
                # Final frame in Ollama's format so clients relying on
                # "done": true don't hang, carrying any tool calls collected.
                final = {"message": {"role": "assistant", "content": ""}, "done": True}
                calls = collected_tool_calls()
                if calls:
                    final["message"]["tool_calls"] = calls
                emit(final)

            # Stream the response in chunks
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break

                # If bridging llama.cpp SSE to Ollama JSONL
                if active == "llama" and is_stream:
                    sse_buffer += chunk.decode(errors="ignore")
                    lines = sse_buffer.split("\n")
                    # Keep the trailing fragment; it completes in a later chunk.
                    sse_buffer = lines.pop()
                    for line in lines:
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            emit_done()
                            return
                        try:
                            j = json.loads(data)
                        except Exception:
                            continue
                        choices = j.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})

                        for tc in delta.get("tool_calls") or []:
                            slot = tool_acc.setdefault(
                                tc.get("index", 0), {"name": "", "arguments": ""}
                            )
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]

                        if choice.get("finish_reason") == "tool_calls":
                            emit_done()
                            return

                        # `or ""` rather than a .get() default: llama.cpp sends an
                        # explicit "content": null on some deltas (e.g. the opening
                        # role-only frame), and the key existing means .get()'s
                        # default never applies. Ollama always sends a string, so
                        # clients doing `content += chunk.message.content` break on null.
                        content = delta.get("content") or ""
                        if content and not emit({
                            "message": {"role": "assistant", "content": content},
                            "done": False,
                        }):
                            _log_event(
                                logging.ERROR,
                                "Client disconnected while streaming response",
                                request_context=request_context,
                                exc_info=True,
                            )
                            return
                else:
                    try:
                        self.wfile.write(chunk)
                        if is_stream:
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        _log_event(
                            logging.ERROR,
                            "Client disconnected while streaming response",
                            request_context=request_context,
                            exc_info=True,
                        )
                        return

        except urllib.error.HTTPError as e:
            _log_event(
                logging.ERROR,
                f"Upstream HTTP error from Ollama: {e.code}",
                request_context=request_context,
                exc_info=True,
            )
            self.send_response(e.code)
            self._cors_headers()
            self.end_headers()
            try:
                self.wfile.write(e.read())
            except:
                pass

        except urllib.error.URLError as e:
            _log_event(
                logging.ERROR,
                "Failed to reach Ollama upstream",
                request_context=request_context,
                exc_info=True,
            )
            self.send_response(502)
            self._cors_headers()
            self.end_headers()
            msg = json.dumps({"error": f"Cannot reach Ollama engine: {str(e.reason)}"})
            self.wfile.write(msg.encode())

        except Exception as e:
            _log_event(
                logging.ERROR,
                "Unhandled proxy error",
                request_context=request_context,
                exc_info=True,
            )
            self.send_response(500)
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


class ThreadedHTTPServer(http.server.HTTPServer):
    """Handle each request in a new thread for concurrent streaming."""
    def process_request(self, request, client_address):
        thread = threading.Thread(target=self._handle, args=(request, client_address))
        thread.daemon = True
        thread.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

HOST_HARDWARE_SPECS = _get_hardware_specs()
LOGGER, LOG_LISTENER = configure_logging()

def open_browser_delayed():
    """Open the browser after a short delay to ensure server is ready."""
    time.sleep(1.0)
    webbrowser.open(f"http://localhost:{CHAT_SERVER_PORT}")

def main():
    ensure_data_dir()

    # Warm up the CPU-sample baseline before the server starts accepting
    # connections, so the first real /api/stats request doesn't pay the
    # ~250ms blocking sample cost itself.
    _get_hw_stats()

    startup_settings = _load_settings_file()
    _set_active_log_mode(startup_settings.get("logMode"))

    # Restore chat engine from saved settings
    saved_engine = startup_settings.get("chatEngine", "ollama")
    _set_active_engine(saved_engine)
    if ACTIVE_ENGINE == "llama":
        model_file = startup_settings.get("llamaModel", "")
        if not model_file:
            gguf_list = _find_gguf_models()
            if gguf_list:
                model_file = gguf_list[0]["file"]
        if model_file:
            model_path = os.path.join(SCRIPT_DIR, "models", model_file)
            _start_llama(model_path)

    # Try to find the local LAN IP
    local_ip = "127.0.0.1"
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
    except Exception:
        pass

    print()
    print("=" * 55)
    print("  Portable AI — Chat Server")
    print("=" * 55)
    print()
    print(f"  Local Access:    http://localhost:{CHAT_SERVER_PORT}")
    print(f"  Network Access:  http://{local_ip}:{CHAT_SERVER_PORT}   <-- Use this on phone/other PC!")
    print(f"  Chat Engine:     {ACTIVE_ENGINE.upper()} ({_get_chat_host()})")
    print()
    print("  All chats auto-save to the USB drive!")
    print("  Press Ctrl+C to shut down.")
    print()
    print("-" * 55)

    server = ThreadedHTTPServer(("0.0.0.0", CHAT_SERVER_PORT), ChatHandler)

    # Open browser in background thread
    if "--no-browser" not in sys.argv:
        threading.Thread(target=open_browser_delayed, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down chat server...")
        server.shutdown()
        print("  Goodbye!")
    finally:
        try:
            LOG_LISTENER.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
