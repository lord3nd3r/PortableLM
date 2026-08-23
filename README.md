# PortableLM

**PortableLM** is a fully air-gapped, zero-dependency, plug-and-play local AI environment
designed to run seamlessly from your **local hard drive** or a **portable USB/SSD**. It
bypasses complex installations — natively executing large language models directly on
your hardware with no internet required after setup.

With a unified architecture, you can initialize your AI models once and choose to keep
them on your system or carry them with you across Windows, macOS, Linux, and Android.

## Core Features

* **No-Install Setup:** Self-contained engine binaries live on the drive. No admin rights, registry edits, package managers, or pip installs. The server itself is pure Python standard library — the only host requirement is **Python 3** (preinstalled on macOS and most Linux distros; on Windows the installer will tell you if it's missing).
* **Cross-Platform:** Uses an intelligent `Shared` volume system — download your 5GB+ AI models *once*, and use them natively on Windows, macOS, Linux, and Android without duplication.
* **Fully Offline:** Runs completely air-gapped after initial setup. Your data never leaves your machine. The install scripts vendor the UI's own JS/CSS/font dependencies (Markdown rendering, syntax highlighting, PDF parsing, icon fonts) into `Shared/vendor/` once, so the chat UI itself never phones home either.
* **Network Proxied UI:** The custom Python HTTP server serves a blazing-fast dark/light mode chat UI. Access the AI from your phone or tablet on the same WiFi — no CORS headaches.
* **Hardware Accelerated:** Natively capitalizes on AVX CPU instructions, NVIDIA CUDA, AMD ROCm, Vulkan, or Apple Metal GPU accelerators dynamically when plugged into different host machines.
* **Dual Chat Backends:** Switch between **Ollama** (multi-model, easy model management) and **llama.cpp** (lower latency, direct GGUF loading) from the settings panel without restarting.
* **In-App Model Acquisition:** Pull any model from the Ollama library by name, or paste a direct HuggingFace `.gguf`/`.safetensors` URL — both download with a live progress bar, no terminal required.
* **AI Image Generation:** Built-in Stable Diffusion image generation via stable-diffusion.cpp — same GPU acceleration, fully offline.
* **Vision — It Can See:** Attach a photo and ask about it. Ships an optional vision model (Qwen2.5-VL 3B) so you can point a camera at a plant, a rash, a part, or a page and get an answer with no signal.
* **Talk To It, It Talks Back:** Offline speech-to-text (whisper.cpp) and text-to-speech (piper). Dictate with the mic button, have replies read aloud, or both. No cloud, no API key.
* **Tool Calling:** Ask "draw me a red barn at sunset" and the model calls the image engine itself, with progress and the result inline. A fixed allowlist of typed actions — never arbitrary commands.
* **File Attachments:** Drop in images, PDFs, or plain text files as chat context.
* **Reasoning-Model Aware:** Models that emit `<think>...</think>` reasoning traces (DeepSeek-R1-style) get their reasoning rendered in a collapsible "Thought Process" section instead of cluttering the reply.

---

## System Requirements

- **Storage:** USB 3.0+ flash drive or SSD with at least **8 GB** free (16 GB recommended).
  The optional voice engines add ~250 MB, and the vision model ~2.8 GB.
- **RAM:** At least **8 GB** for 2–4B models, **16 GB** for 9–12B models.
- **Python 3** on the host machine. Nothing is installed via pip — the server uses only
  the standard library — but the interpreter itself must be present.

---

## Folder Architecture

```text
[PortableLM Drive]
 ├── 📁 Android    # Native Android (Termux) installers & launchers
 ├── 📁 Linux      # Native Ubuntu/Debian offline installers & launchers
 ├── 📁 Mac        # Native macOS offline installers & launchers
 ├── 📁 Windows    # Native Windows offline automatic UI menus
 └── 📁 Shared     # Unified Data System
      ├── 📁 bin         (Isolated executables: ollama, llama-server, sd, piper, whisper)
      ├── 📁 chat_data   (Conversation history, settings.json, generated images)
      ├── 📁 config      (models.json catalog, vendor manifest, tool chat templates)
      ├── 📁 logs        (Rotating structured server log — see Logging below)
      ├── 📁 models      (GGUF/safetensors weights, plus voices/ and whisper/ subfolders)
      ├── 📁 scripts     (Shared installer helpers used by every platform's script)
      └── 📁 vendor      (Vendored offline copies of the UI's JS/CSS/font dependencies)
```

Every installer (Windows/Mac/Linux/Android) reads the **same**
`Shared/config/models.json` catalog, so the model list only needs to be updated in one
place to show up everywhere.

---

## AI Model Library

The installer's model catalog (`Shared/config/models.json`) currently offers, for
desktop platforms (Windows/Mac/Linux):

| # | Model | Size | Notes |
|---|---|---|---|
| 1 | **Gemma 2 2B Abliterated** | ~1.6 GB | Recommended default — fast on almost any hardware |
| 2 | **Gemma 4 E4B Ultra Uncensored Heretic** | ~5.34 GB | Aggressively uncensored fine-tune |
| 3 | **Qwen 3.5 9B Uncensored Aggressive** | ~5.2 GB | Larger reasoning model |
| 4 | **NemoMix Unleashed 12B** | ~7.0 GB | Heavyweight — needs 16GB+ RAM |
| 5 | **Dolphin 2.9 Llama 3 8B** | ~4.9 GB | Uncensored, Llama-3-based |
| 6 | **Phi-3.5 Mini 3.8B** | ~2.2 GB | Lightweight, standard (non-uncensored) tune |
| 7 | **Qwen2.5-VL 3B (Vision)** | ~2.8 GB | **Sees images.** Attach a photo and ask about it. llama.cpp engine only |

> **Why the vision model is llama.cpp only:** vision models ship as two files — the
> language model and a separate `mmproj` projector — and llama-server loads the pair via
> `--mmproj`. Importing a bare GGUF into Ollama can't attach the projector, so it would
> look like it worked and then silently ignore every image. For vision under Ollama,
> pull one by name instead (e.g. `qwen2.5vl:3b`) using the in-app model pull.

Android gets its own lighter catalog (Gemma 2 2B, SmolLM2 1.7B, Qwen2.5 1.5B, Phi-3.5
Mini, Qwen 3.5 9B for 12GB+ devices). Image generation ships with **CyberRealistic v3.3**
(SD 1.5-based, ~1.99 GB) as its preset model.

Beyond the catalog, you can:
- Type **[C]** during install to add any HuggingFace GGUF URL.
- Drop any `.gguf`/`.safetensors` file straight into `Shared/models/` — works with both
  Ollama and llama.cpp without re-running the installer.
- Use the chat UI's own **model pull** (searches the Ollama library by name) or
  **HuggingFace direct download** boxes in the settings panel to fetch a model with a
  live progress bar, no terminal needed.

---

## Chat Backend

PortableLM supports two chat engines, switchable at runtime from the ⚙ settings panel:

| Backend | Description |
|---|---|
| **Ollama** *(default)* | Full-featured model manager. Supports multiple loaded models, easy switching, `ollama pull` for new models. |
| **llama.cpp** | Direct GGUF inference. Lower overhead, no model pull required — just drop a `.gguf` into `Shared/models`. Startup can take up to ~2 minutes on CPU-only hardware; a progress bar tracks it. |

The selected backend and model are persisted across restarts — even a page reload
mid-switch will show the intended engine, since the choice is saved before the slow part
starts. The install script downloads both engine binaries automatically, selecting the
best GPU build for your hardware (CUDA / ROCm / Vulkan / CPU).

- **Image generation works alongside either engine** — it's only blocked while an
  **Ollama** model is actively loaded in RAM (shared-memory conflict on typical consumer
  hardware); llama.cpp runs independently.
- **llama.cpp auto-selects a free port** — no conflicts if port 8080 is already in use on
  your system; it falls back to any free OS-assigned port if the whole default range is
  occupied.
- Some model families need special handling that PortableLM does automatically: Gemma
  models have no "system" slot in their chat template, so your system prompt is folded
  into your first message instead; Llama 2/3 and Mistral-Instruct get an explicit
  chat-template flag so token formatting doesn't drift.

---

## Chat UI Features

- **Streaming responses** with markdown rendering, syntax-highlighted code blocks (with
  a one-click copy button), and light/dark theme.
- **Reasoning traces:** if a model wraps its internal reasoning in `<think>...</think>`
  tags, that section renders as a collapsible "Thought Process" panel instead of mixing
  into the visible answer — including while it's still streaming.
- **File attachments:** images (with an automatic warning if your selected model isn't a
  vision model), PDFs (parsed client-side, no server upload), and plain text files.
- **Message actions:** like/dislike rating, read aloud, per-conversation history saved
  automatically.
- **Model picker** shows each model's on-disk size and flags vision-capable models.
- **Settings panel:** global system prompt, temperature, log verbosity
  (`errors_only`/`all`), engine switch with live startup progress, voice and speech
  pace, tool calling, model pull, and HuggingFace direct download — all in one place.
- **Image generation panel:** prompt/negative prompt, steps, CFG scale, resolution, seed,
  and sampler, with a live step-by-step progress bar and ETA.

---

## Voice & Tools

All three run locally, with no cloud service and no API key.

### Speaking to it

The installer adds **whisper.cpp** plus the `base.en` model (~160 MB). A mic button
appears in the message bar — click to record, click to stop, and the transcript is
appended to whatever you've already typed. Transcription runs at roughly 3x realtime on
CPU.

> **Two limits worth knowing.** Browsers only expose the microphone in a *secure
> context*, so dictation works on the machine running the server (`localhost` counts) but
> **not from your phone over plain `http://` on the LAN** — the button is hidden there
> rather than failing on click. And whisper.cpp publishes no macOS binary, so voice input
> is unavailable on Mac unless you build `whisper-cli` yourself and drop it into
> `Shared/bin/whisper-mac/`, which the server will pick up automatically.

### It speaking back

The installer adds **piper** plus a British English voice (~90 MB). Every assistant
message gets a speaker button, and "Speak replies automatically" in settings turns it
into a hands-free loop. Synthesis runs at roughly 7–11x realtime, so a long reply is
spoken within a couple of seconds of finishing.

**More voices** are available from the settings panel — 17 curated options covering
British, Northern English and American accents in both genders, plus Spanish, French,
German, Italian, Portuguese, Russian, Mandarin and Hindi. Pick one and hit Download
(~63 MB each, ~120 MB for the high-quality ones) with a live progress bar; Remove frees
the space again. Everything is fetched from the same offline-capable piper voice library,
and the last remaining voice can't be removed, so read-aloud never breaks itself.

Markup is stripped before speaking — reasoning traces are skipped, code blocks are
announced rather than read character by character, and links keep their text but drop
the URL.

### Tool calling

Off by default; enable it in the settings panel. Once on, asking for a picture makes the
model call the image engine itself:

> **You:** draw me a red barn at sunset
> **Assistant:** *Generating image... step 12/20* → the image appears in the conversation

| Engine | What's needed |
|---|---|
| **Ollama** | Nothing — takes effect immediately. Registry models ship tool-capable templates. |
| **llama.cpp** | Re-apply the engine after toggling. llama-server picks its chat template at startup, and PortableLM chooses one per model: a model whose own template already asks for `<tool_call>` JSON keeps it, everything else is handed a tool-capable template (many abliterated GGUFs ship a stripped template with no tool support, and some declare tools in a syntax llama-server won't parse — either way the model would otherwise never call anything). |

Models that can't do tool calls keep chatting normally — the request is retried without
tools rather than failing.

**On safety:** the model can only invoke a fixed allowlist of typed actions (currently
image generation). There is deliberately no generic "run this command" tool. This matters
here specifically: the catalog ships abliterated models with refusal training removed,
and dictated input is frequently misheard, so the worst outcome of a bad call should be a
wrong picture — not a wrong shell command.

---

## Quick Start

### Step 1: Initialize the Engine

Run the install script for your OS:

| OS | Command |
|---|---|
| **Windows** | Double-click `Windows/install.bat` |
| **macOS** | Open Terminal -> drag `Mac/install.command` -> Enter |
| **Linux** | `bash Linux/install.sh` |
| **Android** | Open Termux -> `bash Android/install.sh` |

> **Note:** This downloads the execution engines to `Shared/bin` — Ollama, llama.cpp,
> Stable Diffusion, and the optional voice engines (piper for speech, whisper for
> dictation) — plus the UI's offline vendor assets. Each engine is skipped gracefully if
> unavailable for your platform, so a failed optional download never blocks the install.

### Step 2: Download AI Models

The install script presents the model catalog interactively (see **AI Model Library**
above) and lets you pick one, several, or a custom HuggingFace URL. You can also skip
this and pull/download a model later from inside the chat UI itself, or manually drop
`.gguf`/`.safetensors` files into `Shared/models`.

### Step 3: Launch

| OS | Command |
|---|---|
| **Windows** | `Windows/start-fast-chat.bat` |
| **macOS** | `Mac/start.command` |
| **Linux** | `bash Linux/start.sh` |
| **Android** | `bash Android/start.sh` |

The engine spins up and your browser opens the locally-served Chat UI at
`http://localhost:3333`.

### Uninstalling

Each platform has a matching uninstall script (`uninstall.sh` / `uninstall.command` /
`uninstall.bat`) that removes the downloaded engine binaries. Your chat history, settings,
and models are left untouched — delete `Shared/models` and `Shared/chat_data` yourself if
you want a completely clean slate.

---

## Local Disk Installation

Works beautifully as a lightweight local AI setup too:

1. Clone this repo to any folder on your drive.
2. Navigate to your OS folder (Windows/Mac/Linux).
3. Run the install script and choose your models.
4. Run the start script.

Running from an internal SSD is significantly faster than USB — near-instant model loading.

---

## Android (Termux)

Run AI **directly on your phone** — no PC required.

**Requirements:**
- Termux from F-Droid (not Play Store)
- 6 GB+ RAM (8 GB+ recommended)
- WiFi/data for initial setup only
- ARM64 processor

**Setup:**
```bash
# Copy PortableLM to your device, then in Termux:
bash Android/install.sh
# Select your model (Gemma 2 2B recommended)
```

**Launch:**
```bash
bash Android/start.sh
```

**Tips:**
- Run `termux-wake-lock` first to prevent Android from killing the process
- Keep Termux in foreground for best performance
- Close other apps to free RAM
- Use the 2B model on devices under 12 GB RAM
- Plug in charger — LLM inference drains battery
- Expect ~3-10 tokens/sec on 2B (vs 30-50+ on PC with GPU)

---

## LAN Mobile Access

Use your PC's AI from your phone on the couch:

1. PC running the start script + phone on same WiFi.
2. Terminal shows a **Network Access** IP (e.g., `http://192.168.1.15:3333`).
3. Open that URL on your phone browser.

> If pages don't load, check that your firewall (Windows Firewall, `ufw`, etc.) allows
> port `3333`.

---

## Logging

The server writes a detailed, structured log to `Shared/logs/chat_server.log`
(auto-rotated at 10 MB) on every run — this is always on and is the first place to look
if something misbehaves. The **Log Mode** setting in the settings panel
(`errors_only` vs `all`) controls verbosity, not whether logging happens at all. The
terminal window additionally shows a live color-coded one-line summary per request.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Script closes instantly (Windows) | Windows App Execution Aliases conflict. Run via cmd or as Admin. |
| "Engine Not Found" | Run the install script before the start script. |
| Slow generation | Model too large for your RAM. Use the Gemma 2 2B model. |
| Chat error with llama.cpp | Ensure a `.gguf` file is selected in the settings panel before chatting. |
| llama.cpp returns 400 | Usually a model mismatch, or two consecutive messages from the same role confusing the template. Re-select the model in settings and click Apply. |
| llama.cpp option reverts to Ollama | The engine takes up to ~2 minutes to load on CPU-only machines — a progress bar tracks it. Wait for it; the selection is saved immediately and will survive a restart. |
| "llama-server failed to start" | Port conflict or missing binary. Run the install script again to re-download the engine. |
| SD image gen button greyed out | No image model found — pull/download one, or drop a `.safetensors` file into `Shared/models`. No restart needed; the server re-scans on every generation request. |
| Image gen refuses to start with an Ollama model loaded | Expected — click "Unload" next to the chat engine status to free RAM, then try again. |
| Attached a photo but the model ignored it | The model has to be a vision model. Install Qwen2.5-VL from the catalog (llama.cpp engine), or pull one under Ollama. The model picker shows an eye icon for vision-capable models. |
| No mic button | Needs both the whisper engine installed and a secure browser context. Over plain `http://` on a LAN IP browsers block microphone access entirely, so it only appears on the machine running the server. Unavailable on macOS — no upstream whisper binary. |
| No speaker button / voice settings missing | The piper engine or its voice isn't installed. Re-run the installer; a half-downloaded voice is deleted rather than offered. |
| Asked for an image but got only text | Enable "Tool calling" in settings. On llama.cpp you must also re-apply the engine afterwards, since the tool-capable chat template is chosen when llama-server starts. |
| Check what actually went wrong | Open `Shared/logs/chat_server.log` — every request and error is logged there with full context, regardless of the Log Mode setting. |

---

## License

MIT

---

> *PortableLM — your AI, your hardware, zero cloud.*
