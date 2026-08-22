# PortableLM

**PortableLM** is a fully air-gapped, zero-dependency, plug-and-play local AI environment
designed to run seamlessly from your **local hard drive** or a **portable USB/SSD**. It
bypasses complex installations — natively executing large language models directly on
your hardware with no internet required after setup.

With a unified architecture, you can initialize your AI models once and choose to keep
them on your system or carry them with you across Windows, macOS, Linux, and Android.

## Core Features

* **Zero Dependency Setup:** Ships with portable Python and isolated engine binaries. No system permissions, registry edits, or package managers required.
* **Cross-Platform:** Uses an intelligent `Shared` volume system — download your 5GB+ AI models *once*, and use them natively on Windows, macOS, Linux, and Android without duplication.
* **Fully Offline:** Runs completely air-gapped after initial setup. Your data never leaves your machine. The install scripts vendor the UI's own JS/CSS/font dependencies (Markdown rendering, syntax highlighting, PDF parsing, icon fonts) into `Shared/vendor/` once, so the chat UI itself never phones home either.
* **Network Proxied UI:** The custom Python HTTP server serves a blazing-fast dark/light mode chat UI. Access the AI from your phone or tablet on the same WiFi — no CORS headaches.
* **Hardware Accelerated:** Natively capitalizes on AVX CPU instructions, NVIDIA CUDA, AMD ROCm, Vulkan, or Apple Metal GPU accelerators dynamically when plugged into different host machines.
* **Dual Chat Backends:** Switch between **Ollama** (multi-model, easy model management) and **llama.cpp** (lower latency, direct GGUF loading) from the settings panel without restarting.
* **In-App Model Acquisition:** Pull any model from the Ollama library by name, or paste a direct HuggingFace `.gguf`/`.safetensors` URL — both download with a live progress bar, no terminal required.
* **AI Image Generation:** Built-in Stable Diffusion image generation via stable-diffusion.cpp — same GPU acceleration, fully offline.
* **File Attachments:** Drop in images (auto-detects whether your selected model supports vision), PDFs, or plain text files as chat context.
* **Reasoning-Model Aware:** Models that emit `<think>...</think>` reasoning traces (DeepSeek-R1-style) get their reasoning rendered in a collapsible "Thought Process" section instead of cluttering the reply.

---

## System Requirements

- **Storage:** USB 3.0+ flash drive or SSD with at least **8 GB** free (16 GB recommended).
- **RAM:** At least **8 GB** for 2–4B models, **16 GB** for 9–12B models.

---

## Folder Architecture

```text
[PortableLM Drive]
 ├── 📁 Android    # Native Android (Termux) installers & launchers
 ├── 📁 Linux      # Native Ubuntu/Debian offline installers & launchers
 ├── 📁 Mac        # Native macOS offline installers & launchers
 ├── 📁 Windows    # Native Windows offline automatic UI menus
 └── 📁 Shared     # Unified Data System
      ├── 📁 bin         (Isolated executables: ollama, llama-server, sd image engine)
      ├── 📁 chat_data   (Cross-platform persistent conversation history + settings.json)
      ├── 📁 config      (models.json model catalog, vendor asset manifest)
      ├── 📁 logs        (Rotating structured server log — see Logging below)
      ├── 📁 models      (HuggingFace GGUF/safetensors weights, dropped in or downloaded)
      ├── 📁 scripts     (Shared installer helpers used by every platform's script)
      ├── 📁 vendor      (Vendored offline copies of the UI's JS/CSS/font dependencies)
      └── 📁 python      (Isolated portable python environment, where applicable)
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
- **Message actions:** like/dislike rating, per-conversation history saved automatically.
- **Model picker** shows each model's on-disk size and flags vision-capable models.
- **Settings panel:** global system prompt, temperature, log verbosity
  (`errors_only`/`all`), engine switch with live startup progress, model pull, and
  HuggingFace direct download — all in one place.
- **Image generation panel:** prompt/negative prompt, steps, CFG scale, resolution, seed,
  and sampler, with a live step-by-step progress bar and ETA.

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

> **Note:** This downloads the execution engines (Ollama, llama.cpp, and optionally the
> Stable Diffusion binary) to `Shared/bin`, plus the UI's offline vendor assets.

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
| Check what actually went wrong | Open `Shared/logs/chat_server.log` — every request and error is logged there with full context, regardless of the Log Mode setting. |

---

## License

MIT

---

> *PortableLM — your AI, your hardware, zero cloud.*
