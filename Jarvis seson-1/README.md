# 🤖 JARVIS — Season 1

A local, offline-first voice assistant built from scratch in Python — wake-word detection, speech-to-text, a local LLM brain, text-to-speech, persistent memory, live web tools, browser automation, and a mobile remote control. No cloud AI APIs required: everything runs on your own machine through **Ollama** and **faster-whisper**.

![Jarvis Architecture](docs/architecture.png)

> Diagram legend: solid boxes are pipeline stages, each one implemented by a specific file in this repo. Dashed lines mark optional / model-dependent connections. See [Architecture](#-architecture) below for the full breakdown.

---

## 📦 What's in Season 1

Season 1 is nine "episodes," each a self-contained Python script that adds one capability on top of the last. You can run any file independently — they don't import each other.

| # | File | Adds |
|---|------|------|
| 5 | `jarvis_brain.py` | Wake word ("Hey Jarvis") + Whisper STT + Ollama LLM + multilingual TTS |
| 6 | `jarvis_memory.py` | Everything in Ep. 5, plus persistent "remember this" / "what's my …" memory |
| 7 | `jarvis_livedata.py` | Agent-style router: Wikipedia lookups, DuckDuckGo live search, OS app launching |
| 8 | `jarvis_browser_automation.py` | Selenium-driven Chrome control (YouTube/Google search-and-open), spacebar-or-voice activation |
| 9 | `jarvis_body.py` | Streamlit "HUD" GUI — visual chat log, system diagnostics, mic button |
| — | `jarvis_mobile_control/app.py` | Flask + ngrok server so you can control Jarvis from your phone |
| — | `test_mic.py` | Standalone microphone level tester |
| — | `test_wake.py` | Standalone wake-word score tester |
| — | `voice.onnx` / `voice.onnx.json` | A [Piper](https://github.com/rhasspy/piper) neural TTS voice model (German, "Thorsten"), bundled as a **reserved asset for a future episode** — no current script loads it yet (TTS today is handled by `pyttsx3`) |

---

## 🏗️ Architecture

```
 Mic ──▶ Wake Word ──▶ Whisper STT ──▶ Jarvis Agent Router ──▶ pyttsx3 TTS ──▶ Spoken reply
                        (faster-whisper)         │
                                 ┌────────────────┼────────────────┐
                                 ▼                ▼                ▼
                         Ollama Chat Model   Memory Store      Tool Belt
                          (qwen2.5:1.5b)   (jarvis_memory.json)  Wikipedia · DuckDuckGo
                                                                  Selenium Browser · OS apps

 Phone ──▶ Mobile Control UI ──▶ Flask Server (app.py) ──▶ same Ollama model (qwen2.5:1.5b)
                                        (tunneled with pyngrok)
```

Every episode follows the same shape: **capture audio → transcribe → route the text (keyword rules first, LLM as fallback) → act or answer → speak the reply.** Later episodes add more "tools" the router can call before falling back to the LLM.

---

## 🧠 Models used

| Model | Used for | Used in |
|---|---|---|
| **`qwen2.5:1.5b`** (via [Ollama](https://ollama.com), local) | The conversational "brain" — free-form Q&A and chit-chat that isn't handled by a keyword rule | `jarvis_body.py`, `jarvis_brain.py`, `jarvis_memory.py`, `jarvis_livedata.py`, `jarvis_browser_automation.py`, `jarvis_mobile_control/app.py` |
| **`faster-whisper`** — `tiny` or `base` size, `int8` on CPU | Speech-to-text, with automatic language detection | `jarvis_body.py` (tiny), `jarvis_brain.py` (tiny), `jarvis_memory.py` (base), `jarvis_livedata.py` (base), `jarvis_browser_automation.py` (base) |
| **`openWakeWord`** — `hey_jarvis` pretrained model (ONNX) | Always-on wake-word detection ("Hey Jarvis") without sending audio anywhere | `jarvis_brain.py`, `jarvis_memory.py` |
| **`pyttsx3`** (system TTS: SAPI5 / NSSpeechSynthesizer / espeak) | Text-to-speech output, auto-picks an installed voice matching the detected language | all voice-enabled scripts |
| **Piper `voice.onnx`** (bundled, German "Thorsten" voice) | Not yet wired into any script — reserved for a future neural-TTS episode | *(none yet)* |

> **Note:** Ollama and `qwen2.5:1.5b` must be installed and pulled locally — see [Setup](#-setup).

---

## ⚙️ Commands each script understands

These are simple keyword matches checked *before* the request is handed to the LLM, so they respond instantly and don't depend on the model.

| Say / type | Script(s) | Effect |
|---|---|---|
| "open youtube" | `jarvis_body.py`, `jarvis_livedata.py`, mobile UI | Opens youtube.com |
| "open google" | `jarvis_body.py`, `jarvis_livedata.py`, mobile UI | Opens google.com |
| "open calculator" | `jarvis_body.py`, `jarvis_livedata.py`, `jarvis_browser_automation.py`, mobile UI | Launches `calc.exe` |
| "open notepad" | `jarvis_livedata.py`, `jarvis_browser_automation.py` | Launches `notepad.exe` |
| "open chrome" | `jarvis_livedata.py`, `jarvis_browser_automation.py` | Launches Chrome |
| "search youtube for …" / "…on youtube" | `jarvis_browser_automation.py` | Selenium opens a YouTube search |
| "search google for …" | `jarvis_browser_automation.py` | Selenium opens a Google search |
| "what is …" / "who is …" / "define …" | `jarvis_livedata.py` | Wikipedia summary (2 sentences) |
| "weather" / "search …" / "latest …" / "news" | `jarvis_livedata.py` | DuckDuckGo top-3 live results |
| "my `<thing>` is `<value>`" | `jarvis_memory.py` | Saves a fact to `jarvis_memory.json` |
| "what's my `<thing>`" | `jarvis_memory.py` | Recalls a saved fact |
| `[SPACEBAR]` | `jarvis_browser_automation.py` | Instant activation, no wake word needed |
| anything else | all | Falls back to the `qwen2.5:1.5b` chat model |

---

## 🚀 Setup

### 1. Prerequisites
- **Python 3.10–3.11** (recommended for `faster-whisper` / `openwakeword` wheel compatibility)
- **[Ollama](https://ollama.com/download)** installed and running
- A working microphone
- Windows is assumed for the OS-launch commands (`explorer`, `calc.exe`, `notepad.exe`); adapt those lines for macOS/Linux if needed
- **Google Chrome** installed, for `jarvis_browser_automation.py`

### 2. Pull the local LLM
```bash
ollama pull qwen2.5:1.5b
ollama serve      # if it isn't already running as a service
```

### 3. Install Python dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install streamlit numpy sounddevice faster-whisper pyttsx3 ollama psutil ^
    openwakeword wikipedia duckduckgo-search selenium webdriver-manager keyboard ^
    flask pyngrok
```
*(split across lines with `^` for Windows CMD — on macOS/Linux use `\` or one line)*

### 4. Run any episode
```bash
python jarvis_brain.py                 # Episode 5 — wake word + STT + LLM + TTS
python jarvis_memory.py                # Episode 6 — + persistent memory
python jarvis_livedata.py              # Episode 7 — + Wikipedia/web search agent
python jarvis_browser_automation.py    # Episode 8 — + Selenium browser control
streamlit run jarvis_body.py           # Episode 9 — HUD GUI
```

### 5. Run the mobile control server
```bash
cd jarvis_mobile_control
python app.py
```
Then open the printed `http://<your-lan-ip>:5000` on your phone (same Wi-Fi), or configure `pyngrok` with an auth token to get a public URL reachable from anywhere.

### 6. Calibrate first (recommended)
```bash
python test_mic.py     # confirm your mic is picked up and levels are sane
python test_wake.py    # confirm "Hey Jarvis" triggers a high enough score
```

---

## 📁 Project structure
```
Jarvis seson-1/
├── jarvis_brain.py                 # Ep.5 — wake word + STT + LLM + TTS (core loop)
├── jarvis_memory.py                # Ep.6 — + persistent fact memory (JSON)
├── jarvis_livedata.py              # Ep.7 — + Wikipedia + DuckDuckGo + OS tools (agent mode)
├── jarvis_browser_automation.py    # Ep.8 — + Selenium Chrome control, spacebar activation
├── jarvis_body.py                  # Ep.9 — Streamlit HUD GUI
├── jarvis_mobile_control/
│   ├── app.py                      # Flask server + pyngrok tunnel
│   └── templates/
│       └── index.html              # Mobile web remote (buttons + free-text box)
├── test_mic.py                     # Mic input level tester
├── test_wake.py                    # Wake-word score tester
├── voice.onnx                      # Piper neural TTS voice (reserved, unused so far)
└── voice.onnx.json                 # Piper voice config/metadata
```

---

## 🛠️ Troubleshooting

- **"My brain is offline, sir."** → Ollama isn't running, or `qwen2.5:1.5b` hasn't been pulled. Run `ollama serve` and `ollama pull qwen2.5:1.5b`.
- **Wake word never triggers** → run `test_wake.py` and watch the printed scores while you say "Hey Jarvis"; if scores stay near 0, check `test_mic.py` first to confirm the right input device is selected.
- **`WinError 1314` on first run** → this is a Windows symlink-privilege error from Hugging Face's model cache; `jarvis_brain.py` already sets `HF_HUB_DISABLE_SYMLINKS=1` to avoid it — apply the same env var if you hit it in another script.
- **No sound / wrong voice** → `pyttsx3` uses whatever TTS voices are installed on your OS; if no voice matches the detected language, scripts fall back to your default system voice.
- **Selenium can't start Chrome** → `webdriver-manager` downloads the matching ChromeDriver automatically the first time; make sure Chrome itself is installed and you have an internet connection for that first download.

---

## 🗺️ Roadmap (beyond Season 1)
- Wire up the bundled Piper `voice.onnx` for fully offline neural TTS
- Cross-platform command handling (replace the Windows-only `subprocess` calls)
- Season 2: tool-calling with structured function schemas instead of keyword matching

---

## ⚠️ Disclaimer
This is a personal/educational project. OS-automation commands (`subprocess.Popen`, `keyboard`, Selenium) execute real actions on your machine — review the code before running it, especially the mobile server, which listens on your LAN.

## 📄 License
Add your preferred license here (e.g. MIT) before publishing.
