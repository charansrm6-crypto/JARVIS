import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyttsx3
import ollama
import json
import os
import re
import time
import openwakeword
from openwakeword.model import Model

print("🤖 JARVIS - (MEMORY MODE, MULTI-LANGUAGE)")
print("=" * 45)

MEMORY_FILE = "jarvis_memory.json"

# Load saved memory facts from a JSON file if it exists.
# If the file is missing or invalid, return an empty dictionary.
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

# Save the current memory facts to a JSON file so they persist between runs.
def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


facts = load_memory()

# facts is the in-memory storage of saved user data like "my name is ...".
print("🔄 Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

print("🔄 Initializing Voice Engine...")

# ---------------------------------------------------------------------------
# NEW: language -> TTS voice lookup, built once at startup.
# pyttsx3 exposes whatever voices are installed on the OS (SAPI5 on Windows,
# NSSpeechSynthesizer on macOS, espeak on Linux). We inspect each voice's
# metadata and index it by language code so speak() can pick a matching
# voice for whatever language Whisper detects. If no matching voice exists
# on this machine, we just fall back to the engine's default voice --
# the text will still be spoken, just with the default accent.
# ---------------------------------------------------------------------------
def build_voice_map():
    voice_map = {}
    try:
        probe_engine = pyttsx3.init()
        for voice in probe_engine.getProperty('voices'):
            langs = []
            try:
                for l in (voice.languages or []):
                    if isinstance(l, bytes):
                        l = l.decode("utf-8", errors="ignore")
                    langs.append(str(l).lower())
            except Exception:
                pass
            # Some drivers put the language in the voice id/name instead
            langs.append(str(voice.id).lower())
            langs.append(str(voice.name).lower())

            for code in langs:
                # normalize things like "en_US", "en-us", "english" -> "en"
                short = code[:2] if len(code) >= 2 and code[:2].isalpha() else code
                voice_map.setdefault(short, voice.id)
            voice_map.setdefault(str(voice.name).lower(), voice.id)
        probe_engine.stop()
        del probe_engine
    except Exception as e:
        print(f"⚠️ Could not enumerate TTS voices: {e}")
    return voice_map


VOICE_MAP = build_voice_map()
print(f"🗣️ Detected {len(VOICE_MAP)} language/voice mappings for TTS.")

# speak() creates a fresh text-to-speech engine for each utterance.
# This avoids issues where the engine can get stuck after repeated use.
def speak(text, language=None):
    """Create a fresh pyttsx3 engine for every utterance.

    pyttsx3's SAPI5 driver (Windows) gets stuck in a 'busy' state after the
    first runAndWait() call if you reuse the same engine instance across
    multiple say()/runAndWait() cycles -- the second and later calls to
    say() silently do nothing (no exception raised). Re-initializing the
    engine each time avoids that stuck state.

    NEW: optional `language` (an ISO-639-1-ish code such as "en", "es",
    "hi", "te" ...) picks a matching installed voice if one is available,
    so replies come out in a voice suited to the detected language.
    """
    if not text:
        return
    print(f"🔊 Speaking: {text[:80]}{'...' if len(text) > 80 else ''}")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)

        if language:
            voice_id = VOICE_MAP.get(str(language).lower())
            if voice_id:
                engine.setProperty('voice', voice_id)
            else:
                print(f"ℹ️ No installed TTS voice found for language '{language}', using default voice.")

        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
    except Exception as e:
        print(f"❌ TTS error: {e}")

chat_history = [
    {"role": "system", "content": "You are Jarvis, a helpful AI assistant. Keep answers short and clear. Always reply in the same language the user spoke in."}
]

samplerate = 16000

# Memory Action Handler patterns
# "my <thing> is <value>"  -> save
# "what is my <thing>" / "what's my <thing>" -> recall
SAVE_PATTERN = re.compile(r"\bmy (.+?) is (.+)", re.IGNORECASE)
RECALL_PATTERN = re.compile(r"\bwhat(?:'s| is) my (.+)", re.IGNORECASE)


def handle_memory(txt):
    """Return a spoken reply if this was a memory save/recall command, else None."""
    recall_match = RECALL_PATTERN.search(txt)
    if recall_match:
        key = recall_match.group(1).strip().rstrip("?").strip()
        value = facts.get(key)
        if value:
            return f"Your {key} is {value}."
        return f"I don't have your {key} saved yet."

    save_match = SAVE_PATTERN.search(txt)
    if save_match:
        key = save_match.group(1).strip()
        value = save_match.group(2).strip().rstrip(".").strip()
        facts[key] = value
        save_memory(facts)
        return f"Got it, I'll remember your {key} is {value}."

    return None


print("✅ All loaded!")
print(f"🧠 Remembered facts so far: {facts}")
print("🎤 Say 'Hey Jarvis' to activate. (Any language Whisper supports will be auto-detected.)\n")

# Wake word model and audio settings
print("📥 Checking wake word model files...")
openwakeword.utils.download_models(["hey_jarvis"])
owwModel = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

chunk_size = 1280
wake_detected = False

_debug_counter = 0

def wake_callback(indata, frames, time, status):
    global wake_detected, _debug_counter

    if status:
        print(f"⚠️ Audio status: {status}")

    audio_int16 = indata[:, 0]
    level = np.max(np.abs(audio_int16.astype(np.float32))) / 32768.0

    preds = owwModel.predict(audio_int16)
    best_name, best_score = max(preds.items(), key=lambda kv: kv[1])

    _debug_counter += 1
    if _debug_counter % 10 == 0:
        print(f"\r🔊 mic level: {level:.4f}  best={best_name}:{best_score:.3f}", end="", flush=True)

    if best_name == "hey_jarvis" and float(best_score) > 0.22:
        print(f"\n✅✅✅ WAKE WORD DETECTED! ({best_name} score: {float(best_score):.2f}) ✅✅✅\n")
        wake_detected = True
        owwModel.reset()

while True:
    try:
        wake_detected = False

        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            blocksize=chunk_size,
            callback=wake_callback
        ):
            print("👂 Listening for 'Hey Jarvis'...", end="", flush=True)
            while not wake_detected:
                sd.sleep(100)

        print("🎤 Wake word heard. Listening... (speak loudly)")
        audio = sd.rec(int(4 * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()

        level = np.max(np.abs(audio))
        print(f"🔊 Audio Level: {level:.4f}")

        # Lowered from 0.05 -> 0.02 to match typical mic levels
        if level > 0.02:
            print("🎙️ Sound detected! Transcribing...\n")

            # NEW: language=None makes faster-whisper auto-detect the
            # spoken language instead of forcing English. info.language
            # then tells us what it detected (e.g. "en", "es", "hi").
            segments, info = whisper_model.transcribe(
                audio.flatten(),
                beam_size=5,
                language=None,
                vad_filter=True
            )

            detected_lang = getattr(info, "language", None)
            lang_prob = getattr(info, "language_probability", None)
            if detected_lang:
                prob_str = f" (confidence {lang_prob:.2f})" if lang_prob is not None else ""
                print(f"🌐 Detected language: {detected_lang}{prob_str}")

            user_text = "".join([seg.text for seg in segments]).strip()
            print(f"📝 You said: '{user_text}'")

            if not user_text:
                print("⚠️ No text detected. Speak CLEARER and LOUDER!\n")
                continue

            txt = user_text.lower()

            # Memory Action Handler - runs BEFORE the LLM, so name/fact
            # save+recall is instant and 100% reliable (not dependent on
            # the LLM remembering things correctly).
            # NOTE: the save/recall regex patterns are English-only
            # ("my X is Y" / "what's my X"), so this fast path only
            # triggers for English phrasing regardless of detected_lang.
            # Non-English memory commands will fall through to the LLM
            # below instead of being saved/recalled directly.
            memory_reply = handle_memory(txt)

            if memory_reply:
                print(f"⚡ MEMORY ACTION: {memory_reply}")
                speak(memory_reply, language=detected_lang)
            else:
                print("🧠 Thinking...")
                chat_history.append({"role": "user", "content": user_text})
                if len(chat_history) > 11:
                    chat_history = [chat_history[0]] + chat_history[-10:]

                try:
                    response = ollama.chat(model="qwen2.5:1.5b", messages=chat_history)
                    reply = response["message"]["content"]
                    print(f"🤖 Jarvis: {reply}")

                    chat_history.append({"role": "assistant", "content": reply})
                    speak(reply, language=detected_lang)
                except Exception as e:
                    print(f"❌ Ollama Error: {e}")

            print("\n" + "=" * 45)
        else:
            print("⚠️ Too quiet. Speak LOUDER!\n")

    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(1)