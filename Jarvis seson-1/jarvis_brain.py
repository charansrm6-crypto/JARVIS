import os

# Set this environment variable before importing huggingface_hub / faster_whisper.
# On Windows, the HF cache may try to create symlinks, which needs admin rights
# or Developer Mode. Disabling symlinks forces file copying instead.
# This avoids the Windows error WinError 1314.
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import sounddevice as sd
import numpy as np
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
import pyttsx3
import ollama
import time

print("🤖 JARVIS - (OLLAMA BRAIN, ANY-LANGUAGE)")
print("=" * 40)

# 0. Make sure the pretrained wake-word weights exist locally.
#    openwakeword ships without the .onnx files - they must be
#    downloaded once into its package "resources/models" folder.
#    This is a no-op (instant) on every run after the first.
print("📥 Checking wake word model files...")
openwakeword.utils.download_models(["hey_jarvis"])

# 1. Load Wake Word Model
#    Use the built-in model name (not a bare filename) so openwakeword
#    resolves it against the models it just downloaded.
owwModel = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

# 2. Load Whisper Model
#    This model is used later to convert recorded speech into text.
whisper_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

import json

# 3. Initialize Voice Engine
#    pyttsx3 will speak replies through the system audio device.
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ---------------------------------------------------------------------------
# CHANGED: the original _LANG_VOICE_HINTS dict only covered en/hi/te by
# name-matching against a few hardcoded hint words. That meant any other
# language Whisper detected would silently fall back to the default voice
# even if a matching voice was installed. This builds the map dynamically
# from whatever voices pyttsx3 actually reports (SAPI5 on Windows,
# NSSpeechSynthesizer on macOS, espeak on Linux), indexed by language code,
# so any installed language is picked up automatically -- not just the
# three that were hardcoded before.
# ---------------------------------------------------------------------------
_available_voices = engine.getProperty('voices')


def build_voice_map():
    voice_map = {}
    for voice in _available_voices:
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
    return voice_map


_VOICE_MAP = build_voice_map()
print(f"🗣️ Detected {len(_VOICE_MAP)} language/voice mappings for TTS.")


def pick_voice_for_language(lang_code):
    """Return a voice id matching lang_code if one is installed, else None."""
    return _VOICE_MAP.get(str(lang_code).lower())

samplerate = 16000
chunk_size = 1280
wake_detected = False

# Show which mic Windows/sounddevice is actually going to use.
# If this isn't the mic you're speaking into, wake word will never trigger.
default_input_index = sd.default.device[0]
print(f"🎙️ Default input device: {sd.query_devices(default_input_index)['name']}")

print("✅ All models loaded successfully!")
print("🎤 Say 'Hey Jarvis' to activate...\n")

_debug_counter = 0


def wake_callback(indata, frames, time, status):
    global wake_detected, _debug_counter

    if status:
        print(f"⚠️ Audio status: {status}")

    # openWakeWord expects raw int16 PCM samples, NOT normalized float32.
    # Feeding it float32 in [-1, 1] makes every score come out ~0.000
    # regardless of what's said, since the amplitudes are far too small
    # for its internal feature extraction.
    audio_int16 = indata[:, 0]
    level = np.max(np.abs(audio_int16.astype(np.float32))) / 32768.0

    _debug_counter += 1
    if _debug_counter % 10 == 0:
        print(f"\r🔊 mic level: {level:.4f}   ", end="", flush=True)

    preds = owwModel.predict(audio_int16)

    best_name, best_score = max(preds.items(), key=lambda kv: kv[1])
    if _debug_counter % 10 == 0:
        print(f"🧠 '{best_name}' score: {float(best_score):.3f}", end="", flush=True)

    for name, score in preds.items():
        if float(score) > 0.3:
            print(
                f"\n✅✅✅ WAKE WORD DETECTED! "
                f"(Score: {float(score):.2f}) ✅✅✅\n"
            )

            wake_detected = True
            owwModel.reset()


# Core Processing Loop
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

        print("🎤 I am listening... Speak your command:")

        command_audio = sd.rec(
            int(5 * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        print("⏹️ Recording complete\n")

        audio_float = command_audio.flatten().astype(np.float32) / 32768.0

        # language=None => auto-detect whatever language is spoken
        segments, info = whisper_model.transcribe(
            audio_float,
            beam_size=1,
            language=None
        )
        user_text = "".join(
            [segment.text for segment in segments]
        ).strip()

        detected_lang = info.language  # e.g. "en", "hi", "te", "es" ...
        print(f"🌐 Detected language: {detected_lang}")
        print(f"📝 You said: '{user_text}'")

        if user_text:
            print("🤔 Thinking...")

            # CHANGED: the original prompt told the model the user "may
            # speak ... in Telugu, Hindi, or English", which nudges it
            # toward only those three even though Whisper can detect many
            # more. This version names no specific languages and just
            # points at whatever detected_lang actually came back, so any
            # language Whisper supports is handled the same way.
            response = ollama.chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are Jarvis. Answer briefly and clearly. "
                "The user may speak to you in any language "
                f"(this message was detected as language code: {detected_lang}). "
                "Respond with ONLY a JSON object, no other text, no markdown "
                "fences, in this exact shape:\n"
                '{"speak": "<answer written in the SAME language the user used, '
                'i.e. language code ' + detected_lang + '>", '
                '"display": "<the same answer translated into English>"}'
            )
        },
        {"role": "user", "content": user_text}
    ]
)

            raw_reply = response["message"]["content"].strip()

            try:
                cleaned = raw_reply.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
                speak_text = parsed.get("speak", "").strip()
                display_text = parsed.get("display", "").strip()
            except (json.JSONDecodeError, AttributeError):
                # Model didn't return valid JSON - fall back to showing
                # and speaking the raw text as-is.
                speak_text = raw_reply
                display_text = raw_reply

            print(f"🤖 Jarvis: {display_text}")

            voice_id = pick_voice_for_language(detected_lang)
            if voice_id:
                engine.setProperty('voice', voice_id)
            else:
                print(
                    f"⚠️ No installed voice found for '{detected_lang}' - "
                    "speaking with the default voice (may sound off)."
                )

            engine.say(speak_text)
            engine.runAndWait()

        else:
            print("⚠️ No speech detected.")

        print("\n" + "=" * 40)

    except KeyboardInterrupt:
        print("\n👋 Shutting down Jarvis...")
        break

    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(1)