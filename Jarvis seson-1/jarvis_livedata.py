import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyttsx3
import ollama
import subprocess
import time
import wikipedia
from duckduckgo_search import DDGS

# MODEL_NAME is the Ollama model used to generate AI responses.
# It is chosen to be small, fast, and suitable for live demo use.
MODEL_NAME = "qwen2.5:1.5b"   # small, fast, good for live agent demos

print(" JARVIS - (AI AGENT MODE, MULTI-LANGUAGE)")
print("=" * 45)

# ---------------- Load models ----------------
# Load the speech recognition model used to understand spoken input.
print(" Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Initialize the text-to-speech engine so Jarvis can speak replies.
print(" Initializing Voice Engine...")
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# ---------------------------------------------------------------------------
# NEW: language -> TTS voice lookup, built once at startup.
# Inspects whatever voices are installed on this OS (SAPI5 on Windows,
# NSSpeechSynthesizer on macOS, espeak on Linux) and indexes them by
# language code, so we can switch the shared `engine`'s voice to match
# whatever language Whisper detects. If no matching voice is installed,
# we simply leave the engine on its current/default voice.
# ---------------------------------------------------------------------------
def build_voice_map():
    voice_map = {}
    try:
        for voice in engine.getProperty('voices'):
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
    except Exception as e:
        print(f" Could not enumerate TTS voices: {e}")
    return voice_map


VOICE_MAP = build_voice_map()
DEFAULT_VOICE_ID = engine.getProperty('voice')
print(f" Detected {len(VOICE_MAP)} language/voice mappings for TTS.")

# speak() says text aloud. If a language is provided, it tries to use a matching installed voice.

def speak(text, language=None):
    """Speak text using the shared engine, optionally switching to a voice
    that matches the detected language first. Falls back to the current
    default voice if no match is installed on this machine."""
    if not text:
        return
    if language:
        voice_id = VOICE_MAP.get(str(language).lower())
        if voice_id:
            engine.setProperty('voice', voice_id)
        else:
            print(f" No installed TTS voice found for language '{language}', using default voice.")
            engine.setProperty('voice', DEFAULT_VOICE_ID)
    engine.say(text)
    engine.runAndWait()


# chat_history stores recent conversation history for the AI model.
# This helps the model remember the context of the user's previous messages.
chat_history = [
    {"role": "system", "content": "You are Jarvis, an advanced AI agent. Keep answers short and clear. Always reply in the same language the user spoke in."}
]

samplerate = 16000

print(" All loaded!")
print(" Just SPEAK LOUDLY - No wake word needed! (Any language Whisper supports will be auto-detected.)\n")

# Main loop: record sound, transcribe it, decide how to respond, then speak.
while True:
    try:
        print(" Listening... (speak loudly)")

        # Record 4 seconds of audio from the microphone.
        audio = sd.rec(int(4 * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()

        # Check audio volume to decide whether the user spoke loudly enough.
        level = np.max(np.abs(audio))
        print(f" Audio Level: {level:.4f}")

        if level <= 0.05:
            print(" Too quiet. Speak LOUDER!\n")
            continue

        print(" Sound detected! Transcribing...\n")

        # NEW: language=None makes faster-whisper auto-detect the spoken
        # language instead of forcing English. info.language then tells us
        # what it detected (e.g. "en", "es", "hi").
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
            print(f" Detected language: {detected_lang}{prob_str}")

        user_text = "".join([seg.text for seg in segments]).strip()
        print(f" You said: '{user_text}'")

        if not user_text:
            print(" No text detected. Speak CLEARER and LOUDER!\n")
            continue

        # ---------------- AGENT ROUTER ----------------
        # NOTE: all the keyword checks below ("open youtube", "what is",
        # "weather", etc.) are English-only, so this fast path only
        # triggers on English phrasing regardless of detected_lang.
        # Non-English commands fall through to the LLM chat branch instead.
        txt = user_text.lower()
        agent_reply = None

        # Tool 1: Local PC Actions
        if "open youtube" in txt:
            subprocess.Popen("explorer https://www.youtube.com", shell=True)
            agent_reply = "Opening YouTube, sir."
        elif "open google" in txt:
            subprocess.Popen("explorer https://www.google.com", shell=True)
            agent_reply = "Opening Google, sir."
        elif "open chrome" in txt:
            try:
                subprocess.Popen(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
                agent_reply = "Opening Chrome, sir."
            except FileNotFoundError:
                agent_reply = "I could not find Chrome on this PC, sir."
        elif "open notepad" in txt:
            subprocess.Popen("notepad.exe")
            agent_reply = "Opening Notepad, sir."
        elif "open calculator" in txt:
            subprocess.Popen("calc.exe")
            agent_reply = "Opening Calculator, sir."

        # Tool 2: Wikipedia (definitions / who is / what is)
        elif "what is" in txt or "who is" in txt or "define" in txt:
            search_term = (
                txt.replace("what is", "")
                   .replace("who is", "")
                   .replace("define", "")
                   .strip()
            )
            try:
                summary = wikipedia.summary(search_term, sentences=2, auto_suggest=True)
                agent_reply = f"Here is what I found: {summary}"
            except wikipedia.DisambiguationError as e:
                agent_reply = f"That's a bit ambiguous. Did you mean {e.options[0]}?"
            except wikipedia.PageError:
                agent_reply = "I couldn't find anything on Wikipedia for that, sir."
            except Exception as e:
                print(f" Wikipedia error: {e}")
                agent_reply = "I ran into an issue searching Wikipedia, sir."

        # Tool 3: Live Web Search (weather / search / latest / news)
        elif "weather" in txt or "search" in txt or "latest" in txt or "news" in txt:
            search_term = (
                txt.replace("weather", "")
                   .replace("search for", "")
                   .replace("search", "")
                   .replace("latest", "")
                   .replace("news", "")
                   .strip()
            )
            try:
                results = DDGS().text(search_term if search_term else "weather today", max_results=3)
                if results:
                    reply = "Here are the top results: "
                    for i, r in enumerate(results[:3], 1):
                        reply += f"{i}. {r['title']}. "
                    agent_reply = reply
                else:
                    agent_reply = "I couldn't find any results for that, sir."
            except Exception as e:
                print(f" Search error: {e}")
                agent_reply = "The live search failed, sir. Please check your internet connection."

        # ---------------- Respond ----------------
        if agent_reply:
            print(f" AGENT ACTION: {agent_reply}")
            speak(agent_reply, language=detected_lang)
        else:
            # Ollama Brain with Memory
            print(f" Thinking (Agent Mode, model={MODEL_NAME})...")
            chat_history.append({"role": "user", "content": user_text})
            if len(chat_history) > 11:
                chat_history = [chat_history[0]] + chat_history[-10:]

            try:
                response = ollama.chat(model=MODEL_NAME, messages=chat_history)
                reply = response["message"]["content"]
                print(f" Jarvis: {reply}")

                chat_history.append({"role": "assistant", "content": reply})
                speak(reply, language=detected_lang)
            except Exception as e:
                print(f" Ollama Error: {e}")
                print(f" Make sure Ollama is running and you've pulled the model: ollama pull {MODEL_NAME}")

        print("\n" + "=" * 45)

    except KeyboardInterrupt:
        print("\n Shutting down...")
        break
    except Exception as e:
        print(f" Error: {e}")
        time.sleep(1)