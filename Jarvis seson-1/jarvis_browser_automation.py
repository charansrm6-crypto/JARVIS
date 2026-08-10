import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyttsx3
import ollama
import time
import urllib.parse
import subprocess
import keyboard
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# MODEL_NAME is the Ollama model used for conversational replies.
# This model is designed to be small, fast, and support multiple languages.
MODEL_NAME = "qwen2.5:1.5b"   # small, fast, multilingual LLM

# WHISPER_LANGUAGE controls speech transcription language.
# None lets Whisper auto-detect the spoken language.
# Setting it to "en" would force English only.
WHISPER_LANGUAGE = None

print(" JARVIS - (BROWSER CONTROL EDITION)")
print("=" * 50)

# 1. Initialize Voice Engine
# The text-to-speech engine is used for spoken responses.
print(" Starting Voice Engine...")
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# 2. Load Whisper Model
# Whisper converts recorded speech into text.
print(" Loading Whisper Model...")
try:
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    print(" Whisper Loaded!")
except Exception as e:
    print(f" Whisper Error: {e}")
    exit()

# 3. Initialize Browser
# This starts a Chrome browser instance for web automation commands.
print(" Initializing Browser...")
driver = None
try:
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    print(" Browser Ready!\n")
except Exception as e:
    print(f" Browser Error: {e}")
    print(" Jarvis will still run, but browser commands will be unavailable.\n")

# 4. Chat History
# Maintain a short conversation history for the language model.
chat_history = [{"role": "system", "content": "You are Jarvis. Answer briefly and clearly."}]

print(" CONTROLS:")
print(" - Press [SPACEBAR] to activate instantly")
print(" - Or say 'Hey Jarvis'")
print(" - Speak your full question (it will auto-stop)")
print(" - Press [CTRL+C] to exit\n")


# --- DYNAMIC RECORDING (auto-stop on silence) ---
# This function records audio until the speaker becomes silent.
def record_until_silence():
    samplerate = 16000
    threshold = 0.02
    silence_timeout = 1.5
    max_time = 20

    print(" Listening... Speak now!")
    frames = []
    silence_start = None
    is_speaking = False
    start_time = time.time()

    while True:
        chunk = sd.rec(int(0.2 * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        level = np.max(np.abs(chunk))

        if level > threshold:
            is_speaking = True
            silence_start = None
            frames.append(chunk)
        else:
            if is_speaking:
                frames.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_timeout:
                    print(" Silence detected. Processing...")
                    break

        if time.time() - start_time > max_time:
            print(" Max time reached.")
            break

    if frames:
        return np.concatenate(frames, axis=0)
    return np.array([])


# --- BROWSER ACTION (DIRECT URL METHOD) ---
# This function handles direct browser navigation for YouTube and Google.
def browser_action(query):
    if driver is None:
        return "Browser is not available, sir."

    txt = query.lower()

    if "youtube" in txt:
        search_term = (
            txt.replace("open youtube", "")
               .replace("search for", "")
               .replace("search", "")
               .replace("on youtube", "")
               .replace("play", "")
               .strip()
        )
        if not search_term:
            driver.get("https://www.youtube.com")
            return "Opening YouTube, sir."
        encoded_query = urllib.parse.quote(search_term)
        driver.get(f"https://www.youtube.com/results?search_query={encoded_query}")
        return f"Searching YouTube for {search_term}, sir."

    elif "google" in txt:
        search_term = (
            txt.replace("open google", "")
               .replace("search for", "")
               .replace("search", "")
               .replace("on google", "")
               .strip()
        )
        if not search_term:
            driver.get("https://www.google.com")
            return "Opening Google, sir."
        encoded_query = urllib.parse.quote(search_term)
        driver.get(f"https://www.google.com/search?q={encoded_query}")
        return f"Searching Google for {search_term}, sir."

    return None


# --- MAIN LOOP ---
# The main loop waits for activation, records speech, processes commands,
# and then speaks the response.
while True:
    try:
        print(" Waiting for [SPACEBAR] or 'Hey Jarvis'...")
        wake_detected = False

        if keyboard.is_pressed('space'):
            wake_detected = True
            print("\n✅✅✅ SPACEBAR ACTIVATED! ✅✅✅\n")
            time.sleep(0.5)
        else:
            audio_check = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            if np.max(np.abs(audio_check)) > 0.05:
                segs, _ = whisper_model.transcribe(
                    audio_check.flatten(), language=WHISPER_LANGUAGE, beam_size=3
                )
                text = "".join([s.text for s in segs]).strip().lower()
                if "jarvis" in text:
                    wake_detected = True
                    print(f"\n✅✅✅ WAKE WORD DETECTED! ('{text}') ✅✅✅\n")

        if not wake_detected:
            continue

        audio = record_until_silence()
        if len(audio) == 0:
            print(" No speech detected.\n")
            continue

        print(" Transcribing...")
        segments, _ = whisper_model.transcribe(
            audio.flatten(), beam_size=5, language=WHISPER_LANGUAGE, vad_filter=True
        )
        user_text = "".join([seg.text for seg in segments]).strip()
        print(f" You said: '{user_text}'")

        if not user_text:
            continue

        txt = user_text.lower()
        reply = None

        if "youtube" in txt or "google" in txt:
            reply = browser_action(user_text)
        elif "open calculator" in txt:
            subprocess.Popen("calc.exe")
            reply = "Opening Calculator, sir."
        elif "open notepad" in txt:
            subprocess.Popen("notepad.exe")
            reply = "Opening Notepad, sir."
        elif "open chrome" in txt:
            try:
                subprocess.Popen(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
                reply = "Opening Chrome, sir."
            except FileNotFoundError:
                reply = "I could not find Chrome on this PC, sir."

        if reply is None:
            # Fallback to LLM brain (also handles multilingual replies)
            print(f" Thinking (model={MODEL_NAME})...")
            chat_history.append({"role": "user", "content": user_text})
            if len(chat_history) > 11:
                chat_history = [chat_history[0]] + chat_history[-10:]

            try:
                response = ollama.chat(model=MODEL_NAME, messages=chat_history)
                reply = response["message"]["content"]
                chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                print(f" Ollama Error: {e}")
                reply = "My brain is offline, sir."

        print(f" Jarvis: {reply}")
        engine.say(reply)
        engine.runAndWait()
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n Shutting down Jarvis...")
        if driver:
            driver.quit()
        break
    except Exception as e:
        print(f" Unexpected Error: {e}")
        time.sleep(1)