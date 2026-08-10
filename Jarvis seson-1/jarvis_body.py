import streamlit as st
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import pyttsx3
import ollama
import psutil
import time
import subprocess
import threading
import threading

# This file defines a Streamlit dashboard that acts as a voice-controlled Jarvis HUD.
# It uses microphone input, speech recognition, text-to-speech, and an AI assistant backend.

# --- PAGE CONFIGURATION ---
# Set Streamlit page metadata and layout before any UI elements are drawn.
st.set_page_config(
    page_title="JARVIS HUD v2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR IRON MAN HUD LOOK ---
# Apply custom styles so the dashboard looks like an Iron Man-style HUD.
st.markdown("""
<style>
.main { background-color: #050505; color: #00ffff; }
.stButton > button {
    background-color: #00ffff; color: #000000; font-weight: bold;
    border: 2px solid #00ffff; border-radius: 10px; box-shadow: 0 0 15px #00ffff;
}
.stButton > button:hover {
    background-color: #ff0066; color: white; border-color: #ff0066; box-shadow: 0 0 15px #ff0066;
}
div[data-testid="stMetricValue"] {
    color: #00ff00; font-size: 2rem; text-shadow: 0 0 10px #00ff00;
}
h1, h2, h3 { color: #00ffff; text-shadow: 0 0 10px #00ffff; }
.stAlert { background-color: #1a1a2e; color: #00ffff; border: 1px solid #00ffff; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
# session_state stores values across reruns so the chat and voice state are remembered.
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You are Jarvis. Answer briefly and clearly."}]
if 'audio_text' not in st.session_state:
    st.session_state.audio_text = ""
if 'jarvis_reply' not in st.session_state:
    st.session_state.jarvis_reply = ""

# --- SIDEBAR: REAL-TIME SYSTEM STATS ---
# Build the sidebar display with system information for CPU, RAM, and battery.
with st.sidebar:
    st.title("🤖 JARVIS OS")
    st.markdown("---")
    st.subheader("💻 System Diagnostics")

    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    battery = psutil.sensors_battery()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("CPU Usage", f"{cpu_percent}%")
    with col2:
        st.metric("RAM Usage", f"{ram.percent}%")

    if battery:
        st.metric("🔋 Battery", f"{battery.percent}%", delta="Plugged" if battery.power_plugged else "On Battery")
    else:
        st.info("No battery detected (Desktop)")

    st.markdown("---")
    st.caption("🟢 System Status: ONLINE")
    st.caption("🌐 Network: Connected")

# --- MAIN HUD DISPLAY ---
# Layout the main page with two columns: status controls and chat history.
st.title("🎯 JARVIS COMMAND CENTER")
st.markdown("---")

col_status, col_chat = st.columns([1, 1])

# --- COLUMN 1: STATUS & CONTROLS ---
with col_status:
    st.subheader("🎙️ Voice Interface")
    if st.session_state.audio_text:
        st.info(f"🗣️ **You Said:** {st.session_state.audio_text}")
    if st.session_state.jarvis_reply:
        st.success(f"🤖 **Jarvis Reply:** {st.session_state.jarvis_reply}")

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        activate = st.button("🎙️ ACTIVATE", use_container_width=True, type="primary")
    with col_btn2:
        if st.button("🔴 CLEAR", use_container_width=True):
            st.session_state.audio_text = ""
            st.session_state.jarvis_reply = ""
            st.rerun()

# --- COLUMN 2: CHAT HISTORY ---
# Show the last few messages stored in the chat history.
with col_chat:
    st.subheader("🧠 Memory Log")
    chat_container = st.empty()

    history_text = ""
    for msg in st.session_state.chat_history[-6:]:
        if msg['role'] == 'user':
            history_text += f"👤 **User:** {msg['content']}\n"
        elif msg['role'] == 'assistant':
            history_text += f"🤖 **Jarvis:** {msg['content']}\n"

    chat_container.text_area("Chat History", value=history_text, height=300, disabled=True)

# --- CACHED RESOURCES ---
# Load the Whisper speech recognition model once and cache it across reruns.
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def speak(text):
    # Runs in a background thread so a stuck TTS engine can never freeze the app
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass

# --- VOICE PROCESSING LOGIC ---
# This section records audio, recognizes speech, executes commands, or sends the text to Ollama.
if activate:
    st.info("🎙️ Recording... Speak now! (4 seconds)")
    try:
        whisper_model = load_whisper()

        audio = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()

        level = np.max(np.abs(audio))
        if level < 0.01:
            st.warning("⚠️ Too quiet! Speak louder.")
            st.rerun()

        segments, _ = whisper_model.transcribe(
            audio.flatten(),
            beam_size=5,
            language="en",
            vad_filter=True
        )
        user_text = "".join([seg.text for seg in segments]).strip()
        st.session_state.audio_text = user_text

        if not user_text:
            st.warning("⚠️ No speech detected.")
            st.rerun()

        txt = user_text.lower()
        reply = ""

        if "open youtube" in txt:
            subprocess.Popen("explorer https://www.youtube.com", shell=True)
            reply = "Opening YouTube, sir."
        elif "open google" in txt:
            subprocess.Popen("explorer https://www.google.com", shell=True)
            reply = "Opening Google, sir."
        elif "open calculator" in txt:
            subprocess.Popen("calc.exe")
            reply = "Opening Calculator, sir."
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_text})
            try:
                response = ollama.chat(model="qwen2.5:1.5b", messages=st.session_state.chat_history)
                reply = response["message"]["content"]
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            except Exception as e:
                reply = "My brain is offline, sir. Please check Ollama."

        st.session_state.jarvis_reply = reply
        threading.Thread(target=speak, args=(reply,), daemon=True).start()

    except Exception as e:
        st.error(f"❌ Error: {e}")
    st.rerun()

# --- FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #555;'>🤖 JARVIS AI SYSTEM v2.0</div>", unsafe_allow_html=True)