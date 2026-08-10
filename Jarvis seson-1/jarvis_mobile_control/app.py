from flask import Flask, render_template, jsonify, request
import subprocess
import pyttsx3
import threading
import ollama
from pyngrok import ngrok

# Create the Flask app object. This is the central application instance
# that defines routes and handles incoming web requests.
app = Flask(__name__)

# Speak function: a fresh engine is created per call and run in a
# background thread, so text-to-speech can never freeze a request.
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass

# The home() route returns the main mobile web page when the root URL is visited.
# This is the first interface a mobile user sees in the browser.
# It does not perform any logic other than rendering the template.
# Input: GET request to '/'
# Output: HTML page from index.html
# Note: render_template looks for templates in the folder configured by Flask.
# In this app, that is the 'templates' directory under jarvis_mobile_control.

# Route for Mobile UI
@app.route('/')
def home():
    return render_template('index.html')

# Route for fixed Commands (buttons)
# This route receives a path parameter named action.
# The action tells the app which predefined command button was pressed.
# Output: JSON with status and reply text.
@app.route('/command/<action>')
def command(action):
    reply = ""

    if action == 'youtube':
        subprocess.Popen("explorer https://www.youtube.com", shell=True)
        reply = "Opening YouTube, sir."

    elif action == 'google':
        subprocess.Popen("explorer https://www.google.com", shell=True)
        reply = "Opening Google, sir."

    elif action == 'calculator':
        subprocess.Popen("calc.exe")
        reply = "Opening Calculator, sir."

    elif action == 'morning':
        reply = "Good morning sir! Weather is 28 degrees. You have 3 meetings today. Opening your dashboard."
        subprocess.Popen("explorer https://www.google.com", shell=True)
        # Run voice in background so UI doesn't freeze
        threading.Thread(target=speak, args=(reply,), daemon=True).start()
        return jsonify({"status": "success", "reply": reply})

    # Speak the reply
    threading.Thread(target=speak, args=(reply,), daemon=True).start()
    return jsonify({"status": "success", "reply": reply})

# Route for free-text / any-language commands, answered by qwen2.5:1.5b
# This endpoint expects a JSON POST with a 'query' field.
# It sends the query to an Ollama chat model and returns the model's reply.
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"status": "error", "reply": "No input received."})

    try:
        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[
                {"role": "system", "content": "You are Jarvis, a helpful AI assistant. Reply briefly and clearly, in the same language the user used."},
                {"role": "user", "content": query}
            ]
        )
        reply = response["message"]["content"]
    except Exception as e:
        reply = "My brain is offline, sir. Please check Ollama."

    threading.Thread(target=speak, args=(reply,), daemon=True).start()
    return jsonify({"status": "success", "reply": reply})

if __name__ == '__main__':
    print("🚀 JARVIS MOBILE SERVER RUNNING...")
    # host='0.0.0.0' is MUST for mobile access
    app.run(host='0.0.0.0', port=5000, debug=False)