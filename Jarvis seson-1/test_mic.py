import sounddevice as sd
import numpy as np

# Print a message to let the user know the microphone test is active.
print("🎤 MIC TEST: Speak now...")

def callback(indata, frames, time, status):
    # Calculate the maximum audio amplitude from the incoming samples.
    # This gives a simple indication of how loud the microphone input is.
    level = np.max(np.abs(indata))
    # Print the audio level only when it exceeds a low threshold.
    # This avoids printing values for silence or very quiet noise.
    if level > 0.01:
        print(f"🔊 Audio Level: {level:.4f} (Perfect if > 0.3)")

# Open the microphone input stream and attach the callback.
# The program then sleeps in a loop so the audio callback continues running.
with sd.InputStream(channels=1, callback=callback):
    while True:
        sd.sleep(1000)