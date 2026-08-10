import sounddevice as sd
import numpy as np
from openwakeword.model import Model
import os

# Print a simple prompt so the user knows the wake word listener is active.
print("🎯 WAKE WORD TEST: Say 'Hey Jarvis'")

# Find the installed openwakeword package location and construct the path
# to the pre-trained wake word model file on disk.
base_path = os.path.dirname(__import__('openwakeword').__file__)
model_path = os.path.join(base_path, "resources", "models", "hey_jarvis_v0.1.onnx")
# Create the wake word detection model using the model file path.
model = Model(wakeword_models=[model_path])


def callback(indata, frames, time, status):
    # Convert the incoming audio data from the first channel to float32.
    audio = indata[:, 0].astype(np.float32)
    # Measure the loudness of the audio block.
    level = np.max(np.abs(audio))
    # Only run wake word prediction when the audio is above a threshold.
    if level > 0.05:
        preds = model.predict(audio)
        # Iterate over predicted wake word names and their scores.
        for name, score in preds.items():
            # Print any predictions with a very small positive score.
            if float(score) > 0.01:
                print(f"{name}: {float(score):.4f}")
            # If the prediction score is strong enough, signal detection.
            if float(score) > 0.3:
                print("\n✅✅✅ DETECTED! Score: " + f"{float(score):.4f}" + " ✅✅✅\n")


# Open a stream from the microphone with the configured rate and block size.
# The callback will be called for each block of audio data.
with sd.InputStream(samplerate=16000, channels=1, dtype='float32', blocksize=1280, 
                    callback=callback):
    # Keep the program alive so audio processing continues.
    while True:
        sd.sleep(1000)
