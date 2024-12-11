import pyaudio
import numpy as np
from transformers import pipeline

print("Loading model...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
print("Model loaded.")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening... (press Ctrl+C to stop)")

try:
    while True:
        frames = []
        for _ in range(int(RATE / CHUNK)):
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(audio_data)

        audio_bytes = b''.join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        result = transcriber(audio_np)

        print("Transcription:", result['text'])

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stream.stop_stream()
    stream.close()
    
    p.terminate()
