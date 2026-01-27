import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from main import app
from fastapi.testclient import TestClient
import wave
import struct

client = TestClient(app)

def create_wav():
    with wave.open("simple.wav", 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(b'\x00' * 44100)

create_wav()
print("Starting request...")
with open("simple.wav", "rb") as f:
    try:
        res = client.post("/analyze", data={"mode": "EARLY_ACCESS"}, files={"file": ("simple.wav", f, "audio/wav")})
        print(f"Status: {res.status_code}")
        print(f"Body: {res.text}")
    except Exception as e:
        print(f"Exception: {e}")
print("Done.")
if os.path.exists("simple.wav"):
    os.remove("simple.wav")
