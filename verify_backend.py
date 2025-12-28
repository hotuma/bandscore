import requests
import wave
import struct
import math
import os

# Create a dummy WAV file (1 second of silence/sine)
filename = "test_verify.wav"
sample_rate = 22050
duration = 1.0
frequency = 440.0

print(f"Creating dummy WAV file: {filename}")
with wave.open(filename, 'w') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    
    num_samples = int(sample_rate * duration)
    for i in range(num_samples):
        value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
        data = struct.pack('<h', value)
        wav_file.writeframes(data)

# Send request
url = "http://localhost:8000/analyze"
print(f"Sending request to {url}...")
try:
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'audio/wav')}
        response = requests.post(url, files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON:")
        print(response.json())
    else:
        print("Error Response:")
        print(response.text)

except Exception as e:
    print(f"Request failed: {e}")

finally:
    if os.path.exists(filename):
        os.remove(filename)
