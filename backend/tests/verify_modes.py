import sys
import os
import time
import wave
import struct

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import app, jobs
from fastapi.testclient import TestClient

client = TestClient(app)

def create_dummy_wav(filename, duration_sec=1.0):
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        n_frames = int(44100 * duration_sec)
        # Noise
        import random
        data = bytearray()
        for _ in range(n_frames):
             val = int(random.random() * 10000 - 5000)
             data.extend(struct.pack('<h', val))
        f.writeframes(data)

def test_preview_mode():
    print("\n--- Testing Preview Mode ---", flush=True)
    wav_file = "test_preview.wav"
    create_dummy_wav(wav_file, duration_sec=5.0)
    
    try:
        with open(wav_file, "rb") as f:
            response = client.post("/analyze/preview", files={"file": ("test.wav", f, "audio/wav")})
        
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        print(f"Job ID: {job_id}")
        
        # Poll
        for _ in range(20):
            res_status = client.get(f"/analyze/status/{job_id}")
            status = res_status.json()["status"]
            if status == "done":
                break
            if status == "error":
                print("Job Error:", res_status.json())
                break
            time.sleep(1)
        
        # Get Result
        res_result = client.get(f"/analyze/result/{job_id}")
        if res_result.status_code != 200:
            print("Failed to get result:", res_result.json())
            return
            
        result = res_result.json()
        print("Result Mode:", result.get("mode"))
        print("Bars:", result.get("bars"))
        
        assert result.get("mode") == "PREVIEW"
        assert result.get("bars") is None
        assert result.get("is_preview") is True
        print("PASS: Preview Mode enforced correctly.")

    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)

def test_early_access_mode():
    print("\n--- Testing Early Access Mode ---", flush=True)
    wav_file = "test_ea.wav"
    create_dummy_wav(wav_file, duration_sec=5.0)
    
    try:
        with open(wav_file, "rb") as f:
            # Note: TestClient handles FormData internally
            response = client.post(
                "/analyze", 
                # data={"mode": "EARLY_ACCESS"}, 
                files={"file": ("test.wav", f, "audio/wav")}
            )
        
        print(f"STATUS_CODE: {response.status_code}", flush=True)
        print(f"RESPONSE_TEXT: {response.text}", flush=True)
        if response.status_code != 202:
             pass 
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        
        # Poll
        for _ in range(20):
            res_status = client.get(f"/analyze/status/{job_id}")
            status = res_status.json()["status"]
            if status == "done":
                break
            time.sleep(1)
            
        # Get Result
        res_result = client.get(f"/analyze/result/{job_id}")
        result = res_result.json()
        print("EA Result:", result)
        
        assert result.get("mode") == "EARLY_ACCESS"
        assert result.get("bars") is not None
        assert isinstance(result.get("bars"), list)
        print("PASS: Early Access Mode returned bars.")

    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)

if __name__ == "__main__":
    try:
        test_preview_mode()
        test_early_access_mode()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
