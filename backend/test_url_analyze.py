"""Quick test for /analyze/url endpoint"""
import requests
import time

BASE = "http://127.0.0.1:8000"
TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

print("=== Step 1: POST /analyze/url ===")
try:
    resp = requests.post(
        f"{BASE}/analyze/url",
        data={"url": TEST_URL, "mode": "EARLY_ACCESS"},
        timeout=120,
    )
    print(f"Status: {resp.status_code}")
    print(f"Body: {resp.text[:500]}")
    
    if resp.status_code == 202:
        job_id = resp.json().get("job_id")
        print(f"\nJob ID: {job_id}")
        
        print("\n=== Step 2: Polling /analyze/status ===")
        for i in range(60):
            time.sleep(2)
            s = requests.get(f"{BASE}/analyze/status/{job_id}", timeout=10)
            data = s.json()
            print(f"  [{i}] status={data.get('status')}, progress={data.get('progress')}, error={data.get('error')}")
            if data.get("status") in ("done", "error"):
                break
        
        if data.get("status") == "done":
            print("\n=== Step 3: GET /analyze/result ===")
            r = requests.get(f"{BASE}/analyze/result/{job_id}", timeout=10)
            result = r.json()
            print(f"BPM: {result.get('bpm')}")
            print(f"Key: {result.get('key')}")
            print(f"Bars: {len(result.get('bars', []))}")
            print(f"Audio URL: {result.get('audio_url')}")
        elif data.get("status") == "error":
            print(f"\n=== ERROR: {data.get('error')} ===")
    else:
        print(f"\nUnexpected status code: {resp.status_code}")
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
