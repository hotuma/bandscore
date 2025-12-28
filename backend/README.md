# BandScore Backend

FastAPI backend for BandScore application.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   **Note on Audio Support:**
   - `librosa` uses `soundfile` or `audioread` to load audio.
   - For WAV files, `soundfile` (included in requirements) is sufficient.
   - For MP3 files, you may need `ffmpeg` installed on your system and added to PATH.

3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## Endpoints

### POST /analyze
Analyzes an uploaded audio file and returns BPM, duration, and chords.

**Request:**
- `file`: Audio file (multipart/form-data)

**Response:**
```json
{
  "bpm": 120,
  "duration_sec": 180.5,
  "time_signature": "4/4",
  "bars": [
    { "bar": 1, "chord": "C", "tab": {...} },
    ...
  ]
}
```
