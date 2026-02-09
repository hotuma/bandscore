# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BandScore** (also referred to as "guitar-tab") is a full-stack guitar tab generation application that analyzes audio files to extract musical information (BPM, key, chords) and generates guitar tablature synchronized with audio playback.

### Tech Stack
- **Backend**: Python FastAPI with librosa for audio analysis
- **Frontend**: Next.js 16 (App Router) with TypeScript, React 19, Tailwind CSS 4
- **Audio Processing**: librosa, numpy, scipy for signal processing
- **Client-side Audio**: soundfont-player for guitar chord playback

## Development Commands

### Backend (Python/FastAPI)
```bash
cd backend

# Setup virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/Mac

# Install dependencies
pip install -r requirements.txt

# Run development server (port 8000)
uvicorn main:app --reload

# Run tests (if available)
python tests/verify_modes.py
python tests/verify_preview_content.py
```

### Frontend (Next.js)
```bash
cd frontend

# Install dependencies
npm install

# Development server (port 3000)
npm run dev

# Production build
npm run build

# Start production server
npm start

# Lint
npm run lint
```

## Architecture

### Backend Architecture (backend/main.py)

The backend is a single-file FastAPI application (~1286 lines) containing:

1. **Audio Analysis Pipeline** (`analyze_audio_file` function, line ~746):
   - Loads audio with librosa (limited to MAX_ANALYSIS_SEC for memory safety)
   - Applies highpass filter to remove sub-bass noise
   - Computes STFT-based chroma features (more stable than CQT on cloud platforms)
   - Uses fixed BPM grid (120 BPM) for segment-based analysis
   - Aggregates chroma per segment (0.5s intervals, 2 beats per segment)
   - Estimates key using Krumhansl-Schmuckler profiles
   - Detects chords using weighted template matching with stagnation prevention

2. **Chord Detection System**:
   - Template-based matching with 24 chord templates (12 major + 12 minor)
   - Combines main chroma and bass chroma with configurable weights
   - Diatonic penalty system to favor in-key chords
   - Stagnation prevention algorithm to avoid "sticky" chords (lines ~400-500)
   - Smoothing pass to remove single-bar outliers

3. **Analysis Modes** (AnalyzeMode enum, line ~103):
   - `PREVIEW`: 30-second analysis, no chord details returned
   - `EARLY_ACCESS`: 120-second analysis with full chord/tab data
   - `FULL`: 600-second analysis (currently treated same as EARLY_ACCESS)

4. **API Endpoints**:
   - `POST /analyze`: File upload analysis (requires `mode` form parameter)
   - `POST /analyze/preview`: Preview-only analysis endpoint
   - `POST /analyze/url`: YouTube URL analysis via yt-dlp
   - `POST /analyze/url/preview`: YouTube preview analysis
   - `GET /analyze/status/{job_id}`: Job status polling (async job support)
   - `GET /analyze/result/{job_id}`: Fetch completed job result

5. **Memory Management**:
   - Chunk-based processing controlled by `MAX_ANALYSIS_SEC` env var (default 120s)
   - Uses `CHUNK_SEC` for segmented processing (default 30s)
   - Automatic temp file cleanup (6-hour TTL by default)
   - Memory monitoring via psutil

6. **CORS Configuration** (lines ~54-69):
   - Allows localhost:3000/3001 for local development
   - Production: bandscore.vercel.app
   - Regex support for Vercel preview URLs (bandscore-*.vercel.app)

### Frontend Architecture

1. **Routing Structure** (App Router):
   - `/` - Redirects to `/demo`
   - `/demo` - Demo page with pre-loaded sample songs
   - `/preview` - Preview analysis page (30s limit)
   - `/early-access` - Full early access page (120s analysis)
   - `/lab` - Protected lab environment (requires `lab_access` cookie)
   - `/lab/login` - Lab login page
   - `/waitlist` - Waitlist signup
   - `/legal` - Legal/terms page

2. **Middleware** ([middleware.ts:1](middleware.ts#L1)):
   - Protects `/lab/*` routes with cookie-based authentication
   - Redirects unauthenticated users to `/lab/login`

3. **Key Components**:
   - `FileUpload.tsx` - Audio file upload UI component
   - `ResultDisplay.tsx` - Main playback and tab visualization component with:
     - Audio synchronization using Web Audio API timing
     - Auto-scrolling chord display
     - Real-time chord highlighting during playback
     - Configurable offset adjustment for sync correction
     - Guitar chord playback using soundfont-player

4. **API Client** ([lib/api.ts](frontend/lib/api.ts)):
   - `analyzeAudio(file, mode, timeout, opts)` - Upload and analyze audio file
   - `analyzeYoutube(url, cookiesFile, mode)` - Analyze YouTube URL
   - Automatic timeout handling (default 180s for uploads, 240s for YouTube)
   - Normalizes API responses to absolute URLs

5. **Audio Playback** ([lib/guitarSound.ts](frontend/lib/guitarSound.ts)):
   - Lazy-loads soundfont-player guitar instrument
   - Converts guitar tab fret positions to MIDI notes
   - Synchronizes chord playback with audio using AudioContext timing
   - `playChordFromTabWithSoundFont(tab, when, duration)` - Schedules chord notes

### Data Flow

1. **Upload Analysis**:
   ```
   User uploads audio → Frontend sends to /analyze or /analyze/preview
   → Backend loads with librosa → Chroma extraction → Chord detection
   → Response with bars array (chord, tab, start_sec, end_sec per bar)
   → Frontend displays in ResultDisplay with audio sync
   ```

2. **YouTube Analysis**:
   ```
   User enters URL → Frontend sends to /analyze/url or /analyze/url/preview
   → Backend uses yt-dlp to download → Same analysis pipeline
   → Returns audio_url (served from /temp static mount) + analysis
   → Frontend plays served audio with chord sync
   ```

3. **Audio Sync System**:
   - Backend analysis uses librosa timing (MP3 decode)
   - Frontend plays audio via HTMLAudioElement
   - Timing anchor system reconciles librosa vs browser decode differences
   - Default +0.2s offset for remote URLs (yt-dlp audio)
   - User-adjustable offset slider in UI

## Important Implementation Details

### Backend Considerations

- **Memory Limits**: Always respect `MAX_ANALYSIS_SEC` environment variable. Default is 120s for free tier cloud hosting (e.g., Render). Use chunked processing for longer files.

- **Chord Detection Parameters** (lines ~360-500):
  - `main_weight=1.0, bass_weight=0.3` - Balance between melody and bass
  - `penalty_value=0.3` - Penalty for non-diatonic chords
  - `topk=5` - Number of candidate chords to consider
  - `min_hold_segments=2` - Minimum bars before allowing chord change (flicker prevention)
  - `stagnation_threshold=10` - Force change after N bars of same chord
  - `high_flux_threshold=0.3` - Chroma change threshold for detecting transitions

- **FFmpeg Dependency**: For MP3 support, ffmpeg must be in PATH or in `backend/bin/ffmpeg.exe`

- **Temp File Management**: Files in `backend/temp/` are served via `/temp` static mount and auto-cleaned after 6 hours

### Frontend Considerations

- **Environment Variables**:
  - `NEXT_PUBLIC_API_BASE_URL` - Backend API URL (defaults to http://127.0.0.1:8000)

- **Audio Sync Architecture** ([components/ResultDisplay.tsx:72-80](frontend/components/ResultDisplay.tsx#L72-L80)):
  - Use `anchorRef` to maintain audio/AudioContext time correlation
  - Reset anchor on seek operations
  - Schedule chords ahead using `schedulingBarRef` and `schedulerTimerRef`
  - Never schedule the same bar twice (tracked via `lastScheduledBarRef`)

- **Analysis Modes**: Always pass explicit mode to API:
  - Use `PREVIEW` for `/demo` and `/preview` pages (30s limit)
  - Use `EARLY_ACCESS` for `/early-access` and `/lab` pages (120s)
  - Mode is enforced server-side on `/analyze/preview` endpoints

## Testing

Backend tests are located in `backend/tests/`:
- `verify_modes.py` - Tests analysis mode behavior
- `verify_preview_content.py` - Validates preview response format
- `verify_simple.py` - Basic analysis sanity check

Run tests individually with Python:
```bash
cd backend
python tests/verify_modes.py
```

## Common Patterns

### Adding a New Chord Template
Edit the `CHORD_TO_TAB` dictionary (line ~110) and update `build_chord_templates()` function (line ~252) if adding new chord types beyond major/minor.

### Adjusting Analysis Duration
Set environment variable before running backend:
```bash
export MAX_ANALYSIS_SEC=600  # 10 minutes
uvicorn main:app --reload
```

### Debugging Audio Sync Issues
Enable debug mode in ResultDisplay by clicking the "Debug" toggle. This shows:
- Current audio time
- Current bar index
- Offset value
- Scheduling state

### Adding New API Endpoints
Follow existing pattern:
1. Define endpoint with `@app.post()` or `@app.get()`
2. Use `AnalyzeMode` enum for mode parameter
3. Call `analyze_audio_file()` for core analysis
4. Return response matching `AnalysisResult` interface (frontend/lib/api.ts:15)

## Deployment Notes

- Backend designed for Render/similar cloud platforms with memory constraints
- Frontend deployed on Vercel (configured CORS allows preview URLs)
- Ensure CORS regex in backend matches your deployment domain pattern
- Set appropriate `MAX_ANALYSIS_SEC` based on hosting tier
