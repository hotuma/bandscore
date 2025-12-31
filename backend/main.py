from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import librosa
import numpy as np
import tempfile
import os
import shutil
import math
import yt_dlp
from yt_dlp.utils import DownloadError
import time
import uuid
from typing import List, Dict, Optional
from scipy.signal import butter, sosfilt
from collections import Counter

app = FastAPI()

# Create temp directory for served files
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_temp_dir(max_age_sec: int = 60 * 60 * 6):  # 6 hours
    now = time.time()
    for name in os.listdir(TEMP_DIR):
        path = os.path.join(TEMP_DIR, name)
        try:
            if os.path.isfile(path):
                if now - os.path.getmtime(path) > max_age_sec:
                    os.remove(path)
        except Exception:
            pass

# Mount static files
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins list for exact matches
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://bandscore.vercel.app",
    ],
    # Regex for Vercel preview URLs (bandscore-*.vercel.app)
    allow_origin_regex=r"https://bandscore-.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers & Data ---

ChordTab = dict[str, list[str]]

CHORD_TO_TAB: ChordTab = {
    # 6th string -> 1st string
    "C":  ["x", "3", "2", "0", "1", "0"],
    "C#": ["x", "4", "6", "6", "6", "4"], # Barre
    "D":  ["x", "x", "0", "2", "3", "2"],
    "D#": ["x", "6", "8", "8", "8", "6"], # Barre
    "E":  ["0", "2", "2", "1", "0", "0"],
    "F":  ["1", "3", "3", "2", "1", "1"],  # Barre F
    "F#": ["2", "4", "4", "3", "2", "2"],  # Barre F#
    "G":  ["3", "2", "0", "0", "0", "3"],
    "G#": ["4", "6", "6", "5", "4", "4"],  # Barre G#
    "A":  ["x", "0", "2", "2", "2", "0"],
    "A#": ["x", "1", "3", "3", "3", "1"],  # Barre A#
    "B":  ["x", "2", "4", "4", "4", "2"],  # Barre B

    # Minors
    "Cm":  ["x", "3", "5", "5", "4", "3"],
    "C#m": ["x", "4", "6", "6", "5", "4"],
    "Dm":  ["x", "x", "0", "2", "3", "1"],
    "D#m": ["x", "6", "8", "8", "7", "6"],
    "Em":  ["0", "2", "2", "0", "0", "0"],
    "Fm":  ["1", "3", "3", "1", "1", "1"],
    "F#m": ["2", "4", "4", "2", "2", "2"],
    "Gm":  ["3", "5", "5", "3", "3", "3"],
    "G#m": ["4", "6", "6", "4", "4", "4"],
    "Am":  ["x", "0", "2", "2", "1", "0"],
    "A#m": ["x", "1", "3", "3", "2", "1"],
    "Bm":  ["x", "2", "4", "4", "3", "2"],
}

def chord_to_tab(chord: str) -> Optional[list[str]]:
    key = chord.strip()
    return CHORD_TO_TAB.get(key)

# --- Signal Processing ---

def highpass_filter(y: np.ndarray, sr: int, cutoff_hz: float = 60.0) -> np.ndarray:
    """Sub-bass low-cut filter"""
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, y)

def compute_chroma_log(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """
    Compute CQT-based chroma features with log compression and normalization.
    Returns: (12, T)
    """
    # CQT chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # Log compression: log(1 + k * chroma)
    k = 10.0
    chroma_log = np.log1p(k * chroma)

    # L1 Normalization per frame
    chroma_norm = chroma_log / (np.sum(chroma_log, axis=0, keepdims=True) + 1e-8)
    return chroma_norm

def compute_bass_chroma(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """
    Compute Bass Chroma (E1-E4 focus).
    Returns: (12, T)
    """
    fmin = librosa.note_to_hz("E1")
    bass_chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_octaves=3,
    )

    k = 10.0
    bass_chroma_log = np.log1p(k * bass_chroma)
    bass_chroma_norm = bass_chroma_log / (np.sum(bass_chroma_log, axis=0, keepdims=True) + 1e-8)

    return bass_chroma_norm

# --- Chord Templates ---

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def rotate_template(base: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(base, shift)

def build_chord_templates() -> tuple[dict[str, np.ndarray], list[str], np.ndarray]:
    """
    Build weighted chord templates.
    Returns:
        templates: dict {name: vector}
        labels: list of chord names
        matrix: np.ndarray (NumChords, 12)
    """
    templates = {}

    # Weighted templates: Root > 5th > 3rd
    # Intervals: 0(root), 4(maj3), 7(5th)
    base_major = np.zeros(12)
    base_major[0] = 1.0   # root
    base_major[7] = 0.7   # 5th
    base_major[4] = 0.5   # 3rd

    # Intervals: 0(root), 3(min3), 7(5th)
    base_minor = np.zeros(12)
    base_minor[0] = 1.0
    base_minor[7] = 0.7
    base_minor[3] = 0.5

    for i, name in enumerate(NOTE_NAMES):
        templates[f"{name}"] = rotate_template(base_major, i)
        templates[f"{name}m"] = rotate_template(base_minor, i)

    labels = list(templates.keys())
    # Ensure consistent order
    labels.sort() 
    
    matrix = np.stack([templates[label] for label in labels], axis=0)
    # Normalize templates
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    
    return templates, labels, matrix

CHORD_TEMPLATES, CHORD_LABELS, TEMPLATE_MATRIX = build_chord_templates()

# --- Key Estimation ---

# Krumhansl-Schmuckler profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def estimate_key_from_chroma(chroma: np.ndarray) -> tuple[str, str]:
    """
    Estimate key (root_name, 'maj' or 'min') from global chroma sum.
    """
    chroma_sum = np.sum(chroma, axis=1)  # (12,)
    
    best_score = -np.inf
    best_root = 0
    best_mode = "maj"

    for i in range(12):
        maj_profile = np.roll(MAJOR_PROFILE, i)
        min_profile = np.roll(MINOR_PROFILE, i)

        maj_score = np.dot(chroma_sum, maj_profile)
        min_score = np.dot(chroma_sum, min_profile)

        if maj_score > best_score:
            best_score = maj_score
            best_root = i
            best_mode = "maj"
        if min_score > best_score:
            best_score = min_score
            best_root = i
            best_mode = "min"

    return NOTE_NAMES[best_root], best_mode

def get_diatonic_chords_for_key(root_name: str, mode: str) -> list[str]:
    """
    Get list of diatonic chords for a given key.
    """
    root_index = NOTE_NAMES.index(root_name)
    major_scale_degrees = [0, 2, 4, 5, 7, 9, 11]

    if mode == "maj":
        degrees = major_scale_degrees
        # I, ii, iii, IV, V, vi, vii°(dim -> m for simplicity)
        qualities = ["", "m", "m", "", "", "m", "m"]
    else:
        # Natural Minor (Aeolian)
        # Relative major's I is at +3 semitones from minor root (e.g. Am -> C)
        # But here we construct from root. 
        # Minor scale: 0, 2, 3, 5, 7, 8, 10
        degrees = [0, 2, 3, 5, 7, 8, 10]
        # i, ii°, III, iv, v, VI, VII
        qualities = ["m", "m", "", "m", "m", "", ""]

    chords = []
    for deg, q in zip(degrees, qualities):
        note = NOTE_NAMES[(root_index + deg) % 12]
        chords.append(note + q)
    return chords

# --- Chord Detection ---

def chord_root_index(label: str) -> int:
    """
    Get root note index (0-11) from chord label.
    """
    candidates = sorted(NOTE_NAMES, key=len, reverse=True)
    for n in candidates:
        if label.startswith(n):
            return NOTE_NAMES.index(n)
    return 0

def cosine_similarity_matrix(templates: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """
    templates: (C, 12)
    chroma: (12, T)
    Returns: (C, T)
    """
    temp_norm = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-8)
    chrom_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
    return temp_norm @ chrom_norm

def detect_chords_matrix(
    main_matrix: np.ndarray,   # (S, 12)
    bass_matrix: np.ndarray,   # (S, 12)
    penalty_mask: Optional[np.ndarray] = None,
    penalty_value: float = 0.15,
    main_weight: float = 0.7,
    bass_weight: float = 0.3,
) -> list[str]:
    """
    Detect chords using weighted combination of main and bass chroma.
    """
    num_segments = main_matrix.shape[0]
    if num_segments == 0:
        return []

    # shape 安全確認：もし列数が違えば小さい方に合わせる
    if bass_matrix.shape[0] != num_segments:
        min_segs = min(num_segments, bass_matrix.shape[0])
        main_matrix = main_matrix[:min_segs, :]
        bass_matrix = bass_matrix[:min_segs, :]
        num_segments = min_segs

    # Main scores
    main_scores = cosine_similarity_matrix(TEMPLATE_MATRIX, main_matrix.T)  # (C, S)

    num_chords = TEMPLATE_MATRIX.shape[0]
    bass_scores = np.zeros((num_chords, num_segments))

    for chord_idx, label in enumerate(CHORD_LABELS):
        root_idx = chord_root_index(label)
        bass_scores[chord_idx, :] = bass_matrix[:, root_idx]

    if np.max(bass_scores) > 0:
        bass_scores = bass_scores / (np.max(bass_scores) + 1e-8)

    final_scores = main_scores * main_weight + bass_scores * bass_weight

    if penalty_mask is not None and penalty_mask.shape[0] == num_chords:
        final_scores[penalty_mask, :] -= penalty_value

    best_indices = np.argmax(final_scores, axis=0)
    return [CHORD_LABELS[i] for i in best_indices]

def aggregate_chroma_per_segment(
    chroma: np.ndarray,
    times: np.ndarray,
    beat_times: np.ndarray,
    beats_per_segment: int = 2,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Aggregate chroma per segment (e.g. 2 beats).
    Returns:
      segment_chroma: (num_segments, 12)
      segments: [(start_sec, end_sec), ...]
    """
    num_frames = chroma.shape[1]

    # ビートが全く検出できない場合 → 全体を1セグメントとして扱う
    if beat_times is None or len(beat_times) == 0:
        if num_frames == 0:
            return np.zeros((0, 12)), []
        start_t = float(times[0])
        end_t = float(times[-1])
        seg_c = np.mean(chroma, axis=1)  # (12,)
        return seg_c[None, :], [(start_t, end_t)]

    num_beats = len(beat_times)
    num_segments = int(math.ceil(num_beats / beats_per_segment))

    segment_chroma_list: list[np.ndarray] = []
    segments: list[tuple[float, float]] = []

    frame_indices = np.arange(len(times))

    for seg_idx in range(num_segments):
        beat_start_idx = seg_idx * beats_per_segment
        beat_end_idx = min((seg_idx + 1) * beats_per_segment, num_beats)

        if beat_start_idx >= num_beats:
            break

        start_t = beat_times[beat_start_idx]

        if beat_end_idx < num_beats:
            end_t = beat_times[beat_end_idx]
        else:
            # Last segment - extrapolate end time if needed
            if num_beats > 0:
                last_beat_start = beat_times[num_beats - 1]
                if num_beats > 1:
                    avg_beat_dur = (beat_times[-1] - beat_times[0]) / (num_beats - 1)
                    end_t = last_beat_start + avg_beat_dur
                else:
                    end_t = last_beat_start + 0.5
            else:
                 # Should be covered by early exit, but safe fallback
                 end_t = start_t + 1.0

        # times の範囲にクリップ
        if len(times) > 0:
            start_t = max(start_t, float(times[0]))
            end_t = min(end_t, float(times[-1]))

        mask = (times >= start_t) & (times < end_t)
        idx = frame_indices[mask]

        if len(idx) == 0:
            # フレームが1つも含まれない場合はゼロベクトル
            segment_chroma_list.append(np.zeros(12))
        else:
            seg_c = np.mean(chroma[:, idx], axis=1)
            segment_chroma_list.append(seg_c)

        segments.append((float(start_t), float(end_t)))

    if not segment_chroma_list:
        return np.zeros((0, 12)), []

    return np.stack(segment_chroma_list, axis=0), segments

def smooth_chord_sequence(chords: list[str]) -> list[str]:
    if len(chords) < 3:
        return chords[:]

    smoothed = chords[:]
    for i in range(1, len(chords) - 1):
        prev_c = smoothed[i - 1]
        curr_c = smoothed[i]
        next_c = smoothed[i + 1]
        if prev_c == next_c and curr_c != prev_c:
            smoothed[i] = prev_c
    return smoothed

# --- Endpoints ---

class YouTubeRequest(BaseModel):
    url: str

def download_youtube_audio(url: str, cookie_path: str | None = None) -> str:
    """
    Download audio from YouTube URL using yt-dlp.
    Returns the path to the downloaded file.
    """
    print(f"[DEBUG] yt-dlp start: {url}")
    # Determine FFmpeg location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_bin_dir = os.path.join(base_dir, "bin")
    ffmpeg_exe = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
    
    # Check if local ffmpeg exists, otherwise rely on system PATH
    ffmpeg_location = ffmpeg_bin_dir if os.path.exists(ffmpeg_exe) else None

    request_id = uuid.uuid4().hex
    outtmpl = os.path.join(TEMP_DIR, f"{request_id}-%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "noplaylist": True,
        "socket_timeout": 20,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 1,
        "geo_bypass": True,
        "nopart": True,
        "overwrites": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": outtmpl,
        "quiet": False,
        "no_warnings": False,
        "ffmpeg_location": ffmpeg_location,
    }

    if cookie_path:
        ydl_opts["cookiefile"] = cookie_path
        print("[DEBUG] yt-dlp using cookies")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print("[DEBUG] yt-dlp extract_info... (download=True)")
            info = ydl.extract_info(url, download=True)
            print("[DEBUG] yt-dlp extract_info done")
            filename = ydl.prepare_filename(info)
            # yt-dlp might change extension after conversion (e.g. .webm -> .mp3)
            base = os.path.splitext(filename)[0]
            final_path = base + ".mp3"
            
            if os.path.exists(final_path):
                 print(f"[DEBUG] yt-dlp done: {final_path} ({os.path.getsize(final_path)} bytes)")
                 return final_path

            # 変換されず元拡張子のまま残るケースも拾う
            if os.path.exists(filename):
                print(f"[DEBUG] yt-dlp done (no mp3 conversion): {filename} ({os.path.getsize(filename)} bytes)")
                return filename

            raise RuntimeError("yt-dlp download succeeded but output file not found")

        except DownloadError as e:
            msg = str(e)
            print(f"[ERROR] yt-dlp download error: {msg}")
            
            # 429 Too Many Requests
            if "Too many requests" in msg or "HTTP Error 429" in msg:
                raise HTTPException(
                    status_code=429,
                    detail="YouTube Rate Limit Exceeded. Please try again later."
                )

            # 403 Forbidden (Login/Bot/Privacy)
            if "Sign in to confirm you’re not a bot" in msg or "confirm you’re not a bot" in msg or "cookies" in msg or "This video is only available to Music Premium members" in msg or "Private video" in msg:
                raise HTTPException(
                    status_code=403,
                    detail="YouTube Access Denied (Login/Cookies required). Please try a different video or upload cookies.txt."
                )
            raise e

def analyze_audio_file(file_path: str) -> dict:
    """Core analysis logic reusable for both uploads and URLs."""
    print(f"[DEBUG] Starting analysis for {file_path}")
    try:
        # 1. Load & Preprocess
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        print(f"[DEBUG] Audio loaded. Size: {y.size}, SR: {sr}")
        if y.size == 0:
            raise ValueError("Audio file is empty or unreadable")

        y = highpass_filter(y, sr)
        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        print(f"[DEBUG] Audio duration: {duration_sec}s")

        # 2. Beat tracking
        print("[DEBUG] Beat tracking...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # Use float for precision to prevent drift
        val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
        bpm = round(val, 2)
        print(f"[DEBUG] BPM: {bpm}, Beats detected: {len(beat_frames)}")

        # 3. Chroma
        print("[DEBUG] Computing chroma...")
        hop_length = 2048
        chroma = compute_chroma_log(y, sr, hop_length=hop_length)
        bass_chroma = compute_bass_chroma(y, sr, hop_length=hop_length)
        print(f"[DEBUG] Chroma shape: {chroma.shape}")

        if chroma.shape[1] == 0:
            raise ValueError("Chroma extraction failed or audio too short")

        # 4. Time axes
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

        # 5. Aggregate per segment
        print("[DEBUG] Aggregating segments...")
        main_matrix, segments = aggregate_chroma_per_segment(chroma, times, beat_times, beats_per_segment=2)
        bass_matrix, _ = aggregate_chroma_per_segment(bass_chroma, times, beat_times, beats_per_segment=2)
        print(f"[DEBUG] Segments: {len(segments)}")

        # 6. Key estimation
        key_root, key_mode = estimate_key_from_chroma(chroma)
        estimated_key = f"{key_root} {key_mode}"
        print(f"[DEBUG] Key: {estimated_key}")

        # 7. Diatonic penalty
        diatonic_chords = set(get_diatonic_chords_for_key(key_root, key_mode))
        penalty_mask = np.array(
            [label not in diatonic_chords for label in CHORD_LABELS],
            dtype=bool
        )

        # 8. Detection
        print("[DEBUG] Detecting chords...")
        raw_chords = detect_chords_matrix(
            main_matrix,
            bass_matrix,
            penalty_mask=penalty_mask,
            penalty_value=0.15,
            main_weight=0.7,
            bass_weight=0.3
        )
        print(f"[DEBUG] Raw chords detected: {len(raw_chords)}")

        smoothed_chords = smooth_chord_sequence(raw_chords)

        bars = []
        for i, chord_name in enumerate(smoothed_chords):
            tab = chord_to_tab(chord_name)
            bars.append({
                "bar": i + 1,
                "chord": chord_name,
                "tab": {
                    "frets": tab
                } if tab else None
            })
        
        print(f"[DEBUG] Analysis complete. Returning {len(bars)} bars.")
        return {
            "bpm": bpm,
            "duration_sec": round(duration_sec, 1),
            "time_signature": "2/4",
            "key": estimated_key,
            "bars": bars
        }

    except Exception as e:
        print(f"[ERROR] Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "BandScore API is running"}

@app.post("/ping")
async def ping():
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"git_sha": os.getenv("RENDER_GIT_COMMIT", "unknown")}

# --- DUMMY IMPLEMENTATION FOR CONNECTIVITY CHECK ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    return {
        "chords": [{"name":"C","startSec":0,"endSec":2}],
        "meta": {"bpm": 120, "key": "C", "durationSec": 2}
    }

# --- REAL IMPLEMENTATION (TEMPORARILY DISABLED) ---
# @app.post("/analyze")
async def _analyze_audio_real(file: UploadFile = File(...)):
    # 1. Validate File Size
    # UploadFile.file is a SpooledTemporaryFile. We can check size.
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    
    if size > 20 * 1024 * 1024: # 20MB
        raise HTTPException(
            status_code=400, 
            detail={"error": {"code": "FILE_TOO_LARGE", "message": "File size exceeds 20MB limit."}}
        )

    # 2. Validate Extension / MIME
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".mp3", ".wav", ".m4a"]:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "UNSUPPORTED_FORMAT", "message": "Only mp3, wav, and m4a are supported."}}
        )

    # Save uploaded file
    # Use a safe unique name
    import uuid
    safe_filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(TEMP_DIR, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Analyze
        # We need to adapt analyze_audio_file to return exact start/end logic or map it here.
        # Since analyze_audio_file returns 'bars' (equal duration), we can map them back to time.
        # However, analyze_audio_file internal logic actually has segments. 
        # For now, we will wrap analyze_audio_file and transform the result.
        
        raw_result = analyze_audio_file(file_path)
        
        # Transform 'bars' to 'chords' list for the new contract
        # The current 'bars' output assumes fixed structural bars.
        # But 'analyze_audio_file' calculates 2 beats per segment.
        # Let's reconstruct precise timing from the result's bpm/time_sig.
        
        bpm = raw_result["bpm"] # float
        # Each bar in raw_result corresponds to 2 beats (since beats_per_segment=2 in simple logic)?
        # Actually in main.py: aggregate_chroma_per_segment(beats_per_segment=2)
        # So each "bar" in the current simplified logic is 2 beats.
        # Default time signature returned is "2/4".
        
        seconds_per_beat = 60.0 / bpm if bpm > 0 else 0.5
        segment_duration = seconds_per_beat * 2
        
        chords_list = []
        for i, bar in enumerate(raw_result["bars"]):
            # bar has: { "bar": i+1, "chord": "Am", "tab": ... }
            start_sec = i * segment_duration
            end_sec = (i + 1) * segment_duration
            
            # Clip to actual duration
            if start_sec > raw_result["duration_sec"]:
                break
            if end_sec > raw_result["duration_sec"]:
                end_sec = raw_result["duration_sec"]
                
            chords_list.append({
                "startSec": round(start_sec, 3),
                "endSec": round(end_sec, 3),
                "name": bar["chord"],
                "confidence": 1.0 # Mock confidence for now as logic behaves deterministically
            })

        return {
            "chords": chords_list,
            "meta": {
                "durationSec": raw_result["duration_sec"],
                "bpm": bpm,
                "key": raw_result["key"]
            }
        }

    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as e:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "ANALYSIS_FAILED", "message": str(e)}}
        )

@app.post("/analyze/url")
def analyze_url(
    url: str = Form(...),
    cookies: UploadFile | None = File(None),
):
    cleanup_temp_dir()
    
    cookie_path = None
    try:
        # Validate and save cookies if provided
        if cookies:
            # 1. Validation
            if not cookies.filename.endswith(".txt"):
                raise HTTPException(status_code=400, detail="Cookie file must be a .txt file")
            
            # Check size (approximate using seek if needed, but SpooledTemporaryFile might not need it immediately)
            # Safe approach: Read chunks to temp file and count bytes
            
            # 2. Secure Save
            suffix = os.path.splitext(cookies.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                size = 0
                while True:
                    chunk = cookies.file.read(1024 * 1024) # 1MB chunks
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 1024 * 1024 * 1: # 1MB limit
                        raise HTTPException(status_code=400, detail="Cookie file too large (limit 1MB)")
                    tmp.write(chunk)
                cookie_path = tmp.name
            print(f"[DEBUG] cookies loaded: {cookie_path}")

        file_path = download_youtube_audio(url, cookie_path=cookie_path)
        
        result = analyze_audio_file(file_path)
        
        # Return URL
        filename = os.path.basename(file_path)
        result["audio_url"] = f"/temp/{filename}"
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Secure cleanup
        if cookie_path and os.path.exists(cookie_path):
            try:
                os.remove(cookie_path)
                print("[DEBUG] cookies deleted")
            except Exception as e:
                print(f"[WARN] Failed to delete cookie file: {e}")
