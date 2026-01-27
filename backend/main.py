import os

import psutil

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
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
import threading
from scipy.signal import butter, sosfilt
from collections import Counter
from typing import Dict, Any, Optional
from enum import Enum

app = FastAPI()

# Create temp directory for served files
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def mem_mb():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except:
        return 0

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


# --- Job store (NEW) ---
jobs: Dict[str, Dict[str, Any]] = {}
JOB_TTL_SEC = 3600  # 1 hour

def cleanup_jobs():
    now = time.time()
    expired = [jid for jid, j in jobs.items() if j.get("expires_at", 0) < now]
    for jid in expired:
        jobs.pop(jid, None)

# --- Types ---

class AnalyzeMode(str, Enum):
    PREVIEW = "PREVIEW"
    EARLY_ACCESS = "EARLY_ACCESS"
    FULL = "FULL"

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
    Compute STFT-based chroma features (more stable than CQT on Render).
    Returns: (12, T)
    """
    # STFT-based chroma (Numba/JIT依存が小さく、Renderで安定しやすい)
    n_fft = 2048
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)

    # Log compression: log(1 + k * chroma)
    k = 10.0
    chroma_log = np.log1p(k * chroma)

    # L1 Normalization per frame
    chroma_norm = chroma_log / (np.sum(chroma_log, axis=0, keepdims=True) + 1e-8)
    return chroma_norm

def compute_bass_chroma(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """
    Compute Bass Chroma (STFT based).
    Returns: (12, T)
    """
    # 低域を強調したいので、簡易にローカット後の y を使うのではなく、
    # 低域強調のために少しカットオフを上げない/または別フィルタ設計も可能。
    n_fft = 4096
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)

    k = 10.0
    chroma_log = np.log1p(k * chroma)
    chroma_norm = chroma_log / (np.sum(chroma_log, axis=0, keepdims=True) + 1e-8)
    return chroma_norm

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
            best_mode = ""
        if min_score > best_score:
            best_score = min_score
            best_root = i
            best_mode = "m"

    return NOTE_NAMES[best_root], best_mode

def get_diatonic_chords_for_key(root_name: str, mode: str) -> list[str]:
    """
    Get list of diatonic chords for a given key.
    """
    root_index = NOTE_NAMES.index(root_name)
    major_scale_degrees = [0, 2, 4, 5, 7, 9, 11]

    if mode == "maj" or mode == "":
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

def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x * 0.0
    return x / n

def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = _l2_normalize(a, eps)
    b = _l2_normalize(b, eps)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)

def detect_chords_matrix(
    main_matrix: np.ndarray,   # (S, 12)
    bass_matrix: np.ndarray,   # (S, 12)
    penalty_mask: Optional[np.ndarray] = None,
    penalty_value: float = 0.15,
    main_weight: float = 0.7,
    bass_weight: float = 0.3,
    # Stagnation prevention params (User "Golden Master")
    flux_threshold: float = 0.15,         
    high_flux_threshold: float = 0.35,
    max_repeat_segments: int = 6,         # Lowered to 6 (approx 6s) for stricter UX
    min_hold_segments: int = 2,           
    same_chord_penalty: float = 0.20,
    long_stag_penalty: float = 0.60,
    topk: int = 3,
) -> list[str]:
    """
    Detect chords using weighted combination of main and bass chroma,
    with Stagnation Prevention logic to avoid "sticky" chords.
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

    # 1. Calculate Raw Scores (C, S)
    # ---------------------------------------------------------
    main_scores = cosine_similarity_matrix(TEMPLATE_MATRIX, main_matrix.T)  # (C, S)

    num_chords = TEMPLATE_MATRIX.shape[0]
    bass_scores = np.zeros((num_chords, num_segments))

    for chord_idx, label in enumerate(CHORD_LABELS):
        root_idx = chord_root_index(label)
        bass_scores[chord_idx, :] = bass_matrix[:, root_idx]

    if np.max(bass_scores) > 0:
        bass_scores = bass_scores / (np.max(bass_scores) + 1e-8)

    final_scores = main_scores * main_weight + bass_scores * bass_weight

    # Diatonic/Mode Penalty (Global)
    if penalty_mask is not None and penalty_mask.shape[0] == num_chords:
        final_scores[penalty_mask, :] -= penalty_value

    # 2. Stagnation Prevention (Iterative Decoding)
    # ---------------------------------------------------------
    # Mapping to user's variable names for clarity
    chroma = main_matrix      # (n_segments, 12)
    scores = final_scores.T   # (n_segments, n_chords)
    chord_labels = CHORD_LABELS
    
    # Precompute flux (delta)
    delta = np.zeros(num_segments, dtype=np.float32)
    for i in range(1, num_segments):
        cs = _cosine_similarity(chroma[i], chroma[i-1])
        delta[i] = 1.0 - cs  # cosine distance-ish

    out_idx = np.zeros(num_segments, dtype=np.int32)
    
    # Initialize with first segment argmax
    last = int(np.argmax(scores[0]))
    out_idx[0] = last
    run_length = 1

    for i in range(1, num_segments):
        row = scores[i].astype(np.float32, copy=True)
        
        # Get top-k candidates (indices)
        # Using argpartition for speed, identifying top K best scores
        k = min(topk, num_chords)
        # Note: argpartition puts the k-th element in sorted position, others undefined order
        # We want indices of the largest k elements.
        topk_unsorted = np.argpartition(row, -k)[-k:]
        # Sort these top k indices by score descending
        topk_idx = topk_unsorted[np.argsort(-row[topk_unsorted])]
        
        best = int(topk_idx[0])

        # ---- Rule C: Min hold (guard against flicker)
        # If we just switched recently, and flux isn't high, prefer stability.
        if run_length < min_hold_segments and delta[i] < high_flux_threshold:
            out_idx[i] = last
            run_length += 1
            # Skip other rules
            continue

        # ---- Rule B: Long stagnation (UX protection) - strongest intervention
        # If we have suppressed the same chord for too long, try to force a switch.
        if run_length >= max_repeat_segments:
            # Case 1: High Flux -> Strong Penalty (Existing logic)
            # The audio is changing, but the chord stayed same -> likely wrong.
            if delta[i] >= flux_threshold:
                # Strongly penalize last chord to encourage switching
                row[last] = row[last] - long_stag_penalty
                # Recompute best after penalty
                best2 = int(np.argmax(row))
                chosen = best2 if best2 != last else best2  # if still last, accept
            
            # Case 2: Low Flux but Weak Confidence (New logic)
            # The audio isn't changing much (low flux), but we've been here too long.
            # If the current best chord is "barely" winning, it might be an error.
            # If the gap to 2nd place is small, force a switch to break monotony.
            else:
                 if len(topk_idx) >= 2:
                     cand2 = int(topk_idx[1])
                     # Check gap in ORIGINAL scores (best is top1)
                     gap = scores[i, best] - scores[i, cand2]
                     
                     # If ambiguous, force switch to 2nd best to avoid infinite stagnation
                     if gap <= 0.10:
                         chosen = cand2
                     else:
                         chosen = best
                 else:
                     chosen = best

        else:
            chosen = best

            # ---- Rule A: High flux stagnation
            # Normal check: if flux is high but we picked the same chord, penalize it slightly.
            if (delta[i] >= flux_threshold) and (best == last):
                # Apply modest penalty to "same as last" and reselect
                row[last] = row[last] - same_chord_penalty
                best2 = int(np.argmax(row))

                if best2 != last:
                    chosen = best2
                else:
                    # Still stuck: fall back to 2nd candidate if available
                    if len(topk_idx) >= 2:
                        cand2 = int(topk_idx[1])
                        # Only switch if the gap is not huge (avoid random jumps)
                        if (scores[i, best] - scores[i, cand2]) <= 0.10:
                            chosen = cand2

        out_idx[i] = chosen

        # Update run length
        if chosen == last:
            run_length += 1
        else:
            last = chosen
            run_length = 1

    return [chord_labels[j] for j in out_idx]

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
    # ビートが全く検出できない場合 → 時間分割フォールバック (0.5秒間隔)
    if beat_times is None or len(beat_times) < 2:
        if num_frames == 0:
            return np.zeros((0, 12)), []
            
        start_time_all = float(times[0])
        end_time_all = float(times[-1])
        duration = end_time_all - start_time_all
        
        # Fallback interval (e.g. 0.5s = 120BPM 1beat approx)
        interval = 0.5
        num_fallback_segments = int(math.ceil(duration / interval))
        
        segment_chroma_list = []
        segments = []
        frame_indices = np.arange(len(times))
        
        for i in range(num_fallback_segments):
            s = start_time_all + i * interval
            e = min(start_time_all + (i + 1) * interval, end_time_all)
            if s >= e: break
            
            mask = (times >= s) & (times < e)
            idx = frame_indices[mask]
            
            if len(idx) > 0:
                seg_c = np.mean(chroma[:, idx], axis=1)
                segment_chroma_list.append(seg_c)
                segments.append((s, e))
        
        if not segment_chroma_list:
             return np.zeros((0, 12)), []
             
        return np.stack(segment_chroma_list, axis=0), segments

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

def analyze_audio_file(file_path: str, progress_callback=None, offset_sec: float = 0.0, duration_limit_sec: float | None = None) -> dict:
    """Core analysis logic reusable for both uploads and URLs."""
    
    def _progress(p: float):
        if progress_callback:
            try:
                progress_callback(float(p))
            except Exception:
                pass

    print(f"[DEBUG] Starting analysis for {file_path} (offset={offset_sec}, dur={duration_limit_sec})")
    print(f"mem start: {mem_mb():.1f} MB")
    _progress(5) # Start

    try:
        # 1. Load & Preprocess
        # LIMIT DURATION to 2 minutes for stability (Render 502/OOM fix)
        MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "120"))

        # チャンク長の決定：duration_limit_sec が来ていれば優先、なければ MAX_ANALYSIS_SEC
        load_dur = duration_limit_sec if duration_limit_sec is not None else MAX_ANALYSIS_SEC

        y, sr = librosa.load(file_path, sr=22050, mono=True, offset=float(offset_sec), duration=float(load_dur))
        print(f"mem after load: {mem_mb():.1f} MB")
        _progress(20) # Loaded
        
        print(f"[DEBUG] Audio loaded. Size: {y.size}, SR: {sr}")
        if y.size == 0:
            raise ValueError("Audio file is empty or unreadable")

        y = highpass_filter(y, sr)
        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        print(f"[DEBUG] Audio duration: {duration_sec}s")

        # 2. Beat tracking (Removed for speed/stability)
        # Using fixed 0.5s segments (120 BPM)
        bpm = 120.0
        print(f"[DEBUG] Using fixed grid (BPM: {bpm})")
        # _progress(35) # Skip beat track progress
        beat_frames = [] # Unused
        _progress(35)

        # 3. Chroma
        print("[DEBUG] Computing chroma...")
        hop_length = 4096
        chroma = compute_chroma_log(y, sr, hop_length=hop_length)
        print(f"mem after chroma: {mem_mb():.1f} MB")
        
        bass_chroma = chroma # compute_bass_chroma(y, sr, hop_length=hop_length)
        print(f"[DEBUG] Bass chroma disabled for memory stability")
        print(f"mem after bass chroma: {mem_mb():.1f} MB")
        print(f"[DEBUG] Chroma shape: {chroma.shape}")
        _progress(60) # Chroma done

        if chroma.shape[1] == 0:
            raise ValueError("Chroma extraction failed or audio too short")

        # 4. Time axes
        # Fixed 0.5s intervals for 120 beats
        beat_times = np.arange(0, duration_sec, 0.5)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

        # 5. Aggregate per segment
        print("[DEBUG] Aggregating segments...")
        main_matrix, segments = aggregate_chroma_per_segment(chroma, times, beat_times, beats_per_segment=2)
        bass_matrix, _ = aggregate_chroma_per_segment(bass_chroma, times, beat_times, beats_per_segment=2)
        print(f"[DEBUG] Segments: {len(segments)}")
        _progress(75) # Aggregation done

        # 6. Key estimation
        key_root, key_mode = estimate_key_from_chroma(chroma)
        estimated_key = f"{key_root}{key_mode}"
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
        _progress(90) # Detection done

        smoothed_chords = smooth_chord_sequence(raw_chords)

        print(f"[DEBUG] unique chords: {len(set(smoothed_chords))}")
        print(f"[DEBUG] first 20 chords: {smoothed_chords[:20]}")

        bars = []
        for i, chord_name in enumerate(smoothed_chords):
            # Granular progress for final loop (90 -> 99)
            if len(smoothed_chords) > 0:
                 progress_percent = 90 + 9 * ((i + 1) / len(smoothed_chords))
                 _progress(progress_percent)

            tab = chord_to_tab(chord_name)
            bars.append({
                "bar": i + 1,
                "chord": chord_name,
                "tab": {
                    "frets": tab
                } if tab else None
            })
        
        print(f"[DEBUG] Analysis complete. Returning {len(bars)} bars.")
        _progress(99)
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



@app.on_event("startup")
def startup_event():
    # Warmup librosa on startup to reduce first-request latency
    try:
        y = np.zeros(22050)
        librosa.feature.chroma_stft(y=y, sr=22050)
        print("[INFO] Warmup complete")
    except:
        pass

def run_analysis_bg(job_id: str, file_path: str, mode: AnalyzeMode = AnalyzeMode.PREVIEW):
    cleanup_jobs()
    
    # Init progress (Store mode in job)
    jobs[job_id] = {
        **jobs.get(job_id, {}),
        "status": "analyzing",
        "mode": mode,
        # Use 0.01 (1%) as "Started" signal. 0.0 might be confused with "not started"
        "progress": 0.01, 
        "updated_at": time.time(),
        "started_at": time.time()
    }

    def update_progress(p: float):
        p = max(0.0, min(100.0, p))
        job = jobs.get(job_id, {})
        # Only update if job still exists
        if job:
            jobs[job_id] = {
                **job,
                "progress": p,
                "updated_at": time.time(),
                "started_at": job.get("started_at", time.time())
            }
        
    # FORCE UPDATE to prove thread is alive (2%)
    update_progress(2.0)

    try:
        # --- Mode Enforcement ---
        # 1. Preview Hardcap
        if mode == AnalyzeMode.PREVIEW:
             # Force 60s hardcap (ignore environment or input)
             MAX_ANALYSIS_SEC = 60.0
             print("[INFO] Mode: PREVIEW -> Forced duration 60.0s")
        else:
             # Normal logic
             MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "120"))

        # 2. Usage Check (Early Access)
        if mode == AnalyzeMode.EARLY_ACCESS:
            # TODO: song-hash based deduplication
            # Current: Simple 2-use limit placeholder (Logic would go here or at endpoint level)
            pass
            
        CHUNK_SEC = float(os.getenv("CHUNK_SEC", "30"))

        # 3) Remove initial get_duration to avoid "decode stall"
        # We process until MAX_ANALYSIS_SEC or EOF
        
        bpm = 120.0
        seconds_per_beat = 60.0 / bpm
        segment_duration = seconds_per_beat * 2  # beats_per_segment=2 → 1秒

        all_bars: list[dict] = []
        key_votes: list[str] = []

        chunk_idx = 0
        offset = 0.0
        
        # Estimate total chunks for progress calculation (assuming MAX)
        # This is an approximation since we might stop early, but ensures 0-100 scale
        estimated_total_chunks = int(math.ceil(MAX_ANALYSIS_SEC / CHUNK_SEC))

        while offset < MAX_ANALYSIS_SEC:
            dur = min(CHUNK_SEC, MAX_ANALYSIS_SEC - offset)
            
            # Progress calculation based on estimated max duration
            base = (chunk_idx / estimated_total_chunks) * 100.0
            span = 100.0 / estimated_total_chunks
            
            # FORCE UPDATE before heavy chunk processing (monotonic: base + 1%)
            update_progress(base + 1.0)

            def make_chunk_progress_cb():
                def cb(p_in_chunk: float):
                    # analyze_audio_file 0-100 -> normalize 0-1 -> span
                    p01 = max(0.0, min(100.0, p_in_chunk)) / 100.0
                    update_progress(base + p01 * span)
                return cb

            chunk_cb = make_chunk_progress_cb()

            raw = analyze_audio_file(
                file_path,
                progress_callback=chunk_cb,
                offset_sec=offset,
                duration_limit_sec=dur
            )

            # Check for effective end of file (short read)
            actual_dur = raw["duration_sec"]
            
            chunk_bars = raw["bars"]
            key_votes.append(raw.get("key", "Unknown"))

            for i, bar in enumerate(chunk_bars):
                start_sec = offset + i * segment_duration
                
                # Check bounds against actual chunk duration
                chunk_end_abs = offset + actual_dur
                
                # Setup this segment's end
                end_sec = offset + (i + 1) * segment_duration
                
                if start_sec >= chunk_end_abs:
                    break
                if end_sec > chunk_end_abs:
                    end_sec = chunk_end_abs

                # No merging, strictly append grid items
                # Frontend expects "bars" list where index corresponds to time slot
                bar_obj = {
                    "bar": len(all_bars) + 1,
                    "chord": bar["chord"],
                    "tab": bar.get("tab")
                }
                all_bars.append(bar_obj)

            # Check exit conditions
            # If we got significantly less audio than requested, we hit EOF
            if actual_dur < (dur - 0.5) or actual_dur <= 0.1:
                offset += actual_dur
                break
            
            offset += dur
            chunk_idx += 1

        # key matches first chunk
        key = key_votes[0] if key_votes else "Unknown"

        final_result = {
            "bpm": bpm,
            "duration_sec": round(offset, 1),
            "time_signature": "2/4",
            "key": key,
            # Strict Return Schema based on Mode
            "mode": mode,
            "is_preview": (mode == AnalyzeMode.PREVIEW),
            "analyzed_duration_sec": round(offset, 1),
            "export_allowed": (mode == AnalyzeMode.EARLY_ACCESS or mode == AnalyzeMode.FULL),
            "bars": None if mode == AnalyzeMode.PREVIEW else all_bars
        }

        jobs[job_id] = {
            **jobs.get(job_id, {}),
            "status": "done",
            "progress": 100.0,
            "done_at": time.time(),
            "result": final_result,
        }
    except Exception as e:
        # Catch-all for thread safety
        print(f"[ERROR] Thread crashed: {e}")
        import traceback
        traceback.print_exc()
        jobs[job_id] = {
            **jobs.get(job_id, {}),
            "status": "error",
            "done_at": time.time(),
            "error": str(e),
        }

    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass



@app.post("/analyze")
async def analyze(
    mode: Optional[AnalyzeMode] = Form(None), # Require explicit mode in future, allow None for fallback now
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None
):
    # Transition Logic: Fallback to EARLY_ACCESS if missing
    if mode is None:
        print("[WARN] Missing mode in /analyze request. Fallback to EARLY_ACCESS.")
        mode = AnalyzeMode.EARLY_ACCESS

    # Negative Check: Full without Payment (Mock for now)
    # in real world, we check user session/subscription here.
    # if mode == AnalyzeMode.FULL and not is_paid_user(): raise 403
    
    return await _process_analyze(file, mode)

@app.post("/analyze/preview")
async def analyze_preview(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None
):
    # Force PREVIEW mode, ignore client input
    return await _process_analyze(file, AnalyzeMode.PREVIEW)

# Shared Logic
async def _process_analyze(file: UploadFile, mode: AnalyzeMode):
    # 1. Validate File Size
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

    cleanup_jobs()

    job_id = str(uuid.uuid4())
    now = time.time()
    jobs[job_id] = {
        "status": "analyzing",
        "submitted_at": now,
        "expires_at": now + JOB_TTL_SEC,
    }

    # Use a unique name for TEMP storage
    safe_filename = f"{job_id}{ext}"
    file_path = os.path.join(TEMP_DIR, safe_filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Use Thread instead of BackgroundTasks for better survival on Render free tier
    # (BackgroundTasks are tied to request lifecycle, Thread is slightly more detached)
    threading.Thread(target=run_analysis_bg, args=(job_id, file_path, mode)).start()

    return JSONResponse(status_code=202, content={"job_id": job_id})

@app.get("/analyze/status/{job_id}")
def analyze_status(job_id: str):
    cleanup_jobs()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "status": job.get("status"),
        "updated_at": job.get("updated_at") or job.get("done_at") or job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "progress": job.get("progress", 0.0),
        "error": job.get("error")
    }

@app.get("/analyze/result/{job_id}")
def analyze_result(job_id: str):
    cleanup_jobs()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="job not done")
    return job.get("result")

@app.post("/analyze/url")
async def analyze_url(
    url: str = Form(...),
    mode: Optional[AnalyzeMode] = Form(None),
    cookies: UploadFile | None = File(None),
    background_tasks: BackgroundTasks = None
):
    if mode is None:
         print("[WARN] Missing mode in /analyze/url request. Fallback to EARLY_ACCESS.")
         mode = AnalyzeMode.EARLY_ACCESS
    
    return await _process_analyze_url(url, mode, cookies)

@app.post("/analyze/url/preview")
async def analyze_url_preview(
    url: str = Form(...),
    cookies: UploadFile | None = File(None),
    background_tasks: BackgroundTasks = None
):
    return await _process_analyze_url(url, AnalyzeMode.PREVIEW, cookies)

async def _process_analyze_url(url: str, mode: AnalyzeMode, cookies: UploadFile | None):
    cleanup_jobs()
    
    cookie_path = None
    try:
        # Validate and save cookies if provided
        if cookies:
            if not cookies.filename.endswith(".txt"):
                raise HTTPException(status_code=400, detail="Cookie file must be a .txt file")
            
            suffix = os.path.splitext(cookies.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                size = 0
                while True:
                    chunk = cookies.file.read(1024 * 1024) 
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 1024 * 1024 * 1: 
                        raise HTTPException(status_code=400, detail="Cookie file too large (limit 1MB)")
                    tmp.write(chunk)
                cookie_path = tmp.name
            print(f"[DEBUG] cookies loaded: {cookie_path}")

        # Download synchronously (usually fast enough, but ideally this would be part of the job too)
        # However, for MVP, we'll keep download sync to get file_path, then offload analysis.
        # IF download is > 25s, this might still 502. 
        # But moving download to BG requires passing 'url' and 'cookie_path' to BG.
        # 'run_analysis_bg' expects 'file_path'.
        # So we download here. If it times out, it times out. 
        # User accepted focus on /analyze (file upload). 
        # But we can try to be safe.
        file_path = download_youtube_audio(url, cookie_path=cookie_path)
        
        # Now create job
        job_id = str(uuid.uuid4())
        now = time.time()
        jobs[job_id] = {
            "status": "analyzing",
            "submitted_at": now,
            "expires_at": now + JOB_TTL_SEC,
        }
        
        # Threading for URL analysis too
        threading.Thread(target=run_analysis_bg, args=(job_id, file_path, mode)).start()
        
        return JSONResponse(status_code=202, content={"job_id": job_id})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Secure cleanup of COOKIES only. Audio file is needed for BG task.
        if cookie_path and os.path.exists(cookie_path):
            try:
                os.remove(cookie_path)
            except:
                pass

        

