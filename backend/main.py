import os
import anyio

import psutil

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request
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

@app.middleware("http")
async def log_req_lifecycle(request: Request, call_next):
    t0 = time.time()
    print(f"[REQ-START] {request.method} {request.url.path}")
    try:
        resp = await call_next(request)
        return resp
    finally:
        dt = (time.time() - t0) * 1000
        print(f"[REQ-END]   {request.method} {request.url.path} {dt:.1f}ms")


# --- Job store (NEW) ---
jobs: Dict[str, Dict[str, Any]] = {}
JOB_TTL_SEC = 3600  # 1 hour

def cleanup_jobs():
    now = time.time()
    expired = [jid for jid, j in jobs.items() if j.get("expires_at", 0) < now]
    for jid in expired:
        jobs.pop(jid, None)

    for jid in expired:
        jobs.pop(jid, None)

def _save_upload_sync(src_fileobj, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        shutil.copyfileobj(src_fileobj, f, length=1024 * 1024)  # 1MB chunk

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

    # Dominant 7th
    "C7":  ["x", "3", "2", "3", "1", "0"],
    "C#7": ["x", "4", "3", "4", "2", "x"],
    "D7":  ["x", "x", "0", "2", "1", "2"],
    "D#7": ["x", "6", "8", "6", "8", "6"],
    "E7":  ["0", "2", "0", "1", "0", "0"],
    "F7":  ["1", "3", "1", "2", "1", "1"],
    "F#7": ["2", "4", "2", "3", "2", "2"],
    "G7":  ["3", "2", "0", "0", "0", "1"],
    "G#7": ["4", "6", "4", "5", "4", "4"],
    "A7":  ["x", "0", "2", "0", "2", "0"],
    "A#7": ["x", "1", "3", "1", "3", "1"],
    "B7":  ["x", "2", "1", "2", "0", "2"],

    # Major 7th
    "Cmaj7":  ["x", "3", "2", "0", "0", "0"],
    "C#maj7": ["x", "4", "3", "1", "1", "1"],
    "Dmaj7":  ["x", "x", "0", "2", "2", "2"],
    "D#maj7": ["x", "6", "8", "7", "8", "6"],
    "Emaj7":  ["0", "2", "1", "1", "0", "0"],
    "Fmaj7":  ["1", "3", "2", "2", "1", "0"],
    "F#maj7": ["2", "4", "3", "3", "2", "1"],
    "Gmaj7":  ["3", "2", "0", "0", "0", "2"],
    "G#maj7": ["4", "6", "5", "5", "4", "3"],
    "Amaj7":  ["x", "0", "2", "1", "2", "0"],
    "A#maj7": ["x", "1", "3", "2", "3", "1"],
    "Bmaj7":  ["x", "2", "4", "3", "4", "2"],

    # sus4
    "Csus4":  ["x", "3", "3", "0", "1", "1"],
    "C#sus4": ["x", "4", "6", "6", "7", "4"],
    "Dsus4":  ["x", "x", "0", "2", "3", "3"],
    "D#sus4": ["x", "6", "8", "8", "9", "6"],
    "Esus4":  ["0", "2", "2", "2", "0", "0"],
    "Fsus4":  ["1", "3", "3", "3", "1", "1"],
    "F#sus4": ["2", "4", "4", "4", "2", "2"],
    "Gsus4":  ["3", "3", "0", "0", "1", "3"],
    "G#sus4": ["4", "6", "6", "6", "4", "4"],
    "Asus4":  ["x", "0", "2", "2", "3", "0"],
    "A#sus4": ["x", "1", "3", "3", "4", "1"],
    "Bsus4":  ["x", "2", "4", "4", "5", "2"],

    # Minor 7th (m7)
    "Cm7":  ["x", "3", "5", "3", "4", "3"],
    "C#m7": ["x", "4", "6", "4", "5", "4"],
    "Dm7":  ["x", "x", "0", "2", "1", "1"],
    "D#m7": ["x", "6", "8", "6", "7", "6"],
    "Em7":  ["0", "2", "0", "0", "0", "0"],
    "Fm7":  ["1", "3", "1", "1", "1", "1"],
    "F#m7": ["2", "4", "2", "2", "2", "2"],
    "Gm7":  ["3", "5", "3", "3", "3", "3"],
    "G#m7": ["4", "6", "4", "4", "4", "4"],
    "Am7":  ["x", "0", "2", "0", "1", "0"],
    "A#m7": ["x", "1", "3", "1", "2", "1"],
    "Bm7":  ["x", "2", "0", "2", "0", "2"],
}

def chord_to_tab(chord: str) -> Optional[list[str]]:
    key = chord.strip()
    return CHORD_TO_TAB.get(key)

# --- Timing Helpers ---

from typing import Dict, Any, Optional, List, Tuple

def _parse_time_signature(ts: str) -> Tuple[int, int]:
    # "2/4" -> (2,4)
    try:
        if not ts: return 4, 4
        a, b = ts.split("/")
        num = int(a)
        den = int(b)
        if num <= 0 or den <= 0:
            raise ValueError
        return num, den
    except Exception:
        # fallback: assume 4/4
        return 4, 4

def add_bar_timing(
    bars: List[Dict[str, Any]],
    bpm: float | int | None,
    time_signature: str | None,
    analyzed_duration_sec: float | int | None,
) -> List[Dict[str, Any]]:
    n = len(bars)
    if n == 0:
        return bars

    dur = float(analyzed_duration_sec) if analyzed_duration_sec is not None else None
    # duration が無いなら、最後は n で均等割りに逃がす
    if dur is not None and not math.isfinite(dur):
        dur = None

    # bpm が健全なら拍子から秒/小節を作る。ダメなら duration / n にフォールバック
    bpm_f = float(bpm) if bpm is not None else None
    if bpm_f is not None and (not math.isfinite(bpm_f) or bpm_f <= 0):
        bpm_f = None

    ts = time_signature or "4/4"
    beats_per_bar, _ = _parse_time_signature(ts)

    if bpm_f is not None:
        sec_per_bar = (60.0 / bpm_f) * float(beats_per_bar)
    else:
        # bpm が無い/壊れている場合の最終手段
        sec_per_bar = (dur / n) if (dur is not None and dur > 0) else 1.0

    # 生成
    for i, b in enumerate(bars):
        start = i * sec_per_bar
        end = (i + 1) * sec_per_bar
        if dur is not None:
            end = min(end, dur)
        b["start_sec"] = float(start)
        b["end_sec"] = float(end)

    # 誤差吸収：duration があるなら最後はきっちり合わせる
    if dur is not None and dur > 0:
        bars[-1]["end_sec"] = float(dur)

    return bars

# --- Signal Processing ---

def highpass_filter(y: np.ndarray, sr: int, cutoff_hz: float = 60.0) -> np.ndarray:
    """Sub-bass low-cut filter"""
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, y)

def lowpass_filter(y: np.ndarray, sr: int, cutoff_hz: float = 200.0, order: int = 5) -> np.ndarray:
    """Low-pass filter for bass extraction"""
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos, y)

def bandpass_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """Band-pass filter using Butterworth design."""
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, y)

def compute_chroma_log(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """
    Compute STFT-based chroma features with HPSS (harmonic separation).
    HPSS removes percussive components (drums) for cleaner chroma.
    Returns: (12, T)
    """
    # HPSS: 調波成分のみ使用（ドラム打撃をクロマから除去）
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)

    # n_fft=4096 で周波数解像度を向上（低音域のピッチ分離に有効）
    n_fft = 4096
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, hop_length=hop_length, n_fft=n_fft)
    del y_harmonic

    # Log compression: log(1 + k * chroma)
    k = 10.0
    chroma_log = np.log1p(k * chroma)

    # L1 Normalization per frame
    chroma_norm = chroma_log / (np.sum(chroma_log, axis=0, keepdims=True) + 1e-8)
    return chroma_norm

def compute_bass_chroma(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """
    Compute Bass Chroma from low-frequency band (60-300Hz).
    Isolates bass/root notes for improved root detection.
    Returns: (12, T)
    """
    # 低域（60-300Hz）のみ抽出してベース音の根音を正確に取得
    y_bass = bandpass_filter(y, sr, low_hz=60, high_hz=300)
    n_fft = 4096
    chroma = librosa.feature.chroma_stft(y=y_bass, sr=sr, hop_length=hop_length, n_fft=n_fft)
    del y_bass

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
    Includes: Major, Minor, 7th (dominant), m7, Maj7, sus4
    Returns:
        templates: dict {name: vector}
        labels: list of chord names
        matrix: np.ndarray (NumChords, 12)
    """
    templates = {}

    # Major: root(1.0) + maj3(0.5) + 5th(0.7)
    base_major = np.zeros(12)
    base_major[0] = 1.0   # root
    base_major[7] = 0.7   # 5th
    base_major[4] = 0.5   # 3rd

    # Minor: root(1.0) + min3(0.5) + 5th(0.7)
    base_minor = np.zeros(12)
    base_minor[0] = 1.0
    base_minor[7] = 0.7
    base_minor[3] = 0.5

    # Dominant 7th: root(1.0) + maj3(0.4) + 5th(0.6) + min7(0.35)
    base_7 = np.zeros(12)
    base_7[0] = 1.0
    base_7[4] = 0.4   # maj3
    base_7[7] = 0.6   # 5th
    base_7[10] = 0.35  # min7

    # Major 7th: root(1.0) + maj3(0.4) + 5th(0.6) + maj7(0.35)
    base_maj7 = np.zeros(12)
    base_maj7[0] = 1.0
    base_maj7[4] = 0.4
    base_maj7[7] = 0.6
    base_maj7[11] = 0.35  # maj7

    # sus4: root(1.0) + 4th(0.5) + 5th(0.7)
    base_sus4 = np.zeros(12)
    base_sus4[0] = 1.0
    base_sus4[5] = 0.5   # 4th
    base_sus4[7] = 0.7   # 5th

    # Minor 7th (m7): root(1.0) + min3(0.4) + 5th(0.6) + min7(0.35)
    base_m7 = np.zeros(12)
    base_m7[0] = 1.0     # root
    base_m7[3] = 0.4     # minor 3rd
    base_m7[7] = 0.6     # perfect 5th
    base_m7[10] = 0.35   # minor 7th

    for i, name in enumerate(NOTE_NAMES):
        templates[f"{name}"] = rotate_template(base_major, i)
        templates[f"{name}m"] = rotate_template(base_minor, i)
        templates[f"{name}7"] = rotate_template(base_7, i)
        templates[f"{name}maj7"] = rotate_template(base_maj7, i)
        templates[f"{name}sus4"] = rotate_template(base_sus4, i)
        templates[f"{name}m7"] = rotate_template(base_m7, i)

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
    Uses music-theory-correct 7th chord quality per scale degree.
    Also includes sus4 for degrees I, II, IV, V (common suspensions).
    """
    root_index = NOTE_NAMES.index(root_name)

    if mode == "maj" or mode == "":
        # Major scale degrees: I, ii, iii, IV, V, vi, vii
        degrees = [0, 2, 4, 5, 7, 9, 11]
        # Triad qualities
        triad_qualities = ["", "m", "m", "", "", "m", "m"]
        # Diatonic 7th chord qualities per degree:
        # I=maj7, ii=m7, iii=m7, IV=maj7, V=7, vi=m7, vii=m7 (approx for dim7)
        seventh_qualities = ["maj7", "m7", "m7", "maj7", "7", "m7", "m7"]
        # sus4 is common on I, II, IV, V
        sus4_degrees = {0, 2, 5, 7}  # scale degree semitones where sus4 is natural
    else:
        # Natural Minor (Aeolian) degrees: i, ii, III, iv, v, VI, VII
        degrees = [0, 2, 3, 5, 7, 8, 10]
        triad_qualities = ["m", "m", "", "m", "m", "", ""]
        # i=m7, ii=m7(approx dim), III=maj7, iv=m7, v=m7, VI=maj7, VII=7
        seventh_qualities = ["m7", "m7", "maj7", "m7", "m7", "maj7", "7"]
        sus4_degrees = {0, 2, 5, 7}

    chords = []
    for i, (deg, tq, sq) in enumerate(zip(degrees, triad_qualities, seventh_qualities)):
        note = NOTE_NAMES[(root_index + deg) % 12]
        # Basic triad
        chords.append(note + tq)
        # Diatonic 7th
        chords.append(note + sq)
        # sus4 only on specific degrees
        if deg in sus4_degrees:
            chords.append(note + "sus4")

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
    penalty_value: float = 0.20,
    main_weight: float = 0.6,
    bass_weight: float = 0.35,
    # Stagnation prevention params (User "Golden Master")
    flux_threshold: float = 0.15,
    high_flux_threshold: float = 0.35,
    max_repeat_segments: int = 4,         # Lowered to 4 (approx 4s) for stricter UX [Iteration 2]
    min_hold_segments: int = 2,
    same_chord_penalty: float = 0.20,
    long_stag_penalty: float = 0.85,      # Increased from 0.60 to 0.85 [Iteration 2]
    topk: int = 3,
    forced_last_chord: Optional[str] = None,
    forced_run_length: Optional[int] = None,
) -> tuple[list[str], str, int]:
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

    # Initialize with forced state if provided (cross-chunk continuity)
    if forced_last_chord is not None and forced_run_length is not None:
        try:
            last = chord_labels.index(forced_last_chord)
            run_length = forced_run_length
            print(f"[StagnationContinuity] Forcing initial state: last={forced_last_chord}, run_length={run_length}")
        except ValueError:
            # Chord not in labels, fall back to argmax
            print(f"[WARN] forced_last_chord '{forced_last_chord}' not in chord_labels, using argmax")
            last = int(np.argmax(scores[0]))
            run_length = 1
    else:
        # Standard initialization with first segment argmax
        last = int(np.argmax(scores[0]))
        run_length = 1

    out_idx[0] = last

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
            excess = run_length - max_repeat_segments  # 0 at threshold, grows each bar

            # Hard cap: at 2x max_repeat_segments, force switch regardless of flux/confidence
            hard_cap = max_repeat_segments * 2  # default: 12

            if run_length >= hard_cap:
                # Absolute limit reached - apply heavy penalty unconditionally
                row[last] = row[last] - long_stag_penalty
                best2 = int(np.argmax(row))
                chosen = best2  # accept whatever wins after penalty

            elif delta[i] >= flux_threshold:
                # Case 1: High Flux -> Strong Penalty (unchanged)
                row[last] = row[last] - long_stag_penalty
                best2 = int(np.argmax(row))
                chosen = best2 if best2 != last else best2

            # Case 2: Low Flux - progressive gap escalation
            # Gap threshold increases by 0.03 per excess bar, making it progressively
            # harder for the incumbent chord to survive.
            else:
                if len(topk_idx) >= 2:
                    cand2 = int(topk_idx[1])
                    gap = scores[i, best] - scores[i, cand2]

                    # Progressive threshold: starts at 0.08, grows faster with excess bars [Iteration 2]
                    adjusted_gap_threshold = 0.08 + 0.05 * excess

                    if gap <= adjusted_gap_threshold:
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

    # Extract final state for cross-chunk continuity
    final_last_chord = chord_labels[last]
    final_run_length = run_length
    return [chord_labels[j] for j in out_idx], final_last_chord, final_run_length

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

def smooth_chord_sequence(chords: list[str], passes: int = 2) -> list[str]:
    """
    Smooth chord sequence by removing short outlier runs.
    Pass 1+: Single-bar outlier (A-B-A -> A-A-A)
    Pass 2+: Two-bar outlier (A-B-B-A -> A-A-A-A) when surrounded by same chord
    Multiple passes catch cascading corrections.
    """
    if len(chords) < 3:
        return chords[:]

    smoothed = chords[:]

    for _ in range(passes):
        changed = False
        result = smoothed[:]

        # Single-bar outlier: A-B-A -> A-A-A
        for i in range(1, len(result) - 1):
            prev_c = result[i - 1]
            curr_c = result[i]
            next_c = result[i + 1]
            if prev_c == next_c and curr_c != prev_c:
                result[i] = prev_c
                changed = True

        # Two-bar outlier: A-B-B-A -> A-A-A-A
        for i in range(1, len(result) - 2):
            if (result[i - 1] == result[i + 2] and
                result[i] == result[i + 1] and
                result[i] != result[i - 1]):
                result[i] = result[i - 1]
                result[i + 1] = result[i - 1]
                changed = True

        smoothed = result
        if not changed:
            break  # Converged early, skip remaining passes

    return smoothed

def smooth_chord_sequence_stagnation_aware(chords: list[str], passes: int = 2, max_run: int = 6) -> list[str]:
    """
    Smooth chord sequence while preventing long stagnation runs.

    Rules:
    1. Single-bar outlier: A-B-A -> A-A-A (existing)
    2. Two-bar outlier: A-B-B-A -> A-A-A-A (existing)
    3. Stagnation prevention: If smoothing would create run > max_run bars, preserve the outlier

    Args:
        chords: Input chord sequence
        passes: Number of smoothing passes
        max_run: Maximum allowed consecutive bars of same chord
    """
    if len(chords) < 3:
        return chords[:]

    smoothed = chords[:]

    for _ in range(passes):
        changed = False
        result = smoothed[:]

        # Single-bar outlier: A-B-A -> A-A-A
        for i in range(1, len(result) - 1):
            prev_c = result[i - 1]
            curr_c = result[i]
            next_c = result[i + 1]

            if prev_c == next_c and curr_c != prev_c:
                # Check if this smoothing would create excessive stagnation
                run_before = 1
                j = i - 1
                while j > 0 and result[j-1] == prev_c:
                    run_before += 1
                    j -= 1

                run_after = 1
                j = i + 1
                while j < len(result) - 1 and result[j+1] == next_c:
                    run_after += 1
                    j += 1

                potential_run = run_before + 1 + run_after

                # Only smooth if it doesn't create excessive stagnation
                if potential_run <= max_run:
                    result[i] = prev_c
                    changed = True

        # Two-bar outlier: A-B-B-A -> A-A-A-A (with same stagnation check)
        for i in range(1, len(result) - 2):
            if (result[i - 1] == result[i + 2] and
                result[i] == result[i + 1] and
                result[i] != result[i - 1]):

                prev_c = result[i - 1]
                run_before = 1
                j = i - 1
                while j > 0 and result[j-1] == prev_c:
                    run_before += 1
                    j -= 1

                run_after = 1
                j = i + 2
                while j < len(result) - 1 and result[j+1] == prev_c:
                    run_after += 1
                    j += 1

                potential_run = run_before + 2 + run_after

                if potential_run <= max_run:
                    result[i] = result[i - 1]
                    result[i + 1] = result[i - 1]
                    changed = True

        smoothed = result
        if not changed:
            break

    return smoothed

def break_long_stagnation_runs(chords: list[str], max_consecutive: int = 6) -> list[str]:
    """
    Break up any remaining long stagnation runs after detection and smoothing.

    If a chord runs for more than max_consecutive bars, attempt to split it
    by inserting alternative chords from surrounding context.

    This is a safety net for cases where detection and smoothing both failed
    to prevent excessive stagnation.

    Args:
        chords: Input chord sequence
        max_consecutive: Maximum allowed consecutive bars before breaking
    """
    if len(chords) <= max_consecutive:
        return chords[:]

    result = chords[:]
    i = 0

    while i < len(result):
        # Count consecutive run
        j = i
        while j < len(result) and result[j] == result[i]:
            j += 1

        run_length = j - i

        if run_length > max_consecutive:
            # Found a long run - insert breaks
            # Strategy: Every max_consecutive bars, insert a 1-bar variation
            # Use the previous or next different chord if available

            alt_chord = None

            # Strategy 1: Use adjacent different chord
            if i > 0 and result[i-1] != result[i]:
                alt_chord = result[i-1]
            elif j < len(result) and result[j] != result[i]:
                alt_chord = result[j]

            # Strategy 2: If no adjacent chord, use most frequent different chord
            if not alt_chord:
                from collections import Counter
                chord_counts = Counter(result)
                # Find most common chord that's different from current
                for chord, count in chord_counts.most_common():
                    if chord != result[i]:
                        alt_chord = chord
                        print(f"[STAGNATION] Using fallback chord: {alt_chord} (frequency: {count})")
                        break

            # Strategy 3: Ultimate fallback (rare)
            if not alt_chord:
                print(f"[STAGNATION] WARNING: Cannot break stagnation - no alternative chord available")

            if alt_chord:
                # Insert breaks at regular intervals
                insert_positions = list(range(i + max_consecutive, j, max_consecutive + 1))
                print(f"[STAGNATION] Breaking {result[i]} run (length {run_length}) with {alt_chord} at positions: {insert_positions}")
                # Work backwards to avoid index shifting
                for pos in reversed(insert_positions):
                    result[pos] = alt_chord

        i = j

    return result

# --- BPM/Tempo Verification ---

def evaluate_tempo_prior(bpm: float) -> float:
    """
    音楽理論的な妥当性スコア。
    一般的な音楽の多くは60-140 BPMに集中。

    Args:
        bpm: 評価するBPM値

    Returns:
        スコア（0.0-1.0）
    """
    if 60 <= bpm <= 100:
        return 0.7  # バラード/ブルース/R&B（0.7に調整）
    elif 100 < bpm <= 140:
        return 0.7  # ポップ/ロック（0.7に調整）
    elif 140 < bpm <= 180:
        return 0.6  # アップテンポロック/EDM（維持）
    elif 180 < bpm <= 240:
        return 0.5  # 高速ロック/パンク（0.5に上げる）
    elif 40 <= bpm < 60:
        return 0.5  # バラード（遅め、維持）
    else:
        return 0.1  # 異常値（維持）


def evaluate_bass_ac(y: np.ndarray, sr: int, candidate_bpm: float,
                     hop_length: int = 512) -> float:
    """
    低音域(20-200Hz)のオンセットエンベロープの自己相関値で
    候補BPMの周期性の強さを評価。
    バスドラム/ベースは主拍に集中するため、真のテンポで高スコア。
    """
    try:
        y_bass = lowpass_filter(y, sr, cutoff_hz=200)
        y_bass = highpass_filter(y_bass, sr, cutoff_hz=20)
        bass_env = librosa.onset.onset_strength(y=y_bass, sr=sr,
                                                 hop_length=hop_length)
        del y_bass

        ac = librosa.autocorrelate(bass_env)
        if ac[0] > 0:
            ac = ac / ac[0]

        lag = 60.0 * sr / (candidate_bpm * hop_length)
        lag_int = int(round(lag))

        if 0 < lag_int < len(ac) - 1:
            alpha = float(ac[lag_int - 1])
            beta = float(ac[lag_int])
            gamma = float(ac[lag_int + 1])
            return max(0.0, (alpha + beta + gamma) / 3.0)
        elif 0 < lag_int < len(ac):
            return max(0.0, float(ac[lag_int]))
        return 0.0
    except Exception as e:
        print(f"[WARNING] evaluate_bass_ac failed: {e}")
        return 0.0


def evaluate_fullband_ac(onset_env: np.ndarray, sr: int,
                         candidate_bpm: float,
                         hop_length: int = 512) -> float:
    """
    全帯域オンセットエンベロープの自己相関値で
    候補BPMの全体的な周期性を評価。
    """
    ac = librosa.autocorrelate(onset_env)
    if ac[0] > 0:
        ac = ac / ac[0]

    lag = 60.0 * sr / (candidate_bpm * hop_length)
    lag_int = int(round(lag))

    if 0 < lag_int < len(ac) - 1:
        alpha = float(ac[lag_int - 1])
        beta = float(ac[lag_int])
        gamma = float(ac[lag_int + 1])
        return max(0.0, (alpha + beta + gamma) / 3.0)
    elif 0 < lag_int < len(ac):
        return max(0.0, float(ac[lag_int]))
    return 0.0


def evaluate_phase_concentration(onset_env: np.ndarray, sr: int,
                                  candidate_bpm: float,
                                  hop_length: int = 512,
                                  n_bins: int = 8) -> float:
    """
    オンセットエンベロープを候補BPMの周期で折りたたみ、
    エネルギー分布の尖度を評価。
    正しいテンポ → ダウンビートにエネルギー集中（高スコア）
    倍速テンポ → エネルギーが均等分布（低スコア）
    """
    beat_interval = 60.0 / candidate_bpm

    times = librosa.frames_to_time(np.arange(len(onset_env)),
                                    sr=sr, hop_length=hop_length)
    phases = (times % beat_interval) / beat_interval

    bins = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i, phase in enumerate(phases):
        bin_idx = min(int(phase * n_bins), n_bins - 1)
        bins[bin_idx] += onset_env[i]
        counts[bin_idx] += 1

    avg_bins = bins / np.maximum(counts, 1)

    mean_val = np.mean(avg_bins)
    if mean_val > 0:
        cv = np.std(avg_bins) / mean_val
        return min(cv, 1.0)
    return 0.0


def verify_tempo_octave(y: np.ndarray, sr: int, detected_bpm: float, onset_env: np.ndarray) -> tuple[float, float]:
    """
    倍速検出を検証し、必要に応じて半分速に補正。
    位相エネルギー集中度をゲート条件として使用:
    半分速候補に明確な拍構造がある場合のみ補正を許可。
    """
    if detected_bpm < 80 or detected_bpm > 240:
        print(f"[OctaveVerify] BPM {detected_bpm:.1f} outside correction range, keeping as-is")
        return detected_bpm, 1.0

    half_bpm = detected_bpm * 0.5
    if half_bpm < 40:
        return detected_bpm, 1.0

    # 全シグナルを計算
    bass_ac_half = evaluate_bass_ac(y, sr, half_bpm)
    bass_ac_full = evaluate_bass_ac(y, sr, detected_bpm)
    full_ac_half = evaluate_fullband_ac(onset_env, sr, half_bpm)
    full_ac_full = evaluate_fullband_ac(onset_env, sr, detected_bpm)
    phase_half = evaluate_phase_concentration(onset_env, sr, half_bpm)
    phase_full = evaluate_phase_concentration(onset_env, sr, detected_bpm)
    prior_half = evaluate_tempo_prior(half_bpm)
    prior_full = evaluate_tempo_prior(detected_bpm)

    score_half = (bass_ac_half * 0.35 + full_ac_half * 0.25 +
                  phase_half * 0.20 + prior_half * 0.20)
    score_full = (bass_ac_full * 0.35 + full_ac_full * 0.25 +
                  phase_full * 0.20 + prior_full * 0.20)

    print(f"[OctaveVerify] {half_bpm:.1f} BPM (×0.5): "
          f"bass_ac={bass_ac_half:.3f}, full_ac={full_ac_half:.3f}, "
          f"phase={phase_half:.3f}, prior={prior_half:.1f}, total={score_half:.3f}")
    print(f"[OctaveVerify] {detected_bpm:.1f} BPM (×1.0): "
          f"bass_ac={bass_ac_full:.3f}, full_ac={full_ac_full:.3f}, "
          f"phase={phase_full:.3f}, prior={prior_full:.1f}, total={score_full:.3f}")

    # ゲート条件: 半分速候補のビート位相エネルギー集中度
    # 真のテンポが半分速なら、その周期でダウンビートにエネルギーが集中する。
    # 集中度が低い(< PHASE_GATE)場合、半分速にリズム構造がない → 補正しない。
    PHASE_GATE = 0.25
    if phase_half < PHASE_GATE:
        print(f"[OctaveVerify] Half-tempo phase={phase_half:.3f} < {PHASE_GATE} "
              f"(weak beat structure) -> keeping {detected_bpm:.1f}")
        print(f"[OctaveVerify] Selected: {detected_bpm:.1f} BPM (factor=x1.0, score={score_full:.3f})")
        return detected_bpm, 1.0

    # ゲート通過: 半分速に明確なリズム構造あり → スコア比較
    if score_half > score_full:
        print(f"[OctaveVerify] Selected: {half_bpm:.1f} BPM (factor=x0.5, score={score_half:.3f})")
        return half_bpm, 0.5
    else:
        print(f"[OctaveVerify] Selected: {detected_bpm:.1f} BPM (factor=x1.0, score={score_full:.3f})")
        return detected_bpm, 1.0

def refine_bar_phase(y: np.ndarray, sr: int, bpm: float, phase_offset_sec: float,
                     hop_length: int = 512, beats_per_bar: int = 4,
                     window_frames: int = 2, octave_factor: float = 1.0) -> tuple[float, float]:
    """
    2段階でビート位相と小節頭を最適化:
    Step A: バスオンセットで±半拍のビート位相微調整
    Step B: バス-スネア複合スコアでシフトテスト（ダウンビート選択）
            octave_factor=0.5の場合、半拍刻み8シフトで交互グリッドも評価
    """
    beat_duration = 60.0 / bpm
    frame_duration = hop_length / sr

    # octave_factor=0.5 の場合: Step A はスキップ (line 885) だが、
    # Step B は半拍シフトで実行してダウンビート/バックビートを判定する。

    # --- バスオンセットエンベロープ (20-200Hz) ---
    y_bass = lowpass_filter(y, sr, cutoff_hz=200)
    y_bass = highpass_filter(y_bass, sr, cutoff_hz=20)
    bass_env = librosa.onset.onset_strength(y=y_bass, sr=sr, hop_length=hop_length)
    del y_bass

    total_frames = len(bass_env)
    total_duration_sec = total_frames * frame_duration

    if np.max(bass_env) < 1e-6:
        print("[BarPhaseRefine] No bass energy detected, skipping")
        return phase_offset_sec, 0

    # --- スネアバンドエンベロープ (2000-5000Hz) ---
    y_snare = bandpass_filter(y, sr, low_hz=2000, high_hz=5000)
    snare_env = librosa.onset.onset_strength(y=y_snare, sr=sr, hop_length=hop_length)
    del y_snare

    # ============================================================
    # Step A: バス基準ビート位相微調整
    # ±半拍の範囲でフレーム単位に探索し、ビートグリッド上の
    # バスエネルギーが最大になる位相を選択
    # octave_factor=0.5の場合はスキップ（位相は元BPMのビートグリッド上
    # にあり、フレームレベル探索でグリッドからずれるのを防ぐため）
    # ============================================================
    beat_period_frames = beat_duration / frame_duration
    current_phase_frames = phase_offset_sec / frame_duration

    if octave_factor == 0.5:
        # オクターブ補正時: 位相は元BPM(2倍速)のビート上にあるので
        # フレームレベル微調整はスキップ（Step Bの半拍シフトで対応）
        refined_beat_phase = phase_offset_sec
        print(f"[BarPhaseRefine] Step A: skipped (octave_factor=0.5), "
              f"keeping phase at {refined_beat_phase*1000:.1f}ms")
    else:
        search_range_frames = int(beat_period_frames / 2)

        best_bass_score = -1.0
        best_delta = 0

        for delta in range(-search_range_frames, search_range_frames + 1):
            candidate_phase_f = current_phase_frames + delta
            if candidate_phase_f < 0:
                continue

            grid = np.arange(candidate_phase_f, total_frames, beat_period_frames)
            grid_int = np.round(grid).astype(int)
            grid_int = grid_int[(grid_int >= 0) & (grid_int < total_frames)]

            if len(grid_int) < 4:
                continue

            score = 0.0
            for f in grid_int:
                lo = max(0, f - window_frames)
                hi = min(total_frames, f + window_frames + 1)
                score += float(np.max(bass_env[lo:hi]))

            if score > best_bass_score:
                best_bass_score = score
                best_delta = delta

        refined_beat_phase = (current_phase_frames + best_delta) * frame_duration

        if best_delta != 0:
            print(f"[BarPhaseRefine] Step A: adjusted by {best_delta} frames "
                  f"({best_delta * frame_duration * 1000:.1f}ms): "
                  f"{phase_offset_sec*1000:.1f}ms -> {refined_beat_phase*1000:.1f}ms")
        else:
            print(f"[BarPhaseRefine] Step A: unchanged at {refined_beat_phase*1000:.1f}ms")

    # ============================================================
    # Step B: バックビートパターン認識型ダウンビート選択
    # score = bass_ratio + BETA * backbeat_snare_avg
    # bass_ratio = bass / (bass + snare) でバス+スネア同時ヒットを正しく評価
    # ダウンビート(beat1): 高bass_ratio(キックのみ)、高bb_snare(2&4拍にスネア) → 高score
    # バックビート(beat2,4): 低bass_ratio(キック+スネア)、低bb_snare → 低score
    # ============================================================
    BETA = 1.0

    bass_max = float(np.max(bass_env))
    snare_max = float(np.max(snare_env))
    snare_active = snare_max > (bass_max * 0.05)

    if not snare_active:
        print("[BarPhaseRefine] Step B: Snare band too weak, using bass-only scoring")
        BETA = 0.0

    bass_norm = bass_env / bass_max
    snare_norm = snare_env / snare_max if snare_max > 1e-6 else snare_env

    # 半拍刻み8候補: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    # 位相検出が裏拍にロックした場合、整数シフト[0,1,2,3]だけでは
    # 全候補が裏拍になり補正不可能。半拍シフトで表拍もテストする。
    shifts_to_test = [i * 0.5 for i in range(beats_per_bar * 2)]

    scores_b = []
    for shift in shifts_to_test:
        candidate_phase = refined_beat_phase + shift * beat_duration
        bar_starts = np.arange(candidate_phase, total_duration_sec,
                               beat_duration * beats_per_bar)
        bar_frames = (bar_starts / frame_duration).astype(int)

        bass_sum = 0.0
        snare_sum = 0.0
        count = 0
        for f in bar_frames:
            lo = max(0, f - window_frames)
            hi = min(total_frames, f + window_frames + 1)
            if lo < hi:
                bass_sum += float(np.max(bass_norm[lo:hi]))
                snare_sum += float(np.max(snare_norm[lo:hi]))
                count += 1

        # バックビート検出: 各バーの2拍目と4拍目のスネアエネルギー
        bb_snare_sum = 0.0
        bb_count = 0
        for bar_start_sec in bar_starts:
            for offset_beats in [1, 3]:  # beat 2 and beat 4
                bb_sec = bar_start_sec + offset_beats * beat_duration
                if bb_sec >= total_duration_sec:
                    continue
                bb_frame = int(bb_sec / frame_duration)
                lo = max(0, bb_frame - window_frames)
                hi = min(total_frames, bb_frame + window_frames + 1)
                if lo < hi:
                    bb_snare_sum += float(np.max(snare_norm[lo:hi]))
                    bb_count += 1

        bass_avg = bass_sum / max(count, 1)
        snare_avg = snare_sum / max(count, 1)
        bb_snare_avg = bb_snare_sum / max(bb_count, 1)
        bass_ratio = bass_avg / (bass_avg + snare_avg + 1e-8)
        composite = bass_ratio + BETA * bb_snare_avg
        scores_b.append(composite)
        print(f"[BarPhaseRefine] Step B: shift={shift}: "
              f"phase={candidate_phase*1000:.1f}ms, bars={count}, "
              f"bass={bass_avg:.4f}, snare={snare_avg:.4f}, "
              f"bb_snare={bb_snare_avg:.4f}, score={composite:.4f}")

    best_idx = int(np.argmax(scores_b))
    best_shift = shifts_to_test[best_idx]

    # ステータスクオ保護: 2%未満の改善なら変更しない
    if best_idx != 0:
        improvement = (scores_b[best_idx] - scores_b[0]) / (abs(scores_b[0]) + 1e-8)
        if improvement < 0.02:
            print(f"[BarPhaseRefine] Step B: shift={best_shift} "
                  f"improvement={improvement:.3f} < 0.02, keeping shift=0")
            best_shift = 0.0

    final_phase = refined_beat_phase + best_shift * beat_duration
    print(f"[BarPhaseRefine] Final: shift={best_shift}, "
          f"{phase_offset_sec*1000:.1f}ms -> {final_phase*1000:.1f}ms")
    return final_phase, best_shift

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
        # MP3変換を廃止: M4A(AAC)をそのまま配信し、
        # librosaとブラウザ間のエンコーダ遅延不一致を排除
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

            if os.path.exists(filename):
                print(f"[DEBUG] yt-dlp done: {filename} ({os.path.getsize(filename)} bytes)")
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
            if ("Sign in to confirm you're not a bot" in msg
                or "confirm you're not a bot" in msg
                or "cookies" in msg
                or "This video is only available to Music Premium members" in msg
                or "Private video" in msg
                or "HTTP Error 403" in msg
                or "Forbidden" in msg):
                raise HTTPException(
                    status_code=403,
                    detail="YouTube Access Denied (Login/Cookies required). Please try a different video or upload cookies.txt."
                )
            raise e

def analyze_audio_file(file_path: str, progress_callback=None, offset_sec: float = 0.0, duration_limit_sec: float | None = None, forced_bpm: float | None = None, forced_phase: float | None = None, forced_last_chord: str | None = None, forced_run_length: int | None = None) -> dict:
    """Core analysis logic reusable for both uploads and URLs.

    Args:
        forced_phase: Optional phase offset in seconds to force (for multi-chunk consistency)
        forced_last_chord: Optional last chord from previous chunk (for stagnation continuity)
        forced_run_length: Optional run length from previous chunk (for stagnation continuity)
    """
    
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

        # 2. Beat tracking
        # オンセット検出（BPM検出・位相検出の両方で使用）
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames_detected = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units='frames'
        )
        onset_set = set(onset_frames_detected.tolist())
        total_frames = len(onset_env)
        num_onsets = len(onset_frames_detected)
        print(f"[DEBUG] Detected {num_onsets} onsets in {total_frames} frames")

        octave_factor = 1.0  # オクターブ補正時に更新される
        phase_detect_bpm = None  # 位相検出用のBPM（オクターブ補正前）

        if forced_bpm is not None:
            bpm = forced_bpm
            beat_frames = []
            print(f"[DEBUG] Using forced BPM: {bpm:.1f}")
        else:
            print("[DEBUG] Detecting BPM...")

            # BPM 60-240を1刻みでスキャンし、F1スコアが最大のBPMを選択
            best_bpm = 120.0
            best_score = -1.0
            tolerance = 3
            top_candidates = []

            for c in range(60, 241):
                beat_period = 60.0 * sr / (c * 512)
                grid = np.arange(0, total_frames, beat_period)
                if len(grid) == 0:
                    continue

                # Precision: ビートのうちオンセットが近くにある割合
                hits = 0
                for g in grid:
                    g_int = int(round(g))
                    for t in range(-tolerance, tolerance + 1):
                        if (g_int + t) in onset_set:
                            hits += 1
                            break
                precision = hits / len(grid)

                # Recall: オンセットのうちビートが近くにある割合
                grid_set = set()
                for g in grid:
                    g_int = int(round(g))
                    for t in range(-tolerance, tolerance + 1):
                        grid_set.add(g_int + t)
                onset_hits = sum(1 for o in onset_frames_detected if int(o) in grid_set)
                recall = onset_hits / max(1, num_onsets)

                # F_betaスコア (beta=0.8: Precisionを重視)
                # 高Precision = ビート位置にオンセットあり（正しいBPM）
                # 低Recall = ビート間にオンセット（細分音符、正常）
                BEAT_F_BETA = 0.8
                beta_sq = BEAT_F_BETA ** 2  # 0.64
                if precision + recall > 0:
                    score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
                else:
                    score = 0.0

                top_candidates.append((c, score, precision, recall))
                if score > best_score:
                    best_score = score
                    best_bpm = float(c)

            # 上位5候補をログ出力
            top_candidates.sort(key=lambda x: x[1], reverse=True)
            for c, s, p, r in top_candidates[:5]:
                print(f"[DEBUG] BPM candidate: {c} (P={p:.3f} R={r:.3f} Fb={s:.3f})")

            # Stage 2: 自己相関ベースのBPMリファインメント
            coarse_bpm = best_bpm
            coarse_f1 = best_score

            # 高解像度オンセットエンベロープで自己相関を計算（hop=128で4倍精密）
            hop_ac = 128
            onset_env_fine = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_ac)
            ac = librosa.autocorrelate(onset_env_fine)
            if ac[0] > 0:
                ac = ac / ac[0]

            # 粗BPMを期待ラグに変換
            expected_lag = 60.0 * sr / (coarse_bpm * hop_ac)
            expected_lag_int = int(round(expected_lag))

            # 期待ラグの±5%範囲で自己相関ピークを探索
            search_radius = max(3, int(expected_lag * 0.05))
            search_lo = max(1, expected_lag_int - search_radius)
            search_hi = min(len(ac) - 2, expected_lag_int + search_radius)

            if search_hi > search_lo:
                peak_idx = search_lo + int(np.argmax(ac[search_lo:search_hi + 1]))

                # 放物線補間でサブフレーム精度を得る
                if 0 < peak_idx < len(ac) - 1:
                    alpha = float(ac[peak_idx - 1])
                    beta = float(ac[peak_idx])
                    gamma = float(ac[peak_idx + 1])
                    denom = alpha - 2.0 * beta + gamma
                    delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-10 else 0.0
                    refined_lag = peak_idx + delta
                else:
                    refined_lag = float(peak_idx)

                refined_bpm = 60.0 * sr / (refined_lag * hop_ac) if refined_lag > 0 else coarse_bpm

                # サニティチェック
                ac_confidence = float(ac[peak_idx])
                if abs(refined_bpm - coarse_bpm) > 5.0:
                    # 粗BPMから離れすぎ → 棄却
                    bpm = coarse_bpm
                    print(f"[DEBUG] AC peak too far ({refined_bpm:.1f}), keeping coarse {coarse_bpm:.0f}")
                elif abs(refined_bpm - coarse_bpm) < 1.0:
                    # 1BPM未満の差 → ACリファイン結果を採用（累積ドリフト防止）
                    bpm = round(refined_bpm, 2)
                    print(f"[DEBUG] AC refined={refined_bpm:.2f} (delta<1), using refined {bpm:.2f} "
                          f"(lag={refined_lag:.2f}, ac={ac_confidence:.3f})")
                else:
                    # 1-5 BPMの差 → ACリファイン結果を採用
                    bpm = round(refined_bpm, 2)
                    print(f"[DEBUG] BPM refined via autocorrelation: {coarse_bpm:.0f} -> {bpm:.2f} "
                          f"(lag={refined_lag:.2f}, ac={ac_confidence:.3f}, coarse_Fb={coarse_f1:.3f})")
            else:
                bpm = coarse_bpm
                print(f"[DEBUG] Audio too short for AC refinement, keeping coarse BPM: {coarse_bpm:.0f}")

            del onset_env_fine, ac
            beat_frames = []

            # Stage 2.5: オクターブ検証（倍速/半分速検出の補正）
            # 位相検出はオクターブ補正前のBPMで行うため、事前に保存
            phase_detect_bpm = bpm
            if bpm is not None:
                corrected_bpm, octave_factor = verify_tempo_octave(y, sr, bpm, onset_env)
                if octave_factor != 1.0:
                    print(f"[OctaveCorrection] {bpm:.1f} → {corrected_bpm:.1f} BPM (×{octave_factor})")
                    bpm = corrected_bpm

            # Stage 2.7: バス自己相関によるテンポ補正
            # 全帯域オンセットはハイハット等で高速テンポを検出しやすい。
            # バスドラム (20-200Hz) の自己相関ピークが真のテンポを示す場合がある。
            if bpm is not None and bpm > 200:
                y_bass_temp = lowpass_filter(y, sr, cutoff_hz=200)
                y_bass_temp = highpass_filter(y_bass_temp, sr, cutoff_hz=20)
                bass_env_temp = librosa.onset.onset_strength(
                    y=y_bass_temp, sr=sr, hop_length=512)
                del y_bass_temp
                ac_bass = librosa.autocorrelate(bass_env_temp)
                del bass_env_temp
                if ac_bass[0] > 0:
                    ac_bass = ac_bass / ac_bass[0]

                best_bass_bpm = bpm
                best_bass_val = 0.0
                for candidate in range(60, 201, 2):
                    lag = 60.0 * sr / (candidate * 512)
                    lag_int = int(round(lag))
                    if 0 < lag_int < len(ac_bass) - 1:
                        val = float(
                            (ac_bass[lag_int - 1] + ac_bass[lag_int]
                             + ac_bass[lag_int + 1]) / 3.0)
                        if val > best_bass_val:
                            best_bass_val = val
                            best_bass_bpm = float(candidate)

                # 検出BPMのバスAC値を取得
                det_lag = 60.0 * sr / (bpm * 512)
                det_lag_int = int(round(det_lag))
                det_bass_val = 0.0
                if 0 < det_lag_int < len(ac_bass) - 1:
                    det_bass_val = float(
                        (ac_bass[det_lag_int - 1] + ac_bass[det_lag_int]
                         + ac_bass[det_lag_int + 1]) / 3.0)

                del ac_bass

                if abs(best_bass_bpm - bpm) > 10:
                    prior_bass = evaluate_tempo_prior(best_bass_bpm)
                    prior_det = evaluate_tempo_prior(bpm)
                    score_bass = best_bass_val * 0.6 + prior_bass * 0.4
                    score_det = det_bass_val * 0.6 + prior_det * 0.4
                    print(f"[BassTempoCheck] Bass peak: {best_bass_bpm:.0f} BPM "
                          f"(ac={best_bass_val:.3f}, prior={prior_bass:.1f}, "
                          f"score={score_bass:.3f})")
                    print(f"[BassTempoCheck] Detected: {bpm:.1f} BPM "
                          f"(ac={det_bass_val:.3f}, prior={prior_det:.1f}, "
                          f"score={score_det:.3f})")
                    if score_bass > score_det * 1.05:
                        print(f"[BassTempoCorrection] {bpm:.1f} → "
                              f"{best_bass_bpm:.0f} BPM")
                        bpm = best_bass_bpm
                        octave_factor = 1.0
                    else:
                        print(f"[BassTempoCheck] Keeping {bpm:.1f} BPM "
                              f"(bass advantage insufficient)")
                else:
                    print(f"[BassTempoCheck] Bass peak {best_bass_bpm:.0f} BPM "
                          f"close to detected {bpm:.1f}, no correction needed")

        # Stage 3: ビート位相検出 - 最適なグリッド開始位置を探索
        if forced_phase is not None:
            # チャンク統一のため、位相を強制使用
            if offset_sec > 0:
                # チャンクオフセットを考慮してローカル位相を計算
                # グローバルビートグリッド: forced_phase + n * seg_dur
                # このチャンク内の最初のビート位置を求める
                beat_dur = 60.0 / bpm
                seg_dur = beat_dur * 2  # 2 beats per beat_times entry
                elapsed = offset_sec - forced_phase
                if elapsed > 0:
                    remainder = elapsed % seg_dur
                    phase_offset_sec = (seg_dur - remainder) if remainder > 1e-6 else 0.0
                else:
                    phase_offset_sec = forced_phase - offset_sec
                print(f"[DEBUG] Beat phase offset: {phase_offset_sec*1000:.1f}ms "
                      f"(local phase for chunk at offset={offset_sec:.1f}s, "
                      f"global phase={forced_phase*1000:.1f}ms)")
            else:
                phase_offset_sec = forced_phase
                print(f"[DEBUG] Beat phase offset: {phase_offset_sec*1000:.1f}ms (forced from chunk 0)")
            del onset_set
            # onset_env は後述の共通 del で解放
            _progress(35)
        else:
            # 通常の位相検出(最初のチャンクのみ)
            # オクターブ補正前のBPMで位相を検出（補正後だと両グリッドで
            # 同精度になり正しいビートを区別できないため）
            _phase_bpm = phase_detect_bpm if octave_factor != 1.0 else bpm
            beat_period_onset = 60.0 * sr / (_phase_bpm * 512)
            best_phase = 0.0
            best_phase_score = -1.0
            phase_tolerance = 3

            for phase_10x in range(0, int(beat_period_onset * 10)):
                phase = phase_10x / 10.0
                grid = np.arange(phase, total_frames, beat_period_onset)
                if len(grid) == 0:
                    continue
                hits = 0
                for g in grid:
                    g_int = int(round(g))
                    for t in range(-phase_tolerance, phase_tolerance + 1):
                        if (g_int + t) in onset_set:
                            hits += 1
                            break
                precision = hits / len(grid)
                if precision > best_phase_score:
                    best_phase_score = precision
                    best_phase = phase

            phase_offset_sec = best_phase * 512 / sr
            print(f"[DEBUG] Beat phase offset: {phase_offset_sec*1000:.1f}ms (detected at {_phase_bpm:.1f}BPM, precision={best_phase_score:.4f})")
            del onset_set
            _progress(35)

        # Stage 3.5: 小節頭位相リファインメント
        # ビート位相から小節頭（ダウンビート）を特定
        if forced_phase is None:
            refined_phase, bar_shift = refine_bar_phase(y, sr, bpm, phase_offset_sec,
                                                        octave_factor=octave_factor)
            if bar_shift != 0.0:
                print(f"[BarPhaseRefine] Applied: shift={bar_shift} beats, "
                      f"{phase_offset_sec*1000:.1f}ms -> {refined_phase*1000:.1f}ms")
            phase_offset_sec = refined_phase

        # 3. Chroma
        print("[DEBUG] Computing chroma (HPSS + n_fft=4096)...")
        hop_length = 2048  # 4096→2048: 時間解像度2倍（~93ms@22050Hz）
        chroma = compute_chroma_log(y, sr, hop_length=hop_length)
        print(f"mem after chroma: {mem_mb():.1f} MB")

        bass_chroma = compute_bass_chroma(y, sr, hop_length=hop_length)
        print(f"mem after bass chroma: {mem_mb():.1f} MB")
        print(f"[DEBUG] Chroma shape: {chroma.shape}, Bass chroma shape: {bass_chroma.shape}")
        _progress(60) # Chroma done

        if chroma.shape[1] == 0:
            raise ValueError("Chroma extraction failed or audio too short")

        # 4. Time axes
        # Improved: Use frame-based timing for more precise segment boundaries
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

        # Calculate beat duration for diagnostics
        beat_duration = 60.0 / bpm
        target_segment_duration = beat_duration * 2
        total_duration = librosa.frames_to_time(chroma.shape[1], sr=sr, hop_length=hop_length)

        # --- Adaptive Beat Tracking ---
        # 固定BPMグリッドではテンポ揺らぎに追従できず累積ドリフトが発生する。
        # librosa.beat.beat_track で実際のビート位置を検出し、セグメント境界に使用。
        # forced_phase（後続チャンク）では安定性のため固定グリッドを維持。
        if forced_phase is not None:
            # 後続チャンク: 固定グリッドで一貫性を保つ
            beat_times = np.arange(phase_offset_sec, total_duration + target_segment_duration, target_segment_duration)
            beats_per_seg = 2
            print(f"[DEBUG] Using fixed grid (forced_phase): seg_dur={target_segment_duration:.3f}s")
        else:
            # 初回チャンク: adaptive beat tracking
            try:
                bt_hop = 512
                # onset_envは既存のものを再利用（BPM検出で使用済み、delete済みなら再計算）
                try:
                    _ = onset_env
                except NameError:
                    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=bt_hop)

                _, bt_frames = librosa.beat.beat_track(
                    onset_envelope=onset_env, sr=sr, hop_length=bt_hop,
                    bpm=bpm, trim=False
                )
                bt_times = librosa.frames_to_time(bt_frames, sr=sr, hop_length=bt_hop)

                if len(bt_times) >= 4:
                    # ビートトラッカー成功: 実際のビート位置を使用
                    # beats_per_segment=4 で4ビート(1小節)ずつグループ化
                    beat_times = bt_times
                    beats_per_seg = 4
                    avg_beat_interval = float(np.median(np.diff(bt_times)))
                    print(f"[DEBUG] Adaptive beat tracking: {len(bt_times)} beats, "
                          f"median interval={avg_beat_interval:.3f}s "
                          f"(≈{60.0/avg_beat_interval:.1f} BPM)")
                else:
                    # ビートが少なすぎる → 固定グリッドにフォールバック
                    beat_times = np.arange(phase_offset_sec, total_duration + target_segment_duration, target_segment_duration)
                    beats_per_seg = 2
                    print(f"[DEBUG] Beat tracker returned too few beats ({len(bt_times)}), using fixed grid")
            except Exception as e:
                # ビートトラッカーエラー → 固定グリッドにフォールバック
                beat_times = np.arange(phase_offset_sec, total_duration + target_segment_duration, target_segment_duration)
                beats_per_seg = 2
                print(f"[DEBUG] Beat tracker failed ({e}), using fixed grid")

        del onset_env  # メモリ解放

        num_segments = max(1, len(beat_times) - 1)
        print(f"[DEBUG] BPM: {bpm:.2f}, Beat duration: {beat_duration:.3f}s, Segments: ~{num_segments}")

        # 5. Aggregate per segment
        print("[DEBUG] Aggregating segments...")
        main_matrix, segments = aggregate_chroma_per_segment(chroma, times, beat_times, beats_per_segment=beats_per_seg)
        bass_matrix, _ = aggregate_chroma_per_segment(bass_chroma, times, beat_times, beats_per_segment=beats_per_seg)
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
        raw_chords, final_last_chord, final_run_length = detect_chords_matrix(
            main_matrix,
            bass_matrix,
            penalty_mask=penalty_mask,
            penalty_value=0.20,   # 0.15→0.20: 非diatonicコードへのペナルティ強化
            main_weight=0.6,      # 0.7→0.6: ベースクロマ復活に合わせて調整
            bass_weight=0.35,     # 0.4→0.35: ベース過剰強調を緩和
            forced_last_chord=forced_last_chord,
            forced_run_length=forced_run_length,
        )
        print(f"[DEBUG] Raw chords detected: {len(raw_chords)}")
        _progress(90) # Detection done

        # Use stagnation-aware smoothing to prevent creating long runs
        smoothed_chords = smooth_chord_sequence_stagnation_aware(raw_chords, passes=2, max_run=6)

        # Additional safety net: break any remaining long runs
        smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=6)

        print(f"[DEBUG] unique chords: {len(set(smoothed_chords))}")
        print(f"[DEBUG] first 20 chords: {smoothed_chords[:20]}")

        bars = []
        for i, chord_name in enumerate(smoothed_chords):
            # Granular progress for final loop (90 -> 99)
            if len(smoothed_chords) > 0:
                 progress_percent = 90 + 9 * ((i + 1) / len(smoothed_chords))
                 _progress(progress_percent)

            # Retrieve precise segment timing
            start_sec = 0.0
            end_sec = 0.0
            if i < len(segments):
                start_sec, end_sec = segments[i]
            else:
                # Fallback implementation if alignment is off (though they should align)
                # Typically implies end-of-stream edge case
                # Estimate based on previous segment or fixed grid
                if i > 0:
                     prev_end = bars[-1]["end_sec"]
                     avg_dur = (prev_end / i) if i > 0 else 2.0
                     start_sec = prev_end
                     end_sec = start_sec + avg_dur
                else:
                     start_sec = 0.0
                     end_sec = 2.0 # Fallback

            tab = chord_to_tab(chord_name)
            bars.append({
                "bar": i + 1,
                "chord": chord_name,
                "tab": {
                    "frets": tab
                } if tab else None,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec)
            })
        
        print(f"[DEBUG] Analysis complete. Returning {len(bars)} bars.")
        _progress(99)
        return {
            "bpm": bpm,
            "duration_sec": round(duration_sec, 1),
            "time_signature": "2/4",
            "key": estimated_key,
            "bars": bars,
            "phase_offset_sec": round(phase_offset_sec, 4),
            "final_last_chord": final_last_chord,
            "final_run_length": final_run_length,
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

def run_analysis_bg(job_id: str, file_path: str, mode: AnalyzeMode = AnalyzeMode.PREVIEW, source: str = "upload"):
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
        elif mode == AnalyzeMode.EARLY_ACCESS:
             MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "300"))
             print(f"[INFO] Mode: EARLY_ACCESS -> Max duration {MAX_ANALYSIS_SEC}s")
        else:  # FULL
             MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "600"))
             print(f"[INFO] Mode: FULL -> Max duration {MAX_ANALYSIS_SEC}s")

        # 2. Usage Check (Early Access)
        if mode == AnalyzeMode.EARLY_ACCESS:
            # TODO: song-hash based deduplication
            # Current: Simple 2-use limit placeholder (Logic would go here or at endpoint level)
            pass
            
        CHUNK_SEC = float(os.getenv("CHUNK_SEC", "30"))

        # 3) Remove initial get_duration to avoid "decode stall"
        # We process until MAX_ANALYSIS_SEC or EOF

        bpm = None  # 最初のチャンクで検出されたBPMを使用
        forced_phase = None  # 最初のチャンクで検出された位相を使用(チャンク統一)
        segment_duration = None  # BPM検出後に計算

        all_bars: list[dict] = []
        key_votes: list[str] = []
        stag_last_chord: str | None = None
        stag_run_length: int | None = None

        chunk_idx = 0
        offset = 0.0
        
        # Estimate total chunks for progress calculation (assuming MAX)
        # This is an approximation since we might stop early, but ensures 0-100 scale
        estimated_total_chunks = int(math.ceil(MAX_ANALYSIS_SEC / CHUNK_SEC))

        FIRST_CHUNK_SEC = 60.0  # 初回チャンクは60秒（BPM検出精度のため）

        while offset < MAX_ANALYSIS_SEC:
            if chunk_idx == 0:
                dur = min(FIRST_CHUNK_SEC, MAX_ANALYSIS_SEC - offset)
            else:
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
                duration_limit_sec=dur,
                forced_bpm=bpm,  # 最初のチャンク以降はBPMを統一
                forced_phase=forced_phase,  # 最初のチャンク以降は位相を統一
                forced_last_chord=stag_last_chord,
                forced_run_length=stag_run_length,
            )

            # Check for effective end of file (short read)
            actual_dur = raw["duration_sec"]

            chunk_bars = raw["bars"]
            key_votes.append(raw.get("key", "Unknown"))
            stag_last_chord = raw.get("final_last_chord")
            stag_run_length = raw.get("final_run_length")

            # 最初のチャンクからBPMと位相を取得
            if bpm is None:
                bpm = raw.get("bpm", 120.0)
                forced_phase = raw.get("phase_offset_sec", 0.0)  # 位相も保存
                seconds_per_beat = 60.0 / bpm
                segment_duration = seconds_per_beat * 2
                print(f"[ChunkMerge] Using detected BPM: {bpm:.1f}, phase: {forced_phase*1000:.1f}ms, segment_duration: {segment_duration:.3f}s")

            for i, bar in enumerate(chunk_bars):
                # チャンクのbar配列からstart_sec/end_secを直接取得（オフセット加算）
                bar_start = bar.get("start_sec")
                bar_end = bar.get("end_sec")

                if bar_start is not None and bar_end is not None:
                    abs_start = offset + float(bar_start)
                    abs_end = offset + float(bar_end)
                else:
                    # フォールバック: segment_durationベースで計算
                    abs_start = offset + i * segment_duration
                    abs_end = offset + (i + 1) * segment_duration

                # チャンク境界を超えた小節は除外
                chunk_end_abs = offset + actual_dur
                if abs_start >= chunk_end_abs:
                    break
                if abs_end > chunk_end_abs:
                    abs_end = chunk_end_abs

                bar_obj = {
                    "bar": len(all_bars) + 1,
                    "chord": bar["chord"],
                    "tab": bar.get("tab"),
                    "start_sec": abs_start,
                    "end_sec": abs_end,
                }
                all_bars.append(bar_obj)

            # Check exit conditions
            # If we got significantly less audio than requested, we hit EOF
            if actual_dur < (dur - 0.5) or actual_dur <= 0.1:
                offset += actual_dur
                break
            
            offset += dur
            chunk_idx += 1


        # key matches majority vote across all chunks
        if key_votes:
            key_counter = Counter(key_votes)
            key = key_counter.most_common(1)[0][0]
        else:
            key = "Unknown"
        # bpmがNoneのままの場合（チャンク0個）はフォールバック
        if bpm is None:
            bpm = 120.0

        # barにstart_secが既にある場合はadd_bar_timingをスキップ
        if all_bars and "start_sec" not in all_bars[0]:
            all_bars = add_bar_timing(
                all_bars,
                bpm=bpm,
                time_signature="2/4",
                analyzed_duration_sec=offset
            )

        # チャンク結合後のタイミング:
        # forced_phase によりチャンク間タイミングは連続（ギャップ < 1ms）
        # 統一グリッド上書きは廃止: BPM誤差の蓄積ドリフトを防止し、
        # per-chunkの精密なchromaフレーム境界タイミングを保持する

        # 診断: バー間隔を確認
        if len(all_bars) >= 2:
            _diag_dur = round(all_bars[1]["start_sec"] - all_bars[0]["start_sec"], 4)
            _expected = round((60.0 / bpm) * 2, 4)  # 2拍/セグメント × beats_per_segment=2 = 4拍
            print(f"[ChunkMerge] Bar duration: {_diag_dur}s (per-chunk timing, expected ~{_expected}s)")
            print(f"[ChunkMerge] Total bars: {len(all_bars)}, first={all_bars[0]['start_sec']:.4f}s, last_end={all_bars[-1]['end_sec']:.4f}s")

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
            "bars": all_bars, # Return bars even in Preview (limited by duration cap)
            "_build": "build-v5.5.1",
        }
        if source == "url" and os.path.exists(file_path):
            final_result["audio_url"] = "/temp/" + os.path.basename(file_path)

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
            if source == "upload" and os.path.exists(file_path):
                os.remove(file_path)
            # source == "url" の場合はファイルを保持（cleanup_temp_dir の6時間TTLで自動削除）
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
    print(f"[EP] entered /analyze/preview {file.filename} {file.content_type}")
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

    # Use to_thread to prevent blocking event loop during file save
    # This solves the "pending" response issue for large uploads
    await anyio.to_thread.run_sync(_save_upload_sync, file.file, file_path)

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
        threading.Thread(target=run_analysis_bg, args=(job_id, file_path, mode, "url")).start()

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

        

