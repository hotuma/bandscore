import os
import anyio
import sys
import logging

# CRITICAL: Make pkg_resources available before any audio library imports
# This fixes "No module named 'pkg_resources'" error from audioread/librosa
try:
    # Try direct import first
    import pkg_resources
except ImportError:
    try:
        # Fallback: setuptools might not expose pkg_resources at top level
        import setuptools
        from setuptools._distutils.dist import Distribution
        # Force setuptools to expose pkg_resources
        sys.modules['pkg_resources'] = setuptools.pkg_resources
    except (ImportError, AttributeError):
        try:
            # Last resort: use importlib.metadata as replacement
            from importlib.metadata import distributions, version, PackageNotFoundError
            # Create a minimal pkg_resources shim
            class _PkgResourcesShim:
                @staticmethod
                def get_distribution(name):
                    try:
                        from importlib.metadata import distribution
                        d = distribution(name)
                        return type('DistInfo', (), {
                            'version': d.version,
                            'project_name': d.metadata['Name'],
                            'location': d.locate_file(''),
                        })()
                    except PackageNotFoundError:
                        raise
            sys.modules['pkg_resources'] = _PkgResourcesShim()
        except Exception as e:
            sys.stderr.write(f"Warning: Could not setup pkg_resources shim: {e}\n")

import psutil

# 無効化stdoutバッファリング
sys.stdout.reconfigure(line_buffering=True)

# ルートロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend_app.log')
    ]
)

# FastAPIおよび関連モジュールのログレベルを設定
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.DEBUG)

# numbaのログレベルをWARNINGに設定（bytecode dumpを抑制）
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

# アプリケーションログ用の設定
app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.DEBUG)

# Force librosa to use soundfile backend instead of audioread (avoids pkg_resources issue)
os.environ["LIBROSA_BACKEND"] = "soundfile"
# Also try alternative methods
os.environ["AUDIODIR"] = "soundfile"
app_logger.info("Set LIBROSA_BACKEND=soundfile to avoid pkg_resources dependency")

# 明示的にハンドラーを追加（絶対パス）
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend_app.log')
if not app_logger.handlers:
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)

app_logger.info("Backend starting...")

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import librosa
import numpy as np

# Explicitly set librosa audio backend to soundfile (avoids pkg_resources dependency)
try:
    librosa.set_audio_backend('soundfile')
    app_logger.info("Set librosa audio backend to soundfile")
except Exception as e:
    app_logger.warning(f"Could not set librosa audio backend: {e}")

# madmom for more accurate BPM detection (currently not installed - commented out)
# from madmom.features.beats import BeatDetection, BeatTrackingProcessor
# from madmom.audio.signal import Signal
import tempfile
import os
import shutil
import math

# リクエストサイズ制限を緩和
from starlette.datastructures import UploadFile as StarletteUploadFile
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

# Simple test endpoint
@app.get("/test")
def test_endpoint():
    print("[PRINT-TEST] Test endpoint called!", flush=True)
    sys.stderr.write("[STDERR-TEST] Test endpoint called!\n")
    sys.stderr.flush()
    app_logger.info("[LOGGER-TEST] Test endpoint called!")
    return {"status": "ok", "message": "Test endpoint is working"}

# Add CORS logging middleware (BEFORE CORS middleware to see all requests)
@app.middleware("http")
async def cors_debug_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    print(f"[CORS-DEBUG] Method={request.method}, Path={request.url.path}, Origin={origin}")
    app_logger.info(f"[CORS-DEBUG] Method={request.method}, Path={request.url.path}, Origin={origin}")
    response = await call_next(request)
    return response

# Add logging middleware
@app.middleware("http")
async def log_req_lifecycle(request: Request, call_next):
    t0 = time.time()

    # ログファイルに直接書き込む
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend_app.log')
    with open(log_file_path, 'a') as f:
        f.write(f"[DIRECT-LOG] {request.method} {request.url.path}\n")
        f.flush()

    app_logger.info(f"[REQ-START] {request.method} {request.url.path}")
    try:
        resp = await call_next(request)
        return resp
    except Exception as e:
        app_logger.error(f"[REQ-ERROR] {request.method} {request.url.path} - {type(e).__name__}: {str(e)}")
        import traceback
        app_logger.error(traceback.format_exc())
        raise
    finally:
        dt = (time.time() - t0) * 1000
        app_logger.info(f"[REQ-END]   {request.method} {request.url.path} {dt:.1f}ms")

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
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
        "http://localhost:3005",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003",
        "http://127.0.0.1:3004",
        "http://127.0.0.1:3005",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://bandscore.vercel.app",
        "https://bandscore.onrender.com",
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
    stale_after_sec = float(os.getenv("STALE_JOB_SEC", "300"))
    expired = [
        jid
        for jid, j in jobs.items()
        if j.get("expires_at", 0) < now
        or (
            j.get("status") == "analyzing"
            and now - float(j.get("updated_at") or j.get("started_at") or 0) > stale_after_sec
        )
    ]
    for jid in expired:
        app_logger.warning(f"[JobCleanup] Removing expired/stale job {jid}: {jobs.get(jid)}")
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

# --- MADMOM BPM Detection ---
# def detect_bpm_madmom(y, sr):
#     """
#     madmomによる正確な BPM 検出 (currently not installed - commented out)
#
#
#     Args:
#         y: オーディオ信号
#         sr: サンプリングレート (通常 22050 Hz)
#
#     Returns:
#         tuple: (bpm, beat_times) - 検出された BPM とビートタイムスタンプ配列
#     """
#     try:
#         print("[MADMOM] Loading RNN model for onset detection...")
#         sig = Signal(y, sr)
#
#         # RNNベースの onset 検出
#         onsets = BeatDetection()(sig)
#         print(f"[MADMOM] RNN onset detection complete: {len(onsets)} onsets")
#         # HMMベースのビートトラッキング
#         beats = BeatTrackingProcessor()(onsets)
#         print(f"[MADMOM] HMM beat tracking complete: {len(beats)} beats")
#         # BPM を計算
#         if len(beats.times) > 1:
#             beat_intervals = np.diff(beats.times)
#             median_interval = np.median(beat_intervals)
#             bpm = 60.0 / median_interval
#             print(f"[MADMOM] Median beat interval: {median_interval:.3f}s, Calculated BPM: {bpm:.1f}")
#         else:
#             bpm = 120.0  # フォールバック
#             print("[MADMOM] Not enough beats detected, using fallback BPM: 120")
#
#         return bpm, beats.times.tolist() if len(beats.times) > 0 else []
#
#     except ImportError as e:
#         print(f"[ERROR] madmom not installed: {e}")
#         print("[ERROR] Falling back to librosa-based BPM detection")
#         return None, None
#     except Exception as e:
#         print(f"[ERROR] madmom BPM detection failed: {e}")
#         print("[ERROR] Falling back to librosa-based BPM detection")
#         return None, None

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

def compute_time_signature(beats_per_segment: int) -> str:
    """
    Compute time signature from beats per segment.

    Args:
        beats_per_segment: Number of beats per analysis segment (2 or 4)

    Returns:
        Time signature string (e.g., "2/4" for 2 beats, "4/4" for 4 beats)
    """
    if beats_per_segment <= 0:
        return "4/4"  # Safe default
    return f"{beats_per_segment}/4"

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

def compute_chroma_cqt(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """
    Compute CQT-based chroma features with HPSS.
    CQT provides logarithmic frequency resolution — better for guitar low strings
    (E2=82Hz, A2=110Hz) than STFT which has fixed linear frequency bins.
    bins_per_octave=36: 3x oversampling for smoother pitch estimation.
    Returns: (12, T)
    """
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=36,
        norm=None,  # Manual log + L1 normalization below
    )
    del y_harmonic
    chroma_log = np.log1p(10.0 * chroma)
    return chroma_log / (np.sum(chroma_log, axis=0, keepdims=True) + 1e-8)

def compute_chroma_stft_light(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """Low-memory chroma path for constrained hosts."""
    chroma = librosa.feature.chroma_stft(
        y=np.asarray(y, dtype=np.float32),
        sr=sr,
        hop_length=hop_length,
        n_fft=2048,
    ).astype(np.float32, copy=False)
    chroma_log = np.log1p(np.float32(10.0) * chroma)
    chroma_sum = np.sum(chroma_log, axis=0, keepdims=True) + np.float32(1e-8)
    return (chroma_log / chroma_sum).astype(np.float32, copy=False)

def apply_chroma_contrast(chroma: np.ndarray, filter_size: int = 50, blend: float = 0.25) -> np.ndarray:
    """
    ローリング最小値を減算してペダルトーン（通奏低音）を除去する。
    filter_size=50: 50フレーム × 2048/22050 ≈ 4.6秒のウィンドウ
    各ピッチクラスで持続する背景エネルギーを除いて、コード変化を際立たせる。
    blend: 元のchromaをどれだけ残すか (0.0=完全除去, 1.0=除去なし)。
           完全除去するとフレームが均一になり情報量が失われるため、blend分だけ元信号を保持する。
    Input/Output: (12, T) normalized chroma
    """
    from scipy.ndimage import minimum_filter1d
    chroma_bg = minimum_filter1d(chroma, size=filter_size, axis=1, mode='reflect')
    chroma_fg = np.maximum(0.0, chroma - chroma_bg)
    # ブレンド: 完全除去ではなく元信号を blend 分残してコントラストと情報量のバランスを取る
    chroma_out = chroma_fg * (1.0 - blend) + chroma * blend
    chroma_sum = np.sum(chroma_out, axis=0, keepdims=True)
    # chroma_sumが0のフレームはゼロ除算を避けるため元のchromaをそのまま返す
    return np.where(chroma_sum > 1e-8, chroma_out / chroma_sum, chroma)

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

def compute_bass_chroma_light(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """Low-memory bass chroma path using a smaller STFT."""
    y_bass = bandpass_filter(np.asarray(y, dtype=np.float32), sr, low_hz=60, high_hz=300)
    chroma = librosa.feature.chroma_stft(
        y=y_bass,
        sr=sr,
        hop_length=hop_length,
        n_fft=2048,
    ).astype(np.float32, copy=False)
    del y_bass
    chroma_log = np.log1p(np.float32(10.0) * chroma)
    chroma_sum = np.sum(chroma_log, axis=0, keepdims=True) + np.float32(1e-8)
    return (chroma_log / chroma_sum).astype(np.float32, copy=False)

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

# --- HMM: Transition Matrix & Viterbi ---

def build_transition_matrix(chord_labels: list) -> np.ndarray:
    """
    Build chord-to-chord transition log-probability matrix from music theory.

    Design: Non-self weights are normalized to sum to (1-SELF_PROB), then
    the diagonal is filled with SELF_PROB — guaranteeing exact self-transition
    probability regardless of the number of chords or their weights.

    Returns: (n, n) log-probability matrix, log_trans[i, j] = log P(j|i)
    """
    SELF_PROB    = 0.50   # 0.65→0.50: switching cost を緩和してコード変化を起きやすく
    W_5TH_UP    = 5.0   # dominant (C→G, Am→Em)
    W_5TH_DOWN  = 4.0   # subdominant (C→F, Am→Dm)
    W_RELATIVE  = 5.0   # relative major/minor (C↔Am)
    BASE_WEIGHT = 1.0   # all other chords

    n = len(chord_labels)

    def _is_minor(lbl: str) -> bool:
        for note in sorted(NOTE_NAMES, key=len, reverse=True):
            if lbl.startswith(note):
                suffix = lbl[len(note):]
                return suffix == "m" or suffix == "m7"
        return False

    # Build unnormalized non-self weight matrix (diagonal stays 0)
    non_self = np.full((n, n), BASE_WEIGHT, dtype=np.float64)
    np.fill_diagonal(non_self, 0.0)

    for i, ci in enumerate(chord_labels):
        root_i = chord_root_index(ci)
        minor_i = _is_minor(ci)
        for j, cj in enumerate(chord_labels):
            if i == j:
                continue
            root_j = chord_root_index(cj)
            minor_j = _is_minor(cj)
            interval = (root_j - root_i) % 12

            if interval == 7:
                non_self[i, j] += W_5TH_UP
            elif interval == 5:
                non_self[i, j] += W_5TH_DOWN
            if minor_i and not minor_j and interval == 3:
                non_self[i, j] += W_RELATIVE
            elif not minor_i and minor_j and interval == 9:
                non_self[i, j] += W_RELATIVE

    # Scale each row's non-self weights to sum to (1 - SELF_PROB)
    row_sums = non_self.sum(axis=1, keepdims=True) + 1e-12
    trans = non_self / row_sums * (1.0 - SELF_PROB)
    np.fill_diagonal(trans, SELF_PROB)

    return np.log(trans + 1e-12)


def viterbi_decode(
    emission_log_probs: np.ndarray,
    log_trans: np.ndarray,
    log_init: np.ndarray,
) -> np.ndarray:
    """
    Viterbi decoding in log-probability space.

    Args:
        emission_log_probs: (T, n) log-emission probabilities
        log_trans: (n, n) log-transition matrix, log_trans[i, j] = log P(j|i)
        log_init: (n,) log-initial state probabilities

    Returns:
        path: (T,) int array of most likely chord indices
    """
    T, n = emission_log_probs.shape
    dp = np.full((T, n), -np.inf, dtype=np.float64)
    bp = np.zeros((T, n), dtype=np.int32)

    dp[0] = log_init + emission_log_probs[0]

    for t in range(1, T):
        # trans_scores[i, j] = dp[t-1, i] + log_trans[i, j]
        trans_scores = dp[t - 1, :, np.newaxis] + log_trans  # (n, n)
        bp[t] = np.argmax(trans_scores, axis=0)               # (n,)
        dp[t] = trans_scores[bp[t], np.arange(n)] + emission_log_probs[t]

    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = np.argmax(dp[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    return path


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
    min_hold_segments: int = 1,
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


def detect_chords_hmm(
    main_matrix: np.ndarray,
    bass_matrix: np.ndarray,
    penalty_mask: Optional[np.ndarray] = None,
    penalty_value: float = 0.20,
    main_weight: float = 0.6,
    bass_weight: float = 0.35,
    temperature: float = 8.0,
    # Accepted for drop-in compatibility but unused (HMM handles sequence globally)
    _forced_last_chord: Optional[str] = None,
    _forced_run_length: Optional[int] = None,
    **kwargs,
) -> tuple:
    """
    HMM-based chord detection using Viterbi decoding.

    Drop-in replacement for detect_chords_matrix() with the same return type:
        (list[str], str, int)

    Pipeline:
      1. Compute cosine similarity scores (same as detect_chords_matrix)
      2. Apply diatonic penalty before softmax
      3. Convert to emission log-probabilities via temperature-scaled softmax
      4. Viterbi decode using music-theory transition matrix
      5. Return chord list + cross-chunk continuity state
    """
    num_segments = main_matrix.shape[0]
    if num_segments == 0:
        return [], "", 0

    if bass_matrix.shape[0] != num_segments:
        min_segs = min(num_segments, bass_matrix.shape[0])
        main_matrix = main_matrix[:min_segs, :]
        bass_matrix = bass_matrix[:min_segs, :]
        num_segments = min_segs

    num_chords = TEMPLATE_MATRIX.shape[0]

    # 1. Raw scores: cosine similarity (C, S)
    main_scores = cosine_similarity_matrix(TEMPLATE_MATRIX, main_matrix.T)

    bass_scores = np.zeros((num_chords, num_segments), dtype=np.float64)
    for chord_idx, label in enumerate(CHORD_LABELS):
        root_idx = chord_root_index(label)
        bass_scores[chord_idx, :] = bass_matrix[:, root_idx]

    # Fix B: セグメント単位でmax正規化（グローバルmaxではなく各セグメント内の比率を保持）
    # グローバルmax正規化では全セグメントが同一スケールに圧縮され、
    # セグメント間の差異が消えてしまう問題を修正
    bass_max_per_seg = np.max(bass_scores, axis=0, keepdims=True)  # (1, S)
    bass_scores = np.where(bass_max_per_seg > 1e-8,
                           bass_scores / (bass_max_per_seg + 1e-8),
                           bass_scores)

    raw_scores = main_scores * main_weight + bass_scores * bass_weight  # (C, S)

    # 2. Diatonic penalty (applied before softmax to shift distribution)
    if penalty_mask is not None and penalty_mask.shape[0] == num_chords:
        raw_scores[penalty_mask, :] -= penalty_value

    # Fix E: emission前のスコア分散を診断ログ出力
    # セグメント毎のtop-1スコアとtop-2スコアの差（小さければ同一コード固着リスク高）
    raw_T = raw_scores.T  # (S, C)
    sorted_raw = np.sort(raw_T, axis=1)[:, ::-1]  # 降順ソート
    score_gap = sorted_raw[:, 0] - sorted_raw[:, 1]  # top1 - top2
    mean_gap = float(np.mean(score_gap))
    min_gap = float(np.min(score_gap))
    top1_chord_idx = int(np.argmax(raw_T[0]))
    uniform_count = int(np.sum(np.argmax(raw_T, axis=1) == top1_chord_idx))
    print(f"[HMM-Diag] score_gap mean={mean_gap:.4f} min={min_gap:.4f} "
          f"uniform={uniform_count}/{num_segments} segs share top chord '{CHORD_LABELS[top1_chord_idx]}'")

    # 3. Temperature-scaled softmax → emission log-probabilities (T, C)
    scores_T = raw_scores.T * temperature            # (S, C)
    scores_T -= scores_T.max(axis=1, keepdims=True)  # numerical stability
    log_sum_exp = np.log(np.sum(np.exp(scores_T), axis=1, keepdims=True) + 1e-12)
    emission_log_probs = scores_T - log_sum_exp      # (T, C)

    # Fix E: emission分散の診断
    emission_std = float(np.std(emission_log_probs))
    print(f"[HMM-Diag] emission_log_probs std={emission_std:.4f} "
          f"(low=uniform risk, temperature={temperature})")

    # 4. Viterbi decoding
    path = viterbi_decode(emission_log_probs, _HMM_LOG_TRANS, _HMM_LOG_INIT)

    # 5. Convert to chord names
    chord_sequence = [CHORD_LABELS[idx] for idx in path]

    # Fix C: Viterbi結果が全セグメントで均一な場合、greedy+stagnation-awareにフォールバック
    # 全均一はemission確率が全フレームで同一であることを意味し、
    # HMMの遷移行列が支配的になっているため信頼性が低い
    unique_in_result = len(set(chord_sequence))
    if unique_in_result <= 1 and num_segments > 4:
        print(f"[HMM-FALLBACK] Viterbi returned uniform result ({chord_sequence[0] if chord_sequence else '?'} × {num_segments}). "
              f"Falling back to greedy+stagnation-aware decoder.")
        fallback_result = detect_chords_matrix(
            main_matrix, bass_matrix,
            penalty_mask=penalty_mask,
            penalty_value=penalty_value,
            main_weight=main_weight,
            bass_weight=bass_weight,
            forced_last_chord=_forced_last_chord,
            forced_run_length=_forced_run_length,
        )
        return fallback_result

    # Cross-chunk continuity state
    final_last_chord = chord_sequence[-1] if chord_sequence else ""
    final_run_length = 1
    for i in range(len(chord_sequence) - 2, -1, -1):
        if chord_sequence[i] == final_last_chord:
            final_run_length += 1
        else:
            break

    return chord_sequence, final_last_chord, final_run_length


# HMM parameters (computed once at module load, reused for every request)
_HMM_LOG_TRANS: np.ndarray = build_transition_matrix(CHORD_LABELS)
_HMM_LOG_INIT: np.ndarray = np.full(
    len(CHORD_LABELS), np.log(1.0 / len(CHORD_LABELS)), dtype=np.float64
)


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

def break_long_stagnation_runs(
    chords: list[str],
    max_consecutive: int = 6,
    diatonic_chords: Optional[list[str]] = None,
) -> list[str]:
    """
    Break up any remaining long stagnation runs after detection and smoothing.

    If a chord runs for more than max_consecutive bars, attempt to split it
    by inserting alternative chords from surrounding context.

    This is a safety net for cases where detection and smoothing both failed
    to prevent excessive stagnation.

    Args:
        chords: Input chord sequence
        max_consecutive: Maximum allowed consecutive bars before breaking
        diatonic_chords: Optional list of diatonic chords for key-aware fallback
    """
    print(f"[STAGNATION-DEBUG] Function called with {len(chords)} chords")

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
                chord_counts = Counter(result)
                # Find most common chord that's different from current
                for chord, count in chord_counts.most_common():
                    if chord != result[i]:
                        alt_chord = chord
                        print(f"[STAGNATION] Using fallback chord: {alt_chord} (frequency: {count})")
                        break

            # Fix D: Strategy 3: diatonic chordsから代替コードを選定
            # 全バーが同一コードの場合（unique=1）に代替コード候補がない → 音楽理論的代替を使用
            if not alt_chord and diatonic_chords:
                stuck_chord = result[i]
                # diatonic chordsのうち、現在のコードと異なり、かつCHORD_LABELSに存在するものを選ぶ
                diatonic_alts = [c for c in diatonic_chords if c != stuck_chord and c in CHORD_LABELS]
                if diatonic_alts:
                    # 優先順位: minor chords（より自然な変化）> major chords
                    minor_alts = [c for c in diatonic_alts if 'm' in c]
                    alt_chord = minor_alts[0] if minor_alts else diatonic_alts[0]
                    print(f"[STAGNATION] Using diatonic fallback: {alt_chord} (from key diatonic chords)")

            # Strategy 4 (last resort): 常に何らかのコードを選ぶ
            if not alt_chord:
                # 現在のコードのルートに5度のコードを選択（最も自然な動き）
                stuck_root_idx = chord_root_index(result[i])
                fifth_up_root = (stuck_root_idx + 7) % 12
                fifth_chord = NOTE_NAMES[fifth_up_root] + "m7"  # minor 7th - 保守的な選択
                if fifth_chord in CHORD_LABELS:
                    alt_chord = fifth_chord
                    print(f"[STAGNATION] Using 5th-up fallback: {alt_chord}")
                else:
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
        return 0.7  # バラード/ブルース/R&B
    elif 100 < bpm <= 140:
        return 0.8  # ポップ/ロック（中速を強調）
    elif 140 < bpm <= 180:
        return 0.5  # アップテンポロック/EDM（抑制）
    elif 180 < bpm <= 240:
        return 0.4  # 高速ロック/パンク（抑制）
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
    倍速・1.5倍速検出を検証し、必要に応じて補正。
    位相エネルギー集中度をゲート条件として使用:
    候補に明確な拍構造がある場合のみ補正を許可。
    """
    if detected_bpm < 60 or detected_bpm > 240:
        print(f"[OctaveVerify] BPM {detected_bpm:.1f} outside correction range, keeping as-is")
        return detected_bpm, 1.0

    # 候補BPMの計算: 半速、1.5倍速、全速
    half_bpm = detected_bpm * 0.5
    two_thirds_bpm = detected_bpm / 1.5  # 1.5倍速の逆数

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

    # 1.5倍速チェック: 検出されたBPMが本来のBPMの1.5倍である可能性
    if 60 <= two_thirds_bpm <= 200:
        bass_ac_two_thirds = evaluate_bass_ac(y, sr, two_thirds_bpm)
        full_ac_two_thirds = evaluate_fullband_ac(onset_env, sr, two_thirds_bpm)
        phase_two_thirds = evaluate_phase_concentration(onset_env, sr, two_thirds_bpm)
        prior_two_thirds = evaluate_tempo_prior(two_thirds_bpm)

        score_two_thirds = (bass_ac_two_thirds * 0.35 + full_ac_two_thirds * 0.25 +
                           phase_two_thirds * 0.20 + prior_two_thirds * 0.20)

        print(f"[OctaveVerify] {two_thirds_bpm:.1f} BPM (x2/3, potential 1.5x correction): "
              f"bass_ac={bass_ac_two_thirds:.3f}, full={full_ac_two_thirds:.3f}, "
              f"phase={phase_two_thirds:.3f}, prior={prior_two_thirds:.1f}, total={score_two_thirds:.3f}")

        # 1.5倍速補正の条件: 2/3速のスコアが検出BPMより良い場合
        # 閾値を緩和: 5%以上のスコア向上 かつ 位相集中度が高い場合
        score_ratio = score_two_thirds / max(score_full, 1e-9)
        if score_two_thirds > score_full * 1.05:  # 5%以上のスコア向上
            print(f"[OctaveVerify] 1.5x speed correction applied: {detected_bpm:.1f} -> {two_thirds_bpm:.1f} BPM "
                  f"(score improved by {score_two_thirds/score_full:.2f}x)")
            return two_thirds_bpm, 0.666

    # ゲート条件: 半分速候補のビート位相エネルギー集中度
    # 真のテンポが半分速なら、その周期でダウンビートにエネルギーが集中する。
    # 集中度が低い(< PHASE_GATE)場合、半分速にリズム構造がない → 補正しない。
    PHASE_GATE = 0.25
    SCORE_OVERRIDE_RATIO = 1.15  # さらに緩和: 15%以上のスコア差で×0.5を適用
    score_ratio = score_half / max(score_full, 1e-9)
    score_override = score_ratio >= SCORE_OVERRIDE_RATIO

    if phase_half < PHASE_GATE and not score_override:
        print(f"[OctaveVerify] Half-tempo phase={phase_half:.3f} < {PHASE_GATE} "
              f"(weak beat structure, score_ratio={score_ratio:.2f} < {SCORE_OVERRIDE_RATIO}) "
              f"-> keeping {detected_bpm:.1f} for BassTempoCheck")
        print(f"[OctaveVerify] Selected: {detected_bpm:.1f} BPM (factor=x1.0, score={score_full:.3f})")
        return detected_bpm, 1.0
    elif phase_half < PHASE_GATE and score_override:
        print(f"[OctaveVerify] Half-tempo phase={phase_half:.3f} < {PHASE_GATE} "
              f"BUT score_ratio={score_ratio:.2f} >= {SCORE_OVERRIDE_RATIO} "
              f"(very strong half-tempo signal) -> overriding gate to ×0.5")

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

    # ステータスクオ保護: 大きなシフトほど高い改善率を要求 (基準2% + シフト量×2%/beat)
    if best_idx != 0:
        improvement = (scores_b[best_idx] - scores_b[0]) / (abs(scores_b[0]) + 1e-8)
        SHIFT_PENALTY_PER_BEAT = 0.02  # 大きなシフトを抑制: 1ビートあたり追加2%の改善を要求
        required_improvement = 0.02 + abs(best_shift) * SHIFT_PENALTY_PER_BEAT
        if improvement < required_improvement:
            print(f"[BarPhaseRefine] Step B: shift={best_shift} "
                  f"improvement={improvement:.3f} < required={required_improvement:.3f}, keeping shift=0")
            best_shift = 0.0

    final_phase = refined_beat_phase + best_shift * beat_duration
    print(f"[BarPhaseRefine] Final: shift={best_shift}, "
          f"{phase_offset_sec*1000:.1f}ms -> {final_phase*1000:.1f}ms")
    return final_phase, best_shift

# --- Endpoints ---

class YouTubeRequest(BaseModel):
    url: str

def get_default_cookies_path() -> str | None:
    """
    デフォルトの cookies.txt ファイルパスを取得
    プロジェクトルートの cookies.txt を探す
    """
    # backend ディレクトリではなく、プロジェクトルートの cookies.txt を探す
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_cookies_path = os.path.join(project_root, "cookies.txt")
    if os.path.exists(default_cookies_path):
        return default_cookies_path
    return None


def download_youtube_audio(url: str, cookie_path: str | None = None) -> str:
    """
    Download audio from YouTube URL using yt-dlp.
    Returns the path to the downloaded file.
    
    Retry strategy:
      Attempt 1: bestaudio[ext=m4a]/bestaudio/best (preferred for browser playback)
      Attempt 2: bestaudio (any format, broader compatibility)
    """
    app_logger.debug(f"yt-dlp start: {url}")
    # Determine FFmpeg location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_bin_dir = os.path.join(base_dir, "bin")
    ffmpeg_exe = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
    
    # Check if local ffmpeg exists, otherwise rely on system PATH
    ffmpeg_location = ffmpeg_bin_dir if os.path.exists(ffmpeg_exe) else None

    request_id = uuid.uuid4().hex
    outtmpl = os.path.join(TEMP_DIR, f"{request_id}-%(id)s.%(ext)s")

    # Node.js をJSランタイムとして使用（yt-dlpのYouTube署名解読に必須）
    # yt-dlp 2026.03+ ではJSランタイムなしだと一部動画で空ファイルになる
    js_runtimes = {"node": {}}
    # denoもフォールバックとして登録
    try:
        import shutil
        if shutil.which("deno"):
            js_runtimes["deno"] = {}
    except Exception:
        pass

    formats_to_try = [
        "bestaudio[ext=m4a]/bestaudio/best",  # 1st: M4A preferred
        "bestaudio/best",                      # 2nd: any audio format
    ]

    last_error = None
    for attempt, fmt in enumerate(formats_to_try):
        ydl_opts = {
            "format": fmt,
            "noplaylist": True,
            "socket_timeout": 60,
            "retries": 10,
            "fragment_retries": 10,
            "concurrent_fragment_downloads": 3,
            "geo_bypass": True,
            "nopart": True,
            "overwrites": True,
            "extract_flat": False,
            "ignoreerrors": False,
            "nocheckcertificate": False,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android"],
                }
            },
            # HTTPヘッダーを設定（RestrictedMode等の回避）
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.youtube.com/",
                "Origin": "https://www.youtube.com",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Ch-Ua": '"Google Chrome";v="135", "Not-A.Brand";v="8"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
            },
            "outtmpl": outtmpl,
            "quiet": False,
            "no_warnings": False,
            "ffmpeg_location": ffmpeg_location,
            "js_runtimes": js_runtimes,
            "remote_components": ["ejs:github"],  # リモートコンポーネントチャレンジソルバースクリプトをダウンロード
        }

        if cookie_path:
            ydl_opts["cookiefile"] = cookie_path
            app_logger.debug("yt-dlp using cookies")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                app_logger.debug(f"yt-dlp attempt {attempt+1}/{len(formats_to_try)}: format={fmt}")
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)

                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    if file_size > 0:
                        app_logger.debug(f"yt-dlp done: {filename} ({file_size} bytes)")
                        return filename
                    else:
                        # 0バイトファイルの場合、次のフォーマットで再試行
                        app_logger.warning(f"yt-dlp downloaded 0-byte file, removing and retrying...")
                        os.remove(filename)
                        last_error = DownloadError("Downloaded file is empty (0 bytes)")
                        continue

                last_error = RuntimeError("yt-dlp download succeeded but output file not found")
                continue

        except DownloadError as e:
            msg = str(e)
            print(f"[ERROR] yt-dlp download error (attempt {attempt+1}): {msg}")
            app_logger.error(f"yt-dlp download error (attempt {attempt+1}): {msg}")

            # 即座にリトライ不要なエラー（再試行しても改善しない）
            # 429 Too Many Requests
            if "Too many requests" in msg or "HTTP Error 429" in msg:
                raise HTTPException(
                    status_code=429,
                    detail="YouTubeのレート制限に達しました。数分後に再度お試しください。"
                )

            # 403 Forbidden (Login/Bot/Privacy/RestrictedMode)
            if ("Sign in to confirm you're not a bot" in msg
                or "confirm you're not a bot" in msg
                or "This video is only available to Music Premium members" in msg
                or "Private video" in msg
                or "HTTP Error 403" in msg
                or "Forbidden" in msg
                or "RestrictedMode" in msg):
                app_logger.error(f"403 Forbidden detected: {msg}")
                raise HTTPException(
                    status_code=403,
                    detail="YouTubeがアクセスを拒否しました（制限付きモード）。別の動画を試すか、cookies.txt をアップロードしてください。"
                )

            # リトライ可能なエラー: 次のフォーマットで再試行
            last_error = e
            continue

    # すべてのフォーマットで失敗
    if last_error:
        msg = str(last_error)
        # 空ファイルエラーの場合のユーザーフレンドリーなメッセージ
        if "empty" in msg.lower() or "0 bytes" in msg.lower():
            raise HTTPException(
                status_code=422,
                detail="YouTubeからの音声データが空でした。動画が利用可能か確認してください。cookies.txt のアップロードで解決する場合があります。"
            )
        if "JavaScript" in msg or "JS runtime" in msg or "js_runtimes" in msg:
            raise HTTPException(
                status_code=500,
                detail="サーバーにJavaScriptランタイムが不足しています。管理者にお問い合わせください。"
            )
        raise HTTPException(status_code=500, detail=f"ダウンロードに失敗しました: {msg}")
    
    raise HTTPException(status_code=500, detail="ダウンロードに失敗しました（原因不明）")

def analyze_audio_file(file_path: str, progress_callback=None, offset_sec: float = 0.0, duration_limit_sec: float | None = None, forced_bpm: float | None = None, forced_phase: float | None = None, forced_beats_per_seg: int | None = None, forced_last_chord: str | None = None, forced_run_length: int | None = None) -> dict:
    """Core analysis logic reusable for both uploads and URLs.

    Args:
        forced_phase: Optional phase offset in seconds to force (for multi-chunk consistency)
        forced_beats_per_seg: Optional beats_per_segment to force (for multi-chunk consistency)
        forced_last_chord: Optional last chord from previous chunk (for stagnation continuity)
        forced_run_length: Optional run length from previous chunk (for stagnation continuity)
    """
    
    def _progress(p: float):
        if progress_callback:
            try:
                progress_callback(float(p))
            except Exception:
                pass

    app_logger.debug(f"Starting analysis for {file_path} (offset={offset_sec}, dur={duration_limit_sec})")
    app_logger.info(f"mem start: {mem_mb():.1f} MB")
    _progress(5) # Start
    low_memory_mode = os.getenv("LOW_MEMORY_CHROMA", "1").lower() not in ("0", "false", "no")

    try:
        # 1. Load & Preprocess
        # LIMIT DURATION to 2 minutes for stability (Render 502/OOM fix)
        MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "120"))

        # チャンク長の決定：duration_limit_sec が来ていれば優先、なければ MAX_ANALYSIS_SEC
        load_dur = duration_limit_sec if duration_limit_sec is not None else MAX_ANALYSIS_SEC

        # Audio loading with FFmpeg error handling
        try:
            y, sr = librosa.load(file_path, sr=22050, mono=True, offset=float(offset_sec), duration=float(load_dur))
            app_logger.info(f"mem after load: {mem_mb():.1f} MB")
            _progress(20) # Loaded

            app_logger.debug(f"Audio loaded. Size: {y.size}, SR: {sr}")
            if y.size == 0:
                raise ValueError("Audio file is empty or unreadable")
        except Exception as load_error:
            app_logger.error(f"Failed to load audio file {file_path}: {load_error}")
            # FFmpeg関連のエラーを検出
            error_msg = str(load_error).lower()
            if "ffmpeg" in error_msg or "decoder" in error_msg or "codec" in error_msg:
                raise RuntimeError(
                    "Audio decoding failed. FFmpeg may not be installed on the server. "
                    "Please contact support or try a WAV file instead."
                ) from load_error
            raise RuntimeError(f"Audio loading failed: {load_error}") from load_error

        y = highpass_filter(y, sr)
        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        app_logger.debug(f"Audio duration: {duration_sec}s")

        # 2. Beat tracking
        # オンセット検出（BPM検出・位相検出の両方で使用）
        # パラメータ調整: より多くのオンセットを検出するために感度を上げる
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames_detected = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units='frames',
            backtrack=True, pre_max=3, post_max=3  # 感度向上
        )
        onset_set = set(onset_frames_detected.tolist())
        total_frames = len(onset_env)
        num_onsets = len(onset_frames_detected)
        app_logger.debug(f"Detected {num_onsets} onsets in {total_frames} frames")

        octave_factor = 1.0  # オクターブ補正時に更新される
        phase_detect_bpm = None  # 位相検出用のBPM（オクターブ補正前）
        beats_per_seg = 2  # Default to 2 beats per segment (fallback grid)

        if forced_bpm is not None:
            bpm = forced_bpm
            beat_frames = []
            app_logger.debug(f"Using forced BPM: {bpm:.1f}")
        else:
            app_logger.debug("Detecting BPM...")

            # === librosa.beat_trackによるBPM検出（優先） ===
            # librosa.beat_trackは堅牢で、多くの場合で正確なBPMを検出
            app_logger.debug("librosa.beat_track BPM detection...")
            try:
                librosa_tempo, librosa_beats = librosa.beat.beat_track(y=y, sr=sr)
                app_logger.info(f"[librosa.beat_track] Detected BPM: {librosa_tempo:.2f}, Beats: {len(librosa_beats)}")

                # librosa.beat_trackの結果を初期BPMとして使用
                bpm = librosa_tempo
                app_logger.info(f"[BPM Detection] Using librosa.beat_track result as initial BPM: {bpm:.2f}")

                # librosa.beat_trackで十分な信頼性がある場合、他の検出をスキップ
                # ただし、オクターブ補正やバス帯域チェックは適用
            except Exception as e:
                app_logger.warning(f"[librosa.beat_track] Failed: {e}, falling back to full-band detection")
                bpm = None

            # === 全帯域BPM検出（フォールバック） ===
            if bpm is None:
                app_logger.debug("Full-band BPM detection (fallback)...")

            # BPM 60-240を1刻みでスキャンし、F1スコアが最大のBPMを選択
            best_bpm = 120.0
            best_score = -1.0
            tolerance = 4  # 感度向上: より広い許容範囲
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
                app_logger.debug(f"BPM candidate: {c} (P={p:.3f} R={r:.3f} Fb={s:.3f})")

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
                    app_logger.debug(f"AC peak too far ({refined_bpm:.1f}), keeping coarse {coarse_bpm:.0f}")
                elif abs(refined_bpm - coarse_bpm) < 1.0:
                    # 1BPM未満の差 → ACリファイン結果を採用（累積ドリフト防止）
                    bpm = round(refined_bpm, 2)
                    app_logger.debug(f"AC refined={refined_bpm:.2f} (delta<1), using refined {bpm:.2f} "
                          f"(lag={refined_lag:.2f}, ac={ac_confidence:.3f})")
                else:
                    # 1-5 BPMの差 → ACリファイン結果を採用
                    bpm = round(refined_bpm, 2)
                    app_logger.debug(f"BPM refined via autocorrelation: {coarse_bpm:.0f} -> {bpm:.2f} "
                          f"(lag={refined_lag:.2f}, ac={ac_confidence:.3f}, coarse_Fb={coarse_f1:.3f})")
            else:
                bpm = coarse_bpm
                app_logger.debug(f"Audio too short for AC refinement, keeping coarse BPM: {coarse_bpm:.0f}")

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
            if bpm is not None and bpm > 140:
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
                        ratio = min(best_bass_bpm, bpm) / max(best_bass_bpm, bpm)
                        VALID_RATIOS = [0.5, 1.0/3, 0.25, 2.0/3]  # 0.75 を除外：半テンポ補正を優先
                        TOLERANCE = 0.05  # 厳格化: 0.75（半テンポ）を拒絶
                        is_valid_ratio = any(abs(ratio - r) < TOLERANCE for r in VALID_RATIOS)
                        if is_valid_ratio:
                            # 60-80 BPMの範囲内の場合、2倍のBPMも候補として考慮
                            # バスドラムが2拍ごとに強く現れている場合、実際のBPMは2倍の可能性がある
                            if 60 <= best_bass_bpm <= 80:
                                doubled_bpm = best_bass_bpm * 2
                                if 100 <= doubled_bpm <= 140:
                                    doubled_prior = evaluate_tempo_prior(doubled_bpm)
                                    best_bass_prior = evaluate_tempo_prior(best_bass_bpm)
                                    if doubled_prior > best_bass_prior:
                                        print(f"[BassTempoCorrection] {bpm:.1f} → {doubled_bpm:.0f} BPM "
                                              f"(bass={best_bass_bpm:.0f} × 2, 2x has higher prior)")
                                        bpm = doubled_bpm
                                        octave_factor = 1.0
                                    else:
                                        print(f"[BassTempoCorrection] {bpm:.1f} → {best_bass_bpm:.0f} BPM "
                                              f"(ratio={ratio:.3f}, valid ratio = {r:.3f})")
                                        bpm = best_bass_bpm
                                        octave_factor = 1.0
                                else:
                                    print(f"[BassTempoCorrection] {bpm:.1f} → {best_bass_bpm:.0f} BPM "
                                          f"(ratio={ratio:.3f}, valid ratio = {r:.3f})")
                                    bpm = best_bass_bpm
                                    octave_factor = 1.0
                            else:
                                print(f"[BassTempoCorrection] {bpm:.1f} → {best_bass_bpm:.0f} BPM "
                                      f"(ratio={ratio:.3f}, valid ratio = {r:.3f})")
                                bpm = best_bass_bpm
                                octave_factor = 1.0
                        else:
                            candidate_3_2 = best_bass_bpm * 1.5
                            if abs(candidate_3_2 - bpm) / bpm < 0.05:  # 3/2 補正を厳格化
                                # best_bass_bpm は step=2 BPM スキャン (±1 BPM 精度)
                                # hop=128 の bass AC + parabolic interpolation で精密化してから 3/2 補正
                                try:
                                    hop_fine = 128
                                    _y_bass_fine = lowpass_filter(y, sr, cutoff_hz=200)
                                    _y_bass_fine = highpass_filter(_y_bass_fine, sr, cutoff_hz=20)
                                    _bass_env_fine = librosa.onset.onset_strength(
                                        y=_y_bass_fine, sr=sr, hop_length=hop_fine)
                                    del _y_bass_fine
                                    _ac_fine = librosa.autocorrelate(_bass_env_fine)
                                    del _bass_env_fine
                                    if _ac_fine[0] > 0:
                                        _ac_fine = _ac_fine / _ac_fine[0]
                                    _fine_lag_center = 60.0 * sr / (best_bass_bpm * hop_fine)
                                    _fine_radius = max(3, int(_fine_lag_center * 0.03))
                                    _fine_lo = max(1, int(_fine_lag_center) - _fine_radius)
                                    _fine_hi = min(len(_ac_fine) - 2, int(_fine_lag_center) + _fine_radius)
                                    refined_bass_bpm = best_bass_bpm
                                    if _fine_hi > _fine_lo:
                                        _pk = _fine_lo + int(np.argmax(_ac_fine[_fine_lo:_fine_hi + 1]))
                                        if 0 < _pk < len(_ac_fine) - 1:
                                            _a = float(_ac_fine[_pk - 1])
                                            _b = float(_ac_fine[_pk])
                                            _g = float(_ac_fine[_pk + 1])
                                            _denom = _a - 2.0 * _b + _g
                                            _delta = 0.5 * (_a - _g) / _denom if abs(_denom) > 1e-10 else 0.0
                                            _rl = _pk + _delta
                                            _rb = 60.0 * sr / (_rl * hop_fine) if _rl > 0 else best_bass_bpm
                                            if abs(_rb - best_bass_bpm) < 2.0:
                                                refined_bass_bpm = _rb
                                    del _ac_fine
                                    refined_candidate = refined_bass_bpm * 1.5
                                    print(f"[BassTempoCorrection] {bpm:.1f} → {refined_candidate:.2f} BPM "
                                          f"(bass={best_bass_bpm:.0f} → {refined_bass_bpm:.2f} × 3/2, "
                                          f"ratio={ratio:.3f} invalid)")
                                    bpm = refined_candidate
                                except Exception as _e:
                                    print(f"[BassTempoCorrection] {bpm:.1f} → {candidate_3_2:.1f} BPM "
                                          f"(bass={best_bass_bpm:.0f} × 3/2, ratio={ratio:.3f} invalid, "
                                          f"refinement failed: {_e})")
                                    bpm = candidate_3_2
                                octave_factor = 1.0
                            else:
                                print(f"[BassTempoCheck] Ratio {ratio:.3f} not valid musical ratio, "
                                      f"keeping {bpm:.1f} BPM")
                    else:
                        print(f"[BassTempoCheck] Keeping {bpm:.1f} BPM "
                              f"(bass advantage insufficient)")
                else:
                    print(f"[BassTempoCheck] Bass peak {best_bass_bpm:.0f} BPM "
                          f"close to detected {bpm:.1f}, no correction needed")

        # Stage 2.8: BeatNetによるBPM検出（高精度）
        # BeatNetはロック/ポップス音楽に特化しており、高い精度を誇る
        if forced_bpm is None and not low_memory_mode:
            app_logger.info("[BeatNet] Attempting BeatNet BPM detection...")
            beatnet_result = detect_bpm_with_beatnet(y, sr)
            if beatnet_result is not None:
                beatnet_bpm, beatnet_confidence = beatnet_result
                # BeatNetの結果が妥当な範囲（60-200 BPM）か確認
                if 60 <= beatnet_bpm <= 200:
                    # 既存のBPMと比較し、大きく異なる場合は慎重に判断
                    ratio = beatnet_bpm / bpm if bpm > 0 else 1.0
                    # 比率が1.2倍以内、または既存のBPMが極端な値の場合はBeatNetを採用
                    if 0.83 <= ratio <= 1.2 or bpm < 80 or bpm > 180:
                        app_logger.info(f"[BeatNet] Using BeatNet result: {beatnet_bpm:.2f} BPM (replaces {bpm:.1f} BPM)")
                        bpm = beatnet_bpm
                    else:
                        app_logger.info(f"[BeatNet] BeatNet result {beatnet_bpm:.2f} BPM differs from current {bpm:.1f} BPM (ratio={ratio:.2f}x), keeping current BPM")
                else:
                    app_logger.warning(f"[BeatNet] BeatNet result out of range: {beatnet_bpm:.2f} BPM")
            else:
                app_logger.info("[BeatNet] BeatNet detection failed, using current BPM")

        # Stage 3: ビート位相検出 - 最適なグリッド開始位置を探索
        if forced_phase is not None:
            # チャンク統一のため、位相を強制使用
            if offset_sec > 0:
                # チャンクオフセットを考慮してローカル位相を計算
                # グローバルビートグリッド: forced_phase + n * seg_dur
                # このチャンク内の最初のビート位置を求める
                beat_dur = 60.0 / bpm
                _bps_for_phase = forced_beats_per_seg if forced_beats_per_seg is not None else 2
                seg_dur = beat_dur * _bps_for_phase
                elapsed = offset_sec - forced_phase
                if elapsed > 0:
                    remainder = elapsed % seg_dur
                    phase_offset_sec = (seg_dur - remainder) if remainder > 1e-6 else 0.0
                else:
                    phase_offset_sec = forced_phase - offset_sec
                app_logger.debug(f"Beat phase offset: {phase_offset_sec*1000:.1f}ms "
                      f"(local phase for chunk at offset={offset_sec:.1f}s, "
                      f"global phase={forced_phase*1000:.1f}ms)")
            else:
                phase_offset_sec = forced_phase
                app_logger.debug(f"Beat phase offset: {phase_offset_sec*1000:.1f}ms (forced from chunk 0)")
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
            app_logger.debug(f"Beat phase offset: {phase_offset_sec*1000:.1f}ms (detected at {_phase_bpm:.1f}BPM, precision={best_phase_score:.4f})")
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

        # Force garbage collection before chroma computation (memory-intensive)
        import gc
        gc.collect()
        app_logger.info("[GC] Forced garbage collection before chroma")
        low_memory_mode = os.getenv("LOW_MEMORY_CHROMA", "1").lower() not in ("0", "false", "no")

        # Memory check before chroma computation (critical point)
        try:
            chroma_mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
            app_logger.info(f"[Memory] Before chroma: {chroma_mem_mb:.1f}MB")
            if chroma_mem_mb > 450:  # 450MB超過で早期リターン（gc後に再チェック）
                low_memory_mode = True
                app_logger.warning(f"[Memory] Enabling low-memory chroma path at {chroma_mem_mb:.1f}MB")
        except Exception as mem_e:
            if isinstance(mem_e, MemoryError):
                raise
            app_logger.warning(f"[Memory] Could not check memory before chroma: {mem_e}")

        if low_memory_mode:
            try:
                del onset_env
                gc.collect()
                app_logger.info("[GC] Released onset_env before low-memory chroma")
            except Exception:
                pass

        # 3. Chroma
        app_logger.debug("Computing chroma...")
        hop_length = 2048  # 4096→2048: 時間解像度2倍（~93ms@22050Hz）
        if low_memory_mode:
            app_logger.info("Computing chroma using low-memory STFT path...")
            chroma = compute_chroma_stft_light(y, sr, hop_length=hop_length)
        else:
            chroma = compute_chroma_cqt(y, sr, hop_length=hop_length)
        chroma = apply_chroma_contrast(chroma, filter_size=50)  # ペダルトーン除去
        app_logger.info(f"mem after chroma: {mem_mb():.1f} MB")

        if low_memory_mode:
            bass_chroma = compute_bass_chroma_light(y, sr, hop_length=hop_length)
        else:
            bass_chroma = compute_bass_chroma(y, sr, hop_length=hop_length)
        bass_chroma = apply_chroma_contrast(bass_chroma, filter_size=50)  # ペダルトーン除去
        app_logger.info(f"mem after bass chroma: {mem_mb():.1f} MB")
        app_logger.debug(f"Chroma shape: {chroma.shape}, Bass chroma shape: {bass_chroma.shape}")
        _progress(60) # Chroma done

        if chroma.shape[1] == 0:
            raise ValueError("Chroma extraction failed or audio too short")

        # 4. Time axes
        # Improved: Use frame-based timing for more precise segment boundaries
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

        # Calculate beat duration for diagnostics
        beat_duration = 60.0 / bpm
        target_segment_duration = beat_duration * 2

        # --- ハイブリッド BPM 検出の統合 ---
        # 既存の BPM 検出結果とハイブリッド方法を比較し、より良い結果を採用
        # ただし、forced_bpm が渡されている場合はスキップ（2番目以降のチャンクでBPMを統一するため）
        if forced_bpm is None and not low_memory_mode:
            app_logger.debug("Integrating hybrid BPM detection...")
            try:
                # 元のBPMを中心に±40 BPMの範囲でハイブリッド検出を実行
                # これにより、1.5倍速の誤検出（例: 105 BPM → 156 BPM）を防止
                bpm_search_min = max(60, int(bpm - 40))
                bpm_search_max = min(240, int(bpm + 40))
                app_logger.info(f"[Hybrid BPM] Searching BPM range: {bpm_search_min}-{bpm_search_max} (around original {bpm:.1f} BPM)")

                hybrid_bpm, dl_confidence = detect_bpm_with_deep_learning(y, sr, bpm_range=(bpm_search_min, bpm_search_max))
                app_logger.info(f"[Hybrid BPM] DL result: {hybrid_bpm:.1f} BPM (confidence: {dl_confidence:.3f})")

                # ディープラーニングの信頼度が高い（>0.9）場合でも、元のBPMと大きく異なる場合は慎重に判断
                # バス帯域チェック等で修正されたBPM（中速）を尊重するため、ratioが1.3以上または0.77以下の場合は元のBPMを維持
                if dl_confidence > 0.9:
                    ratio = hybrid_bpm / bpm
                    # 元のBPMと大きく異なる場合、元のBPM（バス帯域チェック等で修正されたもの）を優先
                    if ratio >= 1.3 or ratio <= 0.77:
                        app_logger.info(f"[Hybrid BPM] High confidence DL result ({dl_confidence:.3f}): {hybrid_bpm:.1f} BPM (vs original {bpm:.1f} BPM, ratio={ratio:.2f}x)")
                        app_logger.info(f"[Hybrid BPM] Large difference detected, keeping original {bpm:.1f} BPM (likely corrected by bass/tempo check)")
                    else:
                        app_logger.info(f"[Hybrid BPM] High confidence DL result ({dl_confidence:.3f}): {hybrid_bpm:.1f} BPM (vs original {bpm:.1f} BPM, ratio={ratio:.2f}x) -> USING DL RESULT")
                        bpm = hybrid_bpm
                else:
                    # 信頼度が低い場合、既存のロジックを使用
                    app_logger.info(f"[Hybrid BPM] DL confidence too low ({dl_confidence:.3f}), using original logic")

                    # オクターブ誤検出チェック: ハイブリッド結果が元のBPMの1.4-1.6倍の場合、
                    # 本来のBPMは半速（÷1.5）の可能性が高い
                    ratio = hybrid_bpm / bpm
                    if 1.4 <= ratio <= 1.6:
                        half_speed = bpm / 1.5  # 本来のBPM候補
                        if 60 <= half_speed <= 200:
                            app_logger.info(f"[Hybrid BPM] Detected potential 1.5x speed error: {bpm:.1f} -> {hybrid_bpm:.1f} BPM")
                            app_logger.info(f"[Hybrid BPM] Original BPM {bpm:.1f} is likely 1.5x faster than actual {half_speed:.1f} BPM")
                            app_logger.info(f"[Hybrid BPM] Keeping original {bpm:.1f} BPM for octace verification")

                    # 既存の BPM が範囲外（<60 or >240）の場合、ハイブリッド結果を採用
                    # 範囲内の場合、元のBPMを優先（中速バイアスを削除）
                    if bpm < 60 or bpm > 240:
                        app_logger.info(f"[Hybrid BPM] Using hybrid result: {hybrid_bpm:.1f} BPM (original {bpm:.1f} BPM out of valid range)")
                        bpm = hybrid_bpm
                    else:
                        # 元のBPMが有効範囲内の場合、ハイブリッド結果を参考にするが元のBPMを優先
                        # ハイブリッド結果が元のBPMの±10%以内の場合のみ採用を検討
                        ratio = hybrid_bpm / bpm
                        if 0.9 <= ratio <= 1.1:
                            # 両方が近い場合、より中速（120-180）に近い方を採用
                            ideal_bpm = 150.0
                            hybrid_distance = abs(hybrid_bpm - ideal_bpm)
                            original_distance = abs(bpm - ideal_bpm)

                            if hybrid_distance < original_distance * 0.95:  # より厳しい条件
                                app_logger.info(f"[Hybrid BPM] Using hybrid result: {hybrid_bpm:.1f} BPM (closer to ideal {ideal_bpm} BPM)")
                                bpm = hybrid_bpm
                            else:
                                app_logger.info(f"[Hybrid BPM] Keeping original: {bpm:.1f} BPM (already optimal)")
                        else:
                            # 差が大きい場合、元のBPMを維持
                            app_logger.info(f"[Hybrid BPM] Keeping original: {bpm:.1f} BPM (hybrid {hybrid_bpm:.1f} BPM differs by {ratio:.2f}x)")

                # BPM が更新された場合、beat_duration を再計算
                beat_duration = 60.0 / bpm
                target_segment_duration = beat_duration * 2

            except Exception as e:
                app_logger.error(f"[Hybrid BPM] Error in hybrid detection: {e}, using original {bpm:.1f} BPM")
                import traceback
                traceback.print_exc()
                # ハイブリッド方法が失敗した場合、既存の結果を使用
        else:
            app_logger.info(f"[Hybrid BPM] Skipping hybrid detection - using forced BPM: {bpm:.1f}")

        # --- ハイブリッド BPM 検出ここまで ---
        total_duration = librosa.frames_to_time(chroma.shape[1], sr=sr, hop_length=hop_length)

        # --- Adaptive Beat Tracking ---
        # 固定BPMグリッドではテンポ揺らぎに追従できず累積ドリフトが発生する。
        # librosa.beat.beat_track で実際のビート位置を検出し、セグメント境界に使用。
        # forced_phase（後続チャンク）では安定性のため固定グリッドを維持。
        if forced_phase is not None or low_memory_mode:
            # 後続チャンク: 固定グリッドで一貫性を保つ
            # beat_timesは1拍間隔で生成（aggregate_chroma_per_segmentがbeats_per_segmentでグループ化するため）
            beat_times = np.arange(phase_offset_sec, total_duration + target_segment_duration, beat_duration)
            beats_per_seg = forced_beats_per_seg if forced_beats_per_seg is not None else 2
            grid_reason = "forced_phase" if forced_phase is not None else "low_memory_mode"
            app_logger.debug(f"Using fixed grid ({grid_reason}): beats_per_seg={beats_per_seg}, seg_dur={target_segment_duration:.3f}s")
        else:
            # 初回チャンク: adaptive beat tracking
            try:
                bt_hop = 512
                # onset_envは既存のものを再利用（BPM検出で使用済み、delete済みなら再計算）
                try:
                    _ = onset_env
                except NameError:
                    onset_env = librosa.onset.onset_strength(
                        y=y, sr=sr, hop_length=bt_hop,
                        aggregate=np.median,  # 中央値の使用（外れ値の影響を軽減）
                        max_size=5  # 拡張: さらにノイズ抑制（162 BPM のような中速テンポで精度向上）
                    )

                # ビートトラッキング安定化: tightness と start_bpm を設定
                _, bt_frames = librosa.beat.beat_track(
                    onset_envelope=onset_env, sr=sr, hop_length=bt_hop,
                    bpm=bpm, trim=False,
                    tightness=150,  # 安定したテンポを優先
                    start_bpm=120.0  # 一般的な初期テンポ
                )
                bt_times = librosa.frames_to_time(bt_frames, sr=sr, hop_length=bt_hop)

                if len(bt_times) >= 4:
                    # ビートトラッカー成功: adaptive BPM を取得し、
                    # BarPhaseRefine で求めたダウンビート位相を起点に固定グリッドを生成
                    # BPM は変更しない（median interval は missed beats により inflate するため信頼性が低い）
                    avg_beat_interval = float(np.median(np.diff(bt_times)))
                    beat_times = np.arange(phase_offset_sec, total_duration + target_segment_duration, beat_duration)
                    beats_per_seg = 4
                    app_logger.debug(f"Adaptive beat tracking: {len(bt_times)} beats, "
                          f"median interval={avg_beat_interval:.3f}s "
                          f"(≈{60.0/avg_beat_interval:.1f} BPM), "
                          f"phase-aligned from {phase_offset_sec*1000:.1f}ms, "
                          f"grid step={beat_duration:.4f}s (BPM={bpm:.1f})")
                else:
                    # ビートが少なすぎる → 固定グリッドにフォールバック
                    # beat_timesは1拍間隔で生成（aggregate_chroma_per_segmentがbeats_per_segmentでグループ化するため）
                    beat_times = np.arange(phase_offset_sec, total_duration + target_segment_duration, beat_duration)
                    beats_per_seg = 2
                    app_logger.debug(f"Beat tracker returned too few beats ({len(bt_times)}), using fixed grid")
            except Exception as e:
                # ビートトラッカーエラー → 固定グリッドにフォールバック
                # beat_timesは1拍間隔で生成（aggregate_chroma_per_segmentがbeats_per_segmentでグループ化するため）
                beat_times = np.arange(phase_offset_sec, total_duration + target_segment_duration, beat_duration)
                beats_per_seg = 2
                app_logger.debug(f"Beat tracker failed ({e}), using fixed grid")

        try:
            del onset_env  # メモリ解放
        except UnboundLocalError:
            pass

        num_segments = max(1, len(beat_times) - 1)
        app_logger.debug(f"BPM: {bpm:.2f}, Beat duration: {beat_duration:.3f}s, Segments: ~{num_segments}")

        # 5. Aggregate per segment
        app_logger.debug("Aggregating segments...")
        main_matrix, segments = aggregate_chroma_per_segment(chroma, times, beat_times, beats_per_segment=beats_per_seg)
        bass_matrix, _ = aggregate_chroma_per_segment(bass_chroma, times, beat_times, beats_per_segment=beats_per_seg)
        app_logger.debug(f"Segments: {len(segments)}")
        _progress(75) # Aggregation done

        # 6. Key estimation
        key_root, key_mode = estimate_key_from_chroma(chroma)
        estimated_key = f"{key_root}{key_mode}"
        app_logger.debug(f"Key: {estimated_key}")

        # 7. Diatonic penalty
        diatonic_chords = set(get_diatonic_chords_for_key(key_root, key_mode))
        penalty_mask = np.array(
            [label not in diatonic_chords for label in CHORD_LABELS],
            dtype=bool
        )

        # 8. Detection (HMM + Viterbi)
        app_logger.debug("Detecting chords (HMM/Viterbi)...")
        smoothed_chords, final_last_chord, final_run_length = detect_chords_hmm(
            main_matrix,
            bass_matrix,
            penalty_mask=penalty_mask,
            penalty_value=0.20,
            main_weight=0.55,
            bass_weight=0.25,  # 0.50→0.25: contrast 正規化後はペダルトーン増幅不要
            temperature=4.0,   # 8.0→4.0: emission の過度な一極集中を緩和
            _forced_last_chord=forced_last_chord,
            _forced_run_length=forced_run_length,
        )
        app_logger.debug(f"HMM chords detected: {len(smoothed_chords)}")
        _progress(90)

        # Safety net: HMM が同コードに固着した場合の最終防衛（旧 max=4 → 新 max=8 で緩く設定）
        smoothed_chords = break_long_stagnation_runs(
            smoothed_chords, max_consecutive=8, diatonic_chords=list(diatonic_chords)
        )

        def calc_max_run(chords):
            max_run, current_run = 1, 1
            for i in range(1, len(chords)):
                if chords[i] == chords[i - 1]:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            return max_run

        app_logger.debug(f"HMM max stagnation: {calc_max_run(smoothed_chords)} bars")
        app_logger.debug(f"unique chords: {len(set(smoothed_chords))}")
        app_logger.debug(f"first 20 chords: {smoothed_chords[:20]}")

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
        
        app_logger.debug(f"Analysis complete. Returning {len(bars)} bars.")
        _progress(99)
        return {
            "bpm": bpm,
            "duration_sec": round(duration_sec, 1),
            "time_signature": compute_time_signature(beats_per_seg),
            "key": estimated_key,
            "bars": bars,
            "phase_offset_sec": round(phase_offset_sec, 4),
            "final_last_chord": final_last_chord,
            "final_run_length": final_run_length,
            "beats_per_segment": beats_per_seg,
            "beat_times": [float(t) for t in beat_times] if beat_times is not None else [],
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
    return {
        "status": "ok",
        "build": "build-v5.6.1-lowmem-stream-upload",
        "memory_mb": round(mem_mb(), 1),
        "active_jobs": sum(1 for j in jobs.values() if j.get("status") == "analyzing"),
        "env": {
            "MAX_ANALYSIS_SEC": os.getenv("MAX_ANALYSIS_SEC", ""),
            "CHUNK_SEC": os.getenv("CHUNK_SEC", ""),
            "FIRST_CHUNK_SEC": os.getenv("FIRST_CHUNK_SEC", ""),
            "LOW_MEMORY_CHROMA": os.getenv("LOW_MEMORY_CHROMA", ""),
            "ENABLE_LIBROSA_WARMUP": os.getenv("ENABLE_LIBROSA_WARMUP", ""),
        },
    }

@app.get("/health/ffmpeg")
def check_ffmpeg():
    """FFmpegが利用可能かチェックするエンドポイント"""
    try:
        import subprocess

        # FFmpeg binary check
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return {
                "status": "error",
                "ffmpeg_available": False,
                "message": "FFmpeg binary not found or not executable"
            }

        # Extract version from output
        version_line = result.stdout.split('\n')[0]

        return {
            "status": "ok",
            "ffmpeg_available": True,
            "message": f"FFmpeg is installed: {version_line}"
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "ffmpeg_available": False,
            "message": "FFmpeg check timed out"
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "ffmpeg_available": False,
            "message": "FFmpeg binary not found in PATH"
        }
    except Exception as e:
        return {
            "status": "error",
            "ffmpeg_available": False,
            "message": f"FFmpeg check failed: {e}"
        }

@app.get("/version")
def version():
    return {"git_sha": os.getenv("RENDER_GIT_COMMIT", "unknown")}



@app.on_event("startup")
def startup_event():
    if os.getenv("ENABLE_LIBROSA_WARMUP", "0").lower() not in ("1", "true", "yes"):
        app_logger.info("[INFO] Librosa warmup skipped")
        return
    # Warmup librosa on startup to reduce first-request latency
    try:
        y = np.zeros(22050)
        librosa.feature.chroma_stft(y=y, sr=22050)
        print("[INFO] Warmup complete")
    except:
        pass

def run_analysis_bg(job_id: str, file_path: str, mode: AnalyzeMode = AnalyzeMode.PREVIEW, source: str = "upload"):
    app_logger.info(f"[run_analysis_bg] Starting analysis job {job_id}, mode={mode}, source={source}")
    cleanup_jobs()

    # MP3ファイルの場合、FFmpegが利用可能かチェック
    if file_path.lower().endswith('.mp3'):
        try:
            import subprocess
            # FFmpeg binary check
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg binary not found")
        except Exception as e:
            app_logger.error(f"FFmpeg check failed for MP3 processing: {e}")
            jobs[job_id] = {
                **jobs.get(job_id, {}),
                "status": "error",
                "error": "MP3 decoding is not available on this server. Please use WAV format or contact support."
            }
            return

    # Force garbage collection before starting new job (free memory from previous jobs)
    import gc
    gc.collect()
    app_logger.info("[GC] Forced garbage collection before new job")

    # Initial memory check (Render free tier: 512MB limit)
    try:
        process = psutil.Process()
        initial_mem_mb = process.memory_info().rss / 1024 / 1024
        app_logger.info(f"[Memory] Initial check: {initial_mem_mb:.1f}MB")
        # Only reject if extremely high (>450MB), otherwise try to proceed
        if initial_mem_mb > 450:
            error_msg = f"Server memory critically high ({initial_mem_mb:.1f}MB). Please try again in a few minutes."
            app_logger.error(f"[OOM-Precheck] {error_msg}")
            jobs[job_id] = {
                **jobs.get(job_id, {}),
                "status": "error",
                "error": error_msg,
                "done_at": time.time()
            }
            return
    except Exception as mem_e:
        app_logger.warning(f"[Memory] Could not check initial memory: {mem_e}")

    # Concurrent job limit check (Render free tier can only handle 1 job at a time)
    # Exclude self (current job_id) from the count
    active_jobs = [j for jid, j in jobs.items() if j.get("status") == "analyzing" and jid != job_id]
    if len(active_jobs) > 0:
        error_msg = f"Another analysis is in progress. Please wait for it to complete."
        app_logger.warning(f"[Concurrency] {len(active_jobs)} active job(s), rejecting new job")
        jobs[job_id] = {
            **jobs.get(job_id, {}),
            "status": "error",
            "error": error_msg,
            "done_at": time.time()
        }
        return

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
        try:
            p = max(0.0, min(100.0, p))
            job = jobs.get(job_id, {})
            # Only update if job still exists
            if job:
                p = max(p, float(job.get("progress", 0.0) or 0.0))
                jobs[job_id] = {
                    **job,
                    "progress": p,
                    "updated_at": time.time(),
                    "started_at": job.get("started_at", time.time())
                }
        except Exception as e:
            app_logger.error(f"[update_progress] Failed to update progress for job {job_id}: {e}")

    heartbeat_stop = threading.Event()

    def heartbeat():
        while not heartbeat_stop.wait(10.0):
            try:
                job = jobs.get(job_id)
                if not job or job.get("status") != "analyzing":
                    return
                jobs[job_id] = {
                    **job,
                    "updated_at": time.time(),
                    "started_at": job.get("started_at", time.time()),
                }
                app_logger.debug(f"[Heartbeat] job {job_id} still analyzing at {job.get('progress', 0.0)}%")
            except Exception as e:
                app_logger.warning(f"[Heartbeat] Failed for job {job_id}: {e}")
                return

    threading.Thread(target=heartbeat, daemon=True).start()
        
    # FORCE UPDATE to prove thread is alive (2%)
    update_progress(2.0)

    try:
        # --- Mode Enforcement ---
        # 1. Preview Hardcap
        if mode == AnalyzeMode.PREVIEW:
             # Force 60s hardcap (ignore environment or input)
             MAX_ANALYSIS_SEC = 60.0
             app_logger.info(f"[run_analysis_bg] Mode: PREVIEW -> Forced duration 60.0s")
        elif mode == AnalyzeMode.EARLY_ACCESS:
             MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "300"))
             app_logger.info(f"[run_analysis_bg] Mode: EARLY_ACCESS -> Max duration {MAX_ANALYSIS_SEC}s")
        else:  # FULL
             MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "600"))
             app_logger.info(f"[run_analysis_bg] Mode: FULL -> Max duration {MAX_ANALYSIS_SEC}s")

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
        beats_per_seg = None  # Track beats_per_segment from first chunk

        all_bars: list[dict] = []
        all_beat_times: list[float] = []  # 全チャンクのビートタイムスタンプを統合
        key_votes: list[str] = []
        stag_last_chord: str | None = None
        stag_run_length: int | None = None

        chunk_idx = 0
        offset = 0.0

        # Safety limit to prevent infinite loops
        max_chunks = 100

        # Estimate total chunks for progress calculation (assuming MAX)
        # This is an approximation since we might stop early, but ensures 0-100 scale
        estimated_total_chunks = int(math.ceil(MAX_ANALYSIS_SEC / CHUNK_SEC))

        FIRST_CHUNK_SEC = float(os.getenv("FIRST_CHUNK_SEC", "30"))  # Keep Render free tier memory stable.

        while offset < MAX_ANALYSIS_SEC and chunk_idx < max_chunks:
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
                forced_beats_per_seg=beats_per_seg,  # 最初のチャンク以降はbeats_per_segmentを統一
                forced_last_chord=stag_last_chord,
                forced_run_length=stag_run_length,
            )

            # Check for effective end of file (short read)
            actual_dur = raw["duration_sec"]

            chunk_bars = raw["bars"]
            key_votes.append(raw.get("key", "Unknown"))
            stag_last_chord = raw.get("final_last_chord")
            stag_run_length = raw.get("final_run_length")

            # beat_times を統合
            chunk_beat_times = raw.get("beat_times", [])
            if chunk_beat_times:
                # チャンクのオフセットを適用して絶対時間に変換
                chunk_beat_times_abs = [offset + t for t in chunk_beat_times]

                # 重複を除外して追加（チャンク境界での重複を防ぐ）
                if all_beat_times:
                    # 前回の最後のビートと重複している場合、最初のビートをスキップ
                    last_beat = all_beat_times[-1]
                    if chunk_beat_times_abs and abs(chunk_beat_times_abs[0] - last_beat) < 0.001:  # 1ms未満なら重複とみなす
                        all_beat_times.extend(chunk_beat_times_abs[1:])
                    else:
                        all_beat_times.extend(chunk_beat_times_abs)
                else:
                    all_beat_times.extend(chunk_beat_times_abs)

            # 最初のチャンクからBPMと位相を取得
            if bpm is None:
                bpm = raw.get("bpm", 120.0)
                forced_phase = raw.get("phase_offset_sec", 0.0)  # 位相も保存
                beats_per_seg = raw.get("beats_per_segment", 2)  # Extract beats_per_segment
                seconds_per_beat = 60.0 / bpm
                segment_duration = seconds_per_beat * beats_per_seg
                app_logger.info(f"[ChunkMerge] Using detected BPM: {bpm:.1f}, phase: {forced_phase*1000:.1f}ms, "
                      f"beats_per_seg: {beats_per_seg}, segment_duration: {segment_duration:.3f}s")

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

            # Memory check after each chunk (Render free tier: 512MB limit)
            try:
                process = psutil.Process()
                mem_percent = process.memory_percent()
                mem_mb = process.memory_info().rss / 1024 / 1024
                app_logger.info(f"[Memory] After chunk {chunk_idx}: {mem_mb:.1f}MB ({mem_percent:.1f}%)")

                # Fail gracefully if memory is critical (>90% or >450MB)
                if mem_percent > 90 or mem_mb > 450:
                    error_msg = f"Memory limit reached: {mem_mb:.1f}MB used. Try a shorter audio file."
                    app_logger.error(f"[OOM] {error_msg}")
                    jobs[job_id] = {
                        **jobs.get(job_id, {}),
                        "status": "error",
                        "error": error_msg,
                        "done_at": time.time()
                    }
                    return
            except Exception as mem_e:
                app_logger.warning(f"[Memory] Could not check memory: {mem_e}")

            # Force garbage collection after each chunk to free memory
            import gc
            gc.collect()

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
            time_sig = compute_time_signature(beats_per_seg) if beats_per_seg is not None else "4/4"
            all_bars = add_bar_timing(
                all_bars,
                bpm=bpm,
                time_signature=time_sig,
                analyzed_duration_sec=offset
            )

        # チャンク結合後のタイミング:
        # forced_phase によりチャンク間タイミングは連続（ギャップ < 1ms）
        # 統一グリッド上書きは廃止: BPM誤差の蓄積ドリフトを防止し、
        # per-chunkの精密なchromaフレーム境界タイミングを保持する

        # 診断: バー間隔を確認
        if len(all_bars) >= 2:
            _diag_dur = round(all_bars[1]["start_sec"] - all_bars[0]["start_sec"], 4)
            beats_used = beats_per_seg if beats_per_seg is not None else 2
            _expected = round((60.0 / bpm) * beats_used, 4)
            print(f"[ChunkMerge] Bar duration: {_diag_dur}s (per-chunk timing, expected ~{_expected}s @ {beats_used} beats/segment)")
            print(f"[ChunkMerge] Total bars: {len(all_bars)}, first={all_bars[0]['start_sec']:.4f}s, last_end={all_bars[-1]['end_sec']:.4f}s")

        # beat_times が空の場合のフォールバック
        if not all_beat_times:
            app_logger.warning("[ChunkMerge] No beat_times available, generating from BPM")
            seconds_per_beat = 60.0 / bpm
            all_beat_times = [i * seconds_per_beat for i in range(int(bpm * offset / 60) + 2)]
            app_logger.info(f"[ChunkMerge] Generated {len(all_beat_times)} beat times from BPM {bpm:.1f}")
        else:
            # beat_times の妥当性チェック（ソート済み確認）
            all_beat_times = sorted(all_beat_times)
            app_logger.info(f"[ChunkMerge] Merged {len(all_beat_times)} beat times")

        # beat_times から実際の BPM を再計算（beat_times が存在する場合）
        if all_beat_times and len(all_beat_times) > 10:  # 少なくとも10ビート必要
            # 安定したビート間隔を中央値で計算（外れ値除外）
            beat_intervals = np.diff(all_beat_times)
            # 中央値（外れ値に強い統計指標）
            median_interval = np.median(beat_intervals)

            # ヒストグラム分析で最頻出現ビート間隔を取得（外れ値に強い）
            hist, bin_edges = np.histogram(beat_intervals, bins=50, density=True)
            most_common_interval_idx = np.argmax(hist)
            mode_interval = (bin_edges[most_common_interval_idx] + bin_edges[most_common_interval_idx + 1]) / 2
            mode_bpm = 60.0 / mode_interval

            app_logger.info(f"[ModeBPM] Mode interval = {mode_interval:.3f}s, Mode BPM = {mode_bpm:.1f}")

            # 中央値とモードの平均を取る（安定性向上）
            avg_interval = (median_interval + mode_interval) / 2
            calculated_bpm = 60.0 / avg_interval

            # 異常な間隔（外れ値）を除外した平均を計算
            # 中央値の 20% 以外の間隔は外れ値とみなす
            valid_intervals = [dt for dt in beat_intervals if abs(dt - avg_interval) < 0.2 * avg_interval]

            if valid_intervals:
                final_avg_interval = np.mean(valid_intervals)
                final_bpm = 60.0 / final_avg_interval

                # BPM が合理的な範囲（60-240 BPM）内の場合のみ上書き
                if 60 <= final_bpm <= 240:
                    app_logger.info(f"[DynamicBPM] Calculated: {final_bpm:.1f} BPM "
                          f"(mode={mode_bpm:.1f}, median={60.0/median_interval:.1f}, avg={calculated_bpm:.1f})")
                    app_logger.warning(f"[DynamicBPM] OVERWRITING BPM: {bpm:.1f} -> {final_bpm:.1f}")
                    bpm = final_bpm
                else:
                    app_logger.info(f"[DynamicBPM] Calculated BPM {final_bpm:.1f} out of range [60-240], keeping {bpm:.1f}")
            else:
                app_logger.info(f"[DynamicBPM] Not enough valid intervals, keeping {bpm:.1f}")
        else:
            app_logger.info(f"[DynamicBPM] Not enough beat_times ({len(all_beat_times) if all_beat_times else 0}), keeping {bpm:.1f}")

        final_result = {
            "bpm": bpm,
            "duration_sec": round(offset, 1),
            "time_signature": compute_time_signature(beats_per_seg) if beats_per_seg is not None else "4/4",
            "key": key,
            # Strict Return Schema based on Mode
            "mode": mode,
            "is_preview": (mode == AnalyzeMode.PREVIEW),
            "analyzed_duration_sec": round(offset, 1),
            "export_allowed": (mode == AnalyzeMode.EARLY_ACCESS or mode == AnalyzeMode.FULL),
            "bars": all_bars, # Return bars even in Preview (limited by duration cap)
            "beat_times": all_beat_times,
            "_build": "build-v5.6.0",
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
        app_logger.error(f"[run_analysis_bg] Analysis failed for job {job_id}: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        jobs[job_id] = {
            **jobs.get(job_id, {}),
            "status": "error",
            "done_at": time.time(),
            "error": str(e),
        }

    finally:
        heartbeat_stop.set()
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

    # Create job entry immediately (status="pending" until file is saved)
    jobs[job_id] = {
        "status": "pending",
        "submitted_at": now,
        "expires_at": now + JOB_TTL_SEC,
        "progress": 0.0,
        "updated_at": now,
    }

    # Use a unique name for TEMP storage
    safe_filename = f"{job_id}{ext}"
    file_path = os.path.join(TEMP_DIR, safe_filename)

    try:
        file.file.seek(0)
        await anyio.to_thread.run_sync(_save_upload_sync, file.file, file_path)
    except Exception as e:
        jobs.pop(job_id, None)
        app_logger.error(f"[UploadSave] Failed to save upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    jobs[job_id] = {
        **jobs.get(job_id, {}),
        "status": "analyzing",
        "mode": mode,
        "progress": 5.0,
        "updated_at": time.time(),
        "started_at": time.time(),
    }

    threading.Thread(
        target=run_analysis_bg,
        args=(job_id, file_path, mode),
        daemon=True
    ).start()

    return JSONResponse(status_code=202, content={"job_id": job_id})

# Helper function to save file and start analysis in background thread
def _save_and_analyze(job_id: str, file_content: bytes, file_path: str, mode: AnalyzeMode):
    try:
        # Save file to disk
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_content)
        del file_content

        # Update job status to "analyzing" with initial progress
        jobs[job_id] = {
            **jobs.get(job_id, {}),
            "status": "analyzing",
            "mode": mode,
            "progress": 5.0,  # 5% to show progress early (prevent frontend timeout)
            "updated_at": time.time(),
            "started_at": time.time()
        }

        # Run analysis
        run_analysis_bg(job_id, file_path, mode)
    except Exception as e:
        app_logger.error(f"[SaveAndAnalyze] Failed: {e}", exc_info=True)
        jobs[job_id] = {
            **jobs.get(job_id, {}),
            "status": "error",
            "error": str(e),
            "done_at": time.time()
        }

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
    app_logger.debug(f"analyze_url endpoint called: url={url}, mode={mode}, cookies={'yes' if cookies else 'none'}")
    if mode is None:
         app_logger.warning("Missing mode in /analyze/url request. Fallback to EARLY_ACCESS.")
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
        app_logger.debug(f"_process_analyze_url called: url={url}, mode={mode}, cookies={'yes' if cookies else 'none'}")

        if cookies:
            app_logger.debug(f"Cookies file received: filename={cookies.filename}, content_type={cookies.content_type}")

            if not cookies.filename.endswith(".txt"):
                raise HTTPException(status_code=400, detail="Cookie file must be a .txt file")

            # 拡張子を抽出（ファイル名に括弧が含まれる場合の対策）
            suffix = os.path.splitext(cookies.filename)[1]
            if not suffix:
                suffix = ".txt"

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

            file_size = os.path.getsize(cookie_path)
            print(f"[DEBUG] Uploaded cookies saved: {cookie_path} ({file_size} bytes)")

            # クッキーファイルの内容を確認（最初の数行）
            try:
                with open(cookie_path, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                    print(f"[DEBUG] Cookie file preview: {first_lines}")
            except Exception as e:
                print(f"[DEBUG] Could not read cookie file preview: {e}")
        else:
            # Cookie ファイルが提供されない場合、自動的に cookies.txt を読み込む
            cookie_path = get_default_cookies_path()
            if cookie_path:
                print(f"[DEBUG] No uploaded cookies, using default cookies.txt: {cookie_path}")
                if os.path.exists(cookie_path):
                    file_size = os.path.getsize(cookie_path)
                    print(f"[DEBUG] Default cookies.txt size: {file_size} bytes")
                else:
                    print(f"[DEBUG] Default cookies.txt file does not exist!")
                    cookie_path = None
            else:
                print(f"[DEBUG] No cookies.txt found, will try without cookies")

        # Download synchronously (usually fast enough, but ideally this would be part of the job too)
        # However, for MVP, we'll keep download sync to get file_path, then offload analysis.
        # IF download is > 25s, this might still 502.
        # But moving download to BG requires passing 'url' and 'cookie_path' to BG.
        # 'run_analysis_bg' expects 'file_path'.
        # So we download here. If it times out, it times out.
        # User accepted focus on /analyze (file upload).
        # But we can try to be safe.

        print(f"[DEBUG] About to download: URL={url}, Cookie path exists: {cookie_path is not None}")
        if cookie_path:
            print(f"[DEBUG] Cookie path: {cookie_path}")

        file_path = download_youtube_audio(url, cookie_path=cookie_path)

        print(f"[DEBUG] Download completed successfully: {file_path}")
        
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
        print(f"[DEBUG] Exception in _process_analyze_url: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")
    
    finally:
        # Secure cleanup of COOKIES only. Audio file is needed for BG task.
        if cookie_path and os.path.exists(cookie_path):
            try:
                os.remove(cookie_path)
            except:
                pass


# ============================================================================
# ハイブリッド BPM 検出機能
# ============================================================================

def evaluate_tempo_prior_hybrid(bpm: float, target_range: tuple = (110, 150)) -> float:
    """
    テンポ事前確率評価（ハイブリッド用）

    Args:
        bpm: 候補 BPM
        target_range: 目標範囲 (min, max)

    Returns:
    テンポ事前確率スコア (0-1)
    """
    min_bpm, max_bpm = target_range

    # 目標範囲内であれば高いスコア
    if min_bpm <= bpm <= max_bpm:
        mean = (min_bpm + max_bpm) / 2.0
        std = (max_bpm - min_bpm) / 3.0  # より狭い分布で中速を強調
        return math.exp(-0.5 * ((bpm - mean) / std) ** 2)
    else:
        # 範囲外の場合、距離に応じて減少
        dist = min(abs(bpm - min_bpm), abs(bpm - max_bpm))
        return math.exp(-0.5 * (dist / 20.0) ** 2)  # より急激な減衰


def detect_bpm_hybrid(y: np.ndarray, sr: int, bpm_range: tuple[int, int] | None = None) -> float:
    """
    ハイブリッド BPM 検出（高速・中精度）

    Args:
        y: 音声信号
        sr: サンプリングレート
        bpm_range: BPM検索範囲 (min, max)。Noneの場合はデフォルト範囲(60-200)を使用

    Returns:
        最終 BPM
    """
    app_logger.info("[Hybrid BPM] ハイブリッド BPM 検出開始")

    # 第1段階: 初期 BPM 推定
    app_logger.info("[Hybrid BPM] Stage 1: 初期 BPM 推定中...")

    # オンセット検出
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=True,
        pre_max=3,
        post_max=3
    )

    app_logger.info(f"[Hybrid BPM] 検出されたオンセット数: {len(onset_frames)}")

    if len(onset_frames) < 2:
        return 120.0  # デフォルト

    # BPM スキャン（中速範囲に制限）
    total_frames = len(onset_env)
    num_onsets = len(onset_frames)
    onset_set = set(onset_frames)

    best_bpm = 120.0
    best_score = -1.0
    tolerance = 4

    # BPM範囲の設定: 指定があれば使用、なければデフォルト範囲
    if bpm_range is not None:
        bpm_min, bpm_max = bpm_range
        # 範囲を有効な範囲に制限
        bpm_min = max(60, bpm_min)
        bpm_max = min(240, bpm_max)
    else:
        bpm_min, bpm_max = 60, 200

    candidates = []

    BEAT_F_BETA = 0.8
    beta_sq = BEAT_F_BETA ** 2

    for c in range(bpm_min, bpm_max + 1):
        beat_period = 60.0 * sr / (c * 512)
        grid = np.arange(0, total_frames, beat_period)

        if len(grid) == 0:
            continue

        # Precision
        hits = 0
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                if (g_int + t) in onset_set:
                    hits += 1
                    break
        precision = hits / len(grid)

        # Recall
        grid_set = set()
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                grid_set.add(g_int + t)
        onset_hits = sum(1 for o in onset_frames if int(o) in grid_set)
        recall = onset_hits / max(1, num_onsets)

        # F_beta スコア
        if precision + recall > 0:
            fb_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        else:
            fb_score = 0.0

        # テンポ事前確率（強化）
        tempo_prior = evaluate_tempo_prior_hybrid(c)

        # 統合スコア（テンポ事前確率の重みを増強）
        score = fb_score * (0.5 + 0.5 * tempo_prior)
        candidates.append((c, score))

        if score > best_score:
            best_score = score
            best_bpm = float(c)

    # 上位候補をログ出力
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:10]

    app_logger.info(f"[Hybrid BPM] 上位{len(top_candidates)}候補:")
    for i, (bpm, score) in enumerate(top_candidates):
        app_logger.info(f"  {i+1}. {bpm} BPM (score={score:.3f})")

    app_logger.info(f"[Hybrid BPM] 選択された BPM: {best_bpm:.1f} (score={best_score:.3f})")

    return best_bpm


# ============================================================================
# ハイブリッド BPM 検出ここまで
# ============================================================================

# ============================================================================
# BPM調整用APIエンドポイント
# ============================================================================

class BPMAdjustRequest(BaseModel):
    """BPM調整リクエスト"""
    bars: list  # 元のバー配列
    original_bpm: float  # 元のBPM
    adjusted_bpm: float  # 調整後のBPM

class BPMAdjustResponse(BaseModel):
    """BPM調整レスポンス"""
    adjusted_bars: list  # 調整後のバー配列
    original_bpm: float  # 元のBPM
    adjusted_bpm: float  # 調整後のBPM
    adjustment_ratio: float  # 調整比率

@app.post("/bpm/adjust")
async def adjust_bpm(request: BPMAdjustRequest):
    """
    BPM調整用APIエンドポイント
    ユーザーが指定したBPMに合わせてタブ譜のタイミングを再計算
    """
    try:
        # 調整比率の計算
        if request.original_bpm <= 0:
            raise HTTPException(status_code=400, detail="Original BPM must be positive")

        adjustment_ratio = request.original_bpm / request.adjusted_bpm

        # 各バーのタイミングを調整
        adjusted_bars = []
        for bar in request.bars:
            adjusted_bar = {
                **bar,
                'start_sec': bar.get('start_sec', 0) * adjustment_ratio,
                'end_sec': bar.get('end_sec', 0) * adjustment_ratio
            }
            adjusted_bars.append(adjusted_bar)

        return BPMAdjustResponse(
            adjusted_bars=adjusted_bars,
            original_bpm=request.original_bpm,
            adjusted_bpm=request.adjusted_bpm,
            adjustment_ratio=adjustment_ratio
        )

    except Exception as e:
        print(f"[ERROR] BPM adjustment failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"BPM adjustment failed: {str(e)}")

# ============================================================================
# BPM調整用APIエンドポイントここまで
# ============================================================================

# ============================================================================
# ディープラーニングBPM検出モデルの統合
# ============================================================================

# モデルのグローバル変数
_pytorch_model = None
_pytorch_device = None
_pytorch_model_loaded = False

def load_pytorch_bpm_model():
    """PyTorch BPM検出モデルをロード"""
    global _pytorch_model, _pytorch_device, _pytorch_model_loaded

    if _pytorch_model_loaded:
        return _pytorch_model

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # デバイスの設定
        _pytorch_device = torch.device('cpu')

        # モデル定義
        class AdvancedBPMDetector(nn.Module):
            def __init__(self, input_size, num_bpm_classes=181):
                super(AdvancedBPMDetector, self).__init__()
                self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
                self.bn1 = nn.BatchNorm1d(64)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm1d(128)
                self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm1d(256)
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(256, 512)
                self.dropout1 = nn.Dropout(0.3)
                self.fc2 = nn.Linear(512, 256)
                self.dropout2 = nn.Dropout(0.3)
                self.fc3 = nn.Linear(256, num_bpm_classes)

            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.max_pool1d(x, 2)
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool1d(x, 2)
                x = F.relu(self.bn3(self.conv3(x)))
                x = F.max_pool1d(x, 2)
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout1(x)
                x = F.relu(self.fc2(x))
                x = self.dropout2(x)
                x = self.fc3(x)
                return x

        # モデルのロード
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bpm_model_augmented.pth')
        if not os.path.exists(model_path):
            app_logger.warning("[DeepLearning] BPM model file not found, using librosa-based detection")
            return None

        checkpoint = torch.load(model_path, map_location=_pytorch_device, weights_only=False)
        input_size = checkpoint.get('input_size', 1292)

        _pytorch_model = AdvancedBPMDetector(input_size)
        _pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        _pytorch_model.to(_pytorch_device)
        _pytorch_model.eval()

        _pytorch_model_loaded = True
        app_logger.info(f"[DeepLearning] BPM model loaded successfully (input_size: {input_size})")

        return _pytorch_model

    except Exception as e:
        app_logger.error(f"[DeepLearning] Failed to load model: {e}")
        return None

def detect_bpm_pytorch(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    PyTorchディープラーニングモデルによるBPM検出

    Returns:
        (検出されたBPM, 信頼度)
    """
    global _pytorch_model, _pytorch_device, _pytorch_model_loaded

    # モデルのロード
    if not _pytorch_model_loaded:
        model = load_pytorch_bpm_model()
        if model is None:
            return 120.0, 0.0  # モデルがない場合はデフォルト値

    import torch
    import torch.nn.functional as F

    # オンセット強度の計算
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # 正規化
    onset_env_normalized = (onset_env - np.mean(onset_env)) / (np.std(onset_env) + 1e-8)

    # テンソルに変換
    onset_tensor = torch.from_numpy(onset_env_normalized).float().unsqueeze(0).to(_pytorch_device)

    # 予測
    with torch.no_grad():
        output = _pytorch_model(onset_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_bpm = predicted_class + 60
        confidence = probabilities[0][predicted_class].item()

    app_logger.info(f"[DeepLearning] Detected BPM: {predicted_bpm:.1f} BPM (confidence: {confidence:.3f})")

    return predicted_bpm, confidence

def detect_bpm_with_deep_learning(y: np.ndarray, sr: int, bpm_range: tuple[int, int] | None = None) -> tuple[float, float]:
    """
    ディープラーニングモデルを使用したBPM検出
    モデルが利用可能な場合はディープラーニングを優先、そうでない場合はlibrosaを使用

    Returns:
        (検出されたBPM, 信頼度)
    """
    app_logger.info("[BPM Detection] Checking for deep learning model...")

    try:
        # ディープラーニングモデルによる検出を試みる
        dl_bpm, confidence = detect_bpm_pytorch(y, sr)

        # 信頼度が低い（<0.5）の場合はlibrosaにフォールバック
        if confidence < 0.5:
            app_logger.info(f"[BPM Detection] Deep learning confidence too low ({confidence:.3f}), using librosa fallback")
            fallback_bpm = detect_bpm_hybrid(y, sr, bpm_range)
            return fallback_bpm, confidence

        app_logger.info(f"[BPM Detection] Using deep learning result: {dl_bpm:.1f} BPM (confidence: {confidence:.3f})")
        return dl_bpm, confidence

    except Exception as e:
        app_logger.error(f"[BPM Detection] Deep learning failed: {e}, using librosa fallback")
        fallback_bpm = detect_bpm_hybrid(y, sr, bpm_range)
        return fallback_bpm, 0.0

# ============================================================================
# BeatNet BPM検出
# ============================================================================

def detect_bpm_with_beatnet(y: np.ndarray, sr: int, model_id: str = "default") -> tuple[float, float] | None:
    """
    BeatNet-Plus APIを使用したBPM検出

    Args:
        y: 音声信号
        sr: サンプリングレート
        model_id: 使用するモデルID（"default" またはカスタムモデルID）

    Returns:
        (検出されたBPM, 信頼度) または失敗時はNone
    """
    import os
    import requests
    import io
    import soundfile as sf

    # BeatNet URLの取得（環境変数またはデフォルト）
    beatnet_url = os.environ.get("BEATNET_URL", "http://localhost:8001")

    try:
        # 音声をWAVバイト列に変換
        buffer = io.BytesIO()
        sf.write(buffer, y.T if y.ndim > 1 else y, sr, format='WAV')
        buffer.seek(0)

        # BeatNet-Plus APIを呼び出す（model_idを含める）
        response = requests.post(
            f"{beatnet_url}/detect_bpm",
            files={"audio_file": ("audio.wav", buffer, "audio/wav")},
            data={"model_id": model_id},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            bpm = data.get("bpm")
            if bpm is not None:
                app_logger.info(f"[BeatNet-Plus] Detected BPM: {bpm:.2f} with model {model_id} (beats={data.get('beats_count', 0)})")
                # BeatNet-Plusは高精度なので信頼度は1.0
                return float(bpm), 1.0
            else:
                app_logger.warning("[BeatNet-Plus] No BPM in response")
                return None
        else:
            app_logger.warning(f"[BeatNet-Plus] API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        app_logger.warning(f"[BeatNet-Plus] Connection failed: {e}")
        return None
    except Exception as e:
        app_logger.error(f"[BeatNet-Plus] Detection failed: {e}", exc_info=True)
        return None


# ============================================================================
# BeatNet-Plus モデル選択
# ============================================================================

class ModelSelectionRequest(BaseModel):
    model_id: str  # "default" or custom model ID

class ModelSelectionResponse(BaseModel):
    model_id: str
    model_name: str
    status: str

@app.get("/beatnet/models")
async def list_beatnet_models():
    """
    利用可能な BeatNet-Plus モデルの一覧を取得
    """
    import os
    import requests

    beatnet_url = os.environ.get("BEATNET_URL", "http://localhost:8001")
    try:
        response = requests.get(f"{beatnet_url}/models", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        app_logger.error(f"Failed to fetch models: {e}")
        return []

@app.post("/beatnet/select-model", response_model=ModelSelectionResponse)
async def select_beatnet_model(request: ModelSelectionRequest):
    """
    アクティブな BeatNet-Plus モデルを選択
    """
    import os
    import requests

    beatnet_url = os.environ.get("BEATNET_URL", "http://localhost:8001")
    try:
        # モデルが存在するか確認
        response = requests.get(f"{beatnet_url}/models/{request.model_id}", timeout=10)
        if response.status_code == 200:
            model_data = response.json()
            return ModelSelectionResponse(
                model_id=request.model_id,
                model_name=model_data.get("name", "Unknown"),
                status="selected"
            )
        raise HTTPException(status_code=404, detail="Model not found")
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Failed to select model: {e}")
        raise HTTPException(status_code=500, detail="Failed to select model")

# ============================================================================
# BeatNet-Plus モデル選択ここまで
# ============================================================================


