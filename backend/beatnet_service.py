"""
BeatNet BPM Detection Service

FastAPIエンドポイントとしてBeatNetを提供するサービス。
Dockerコンテナ内で動作し、BPM検出APIを提供します。
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import librosa
import io
import logging
from typing import Optional

# Python 3.11 で collections.MutableSequence が削除された問題を修正
import collections.abc
if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'Callable'):
    collections.Callable = collections.abc.Callable

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BeatNet BPM Detection Service",
    description="BeatNetを使用した高精度BPM検出API",
    version="1.0.0"
)

# BeatNetモデルのロード（グローバル変数としてキャッシュ）
_tracker = None

def get_tracker():
    """BeatNetモデルを遅延ロード"""
    global _tracker
    if _tracker is None:
        try:
            from BeatNet.BeatNet import BeatNet
            logger.info("Loading BeatNet model...")
            # model=3: Rock_corpus（ロック/ポップス音楽に特化）
            # mode='online': リアルタイム処理
            # inference_model='PF': Particle Filter（madmom不要）
            _tracker = BeatNet(model=3, mode='online', inference_model='PF')
            logger.info("BeatNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BeatNet model: {e}")
            raise
    return _tracker

class BPMDetectionResponse(BaseModel):
    """BPM検出レスポンス"""
    bpm: float
    beats_count: int
    downbeats_count: int
    beats: list[float]
    downbeats: list[float]

class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    status: str
    model: str
    ready: bool

@app.get("/health", response_model=HealthResponse)
async def health():
    """ヘルスチェックエンドポイント"""
    try:
        tracker = get_tracker()
        return HealthResponse(
            status="healthy",
            model="BeatNet (Rock_corpus)",
            ready=True
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model="BeatNet",
            ready=False
        )

@app.post("/detect_bpm", response_model=BPMDetectionResponse)
async def detect_bpm(audio_file: UploadFile = File(...)):
    """
    音声ファイルからBPMを検出

    Args:
        audio_file: アップロードされた音声ファイル

    Returns:
        BPMDetectionResponse: BPM検出結果
    """
    try:
        # 音声ファイルの読み込み
        contents = await audio_file.read()
        y, sr = librosa.load(io.BytesIO(contents), sr=44100)

        logger.info(f"Audio loaded: duration={len(y)/sr:.1f}s, sr={sr}Hz")

        # モノラル変換
        if y.ndim > 1:
            y = np.mean(y, axis=1)
            logger.info("Converted stereo to mono")

        # BeatNetでBPM検出
        tracker = get_tracker()
        logger.info("Running BeatNet detection...")

        result = tracker.process(y)
        beats = result[:, 0]
        downbeats = result[result[:, 1] == 1, 0]

        # BPM計算
        if len(beats) >= 2:
            intervals = np.diff(beats)
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval

            logger.info(f"BeatNet detection complete: bpm={bpm:.2f}, beats={len(beats)}, downbeats={len(downbeats)}")

            return BPMDetectionResponse(
                bpm=float(bpm),
                beats_count=len(beats),
                downbeats_count=len(downbeats),
                beats=beats.tolist(),
                downbeats=downbeats.tolist()
            )
        else:
            logger.error("Not enough beats detected")
            raise HTTPException(
                status_code=400,
                detail="Could not detect enough beats from the audio"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BPM detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "service": "BeatNet BPM Detection Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect_bpm": "/detect_bpm (POST, multipart/form-data)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
