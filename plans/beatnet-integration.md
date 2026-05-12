# BeatNet統合計画

## Context

librosa.beat_trackはある程度の精度がありますが、より高精度なBeatNetの導入を決定。
Windows環境ではmadmomのインストールが難しいため、Dockerコンテナを使用します。

## Implementation Plan

### ステップ1: Dockerコンテナのビルド

**コマンド**:
```bash
docker build -f Dockerfile.beatnet -t beatnet-service .
```

**注意**: Docker Desktopを起動する必要があります。

### ステップ2: BeatNet APIサービスの作成

BeatNetをFastAPIエンドポイントとして公開するサービスを作成します。

**ファイル**: `backend/beatnet_service.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import librosa
import io
from BeatNet.BeatNet import BeatNet

app = FastAPI(title="BeatNet BPM Detection Service")

# モデルをグローバルにロード
tracker = BeatNet(model=3, mode='online', inference_model='PF')

class BPMDetectionRequest(BaseModel):
    audio_data: str  # base64エンコードされた音声データ

class BPMDetectionResponse(BaseModel):
    bpm: float
    beats: list[float]
    downbeats: list[float]

@app.post("/detect_bpm")
async def detect_bpm(audio: bytes):
    """
    音声データからBPMを検出
    """
    try:
        # 音声をロード
        y, sr = librosa.load(io.BytesIO(audio), sr=44100)

        # モノラル変換
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # BeatNetでBPM検出
        result = tracker.process(y)
        beats = result[:, 0].tolist()
        downbeats = result[result[:, 1] == 1, 0].tolist()

        # BPM計算
        if len(beats) >= 2:
            intervals = np.diff(beats)
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval
        else:
            raise HTTPException(status_code=400, detail="Could not detect enough beats")

        return {
            "bpm": float(bpm),
            "beats": beats,
            "downbeats": downbeats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "BeatNet"}
```

### ステップ3: DockerイメージをAPIサービス用に更新

**ファイル**: `Dockerfile.beatnet`（更新）

```dockerfile
FROM python:3.11-slim  # Python 3.11はmadmomとの互換性が良い

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# Pythonパッケージのインストール
RUN pip install --no-cache-dir cython==0.29.37
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.11.4 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    torch==2.2.0 \
    madmom==0.16.1 \
    BeatNet==1.1.1 \
    fastapi==0.104.1 \
    uvicorn==0.24.0

# APIサービスのコピー
COPY backend/beatnet_service.py /app/

# ポート8000を公開
EXPOSE 8000

# サービス起動
CMD ["uvicorn", "beatnet_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

### ステップ4: バックエンドからBeatNet APIを呼び出す

**ファイル**: `backend/main.py`

BeatNetを優先的に使用するようにBPM検出ロジックを更新：

```python
def detect_bpm_with_beatnet(y: np.ndarray, sr: int) -> tuple[float, float] | None:
    """
    BeatNet APIを使用したBPM検出
    """
    try:
        import requests
        import io

        # 音声をバイト列に変換
        buffer = io.BytesIO()
        librosa.output.write_wav(buffer, y, sr)
        buffer.seek(0)

        # BeatNet APIを呼び出す
        response = requests.post(
            "http://beatnet-service:8000/detect_bpm",
            files={"audio": ("audio.wav", buffer, "audio/wav")},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data["bpm"], 1.0  # 信頼度は常に1.0（BeatNetは高精度）
        else:
            app_logger.warning(f"BeatNet API error: {response.status_code}")
            return None

    except Exception as e:
        app_logger.warning(f"BeatNet detection failed: {e}")
        return None
```

### ステップ5: docker-compose.ymlの作成

バックエンドとBeatNetサービスを連携させる。

```yaml
version: '3.8'

services:
  beatnet-service:
    build:
      context: .
      dockerfile: Dockerfile.beatnet
    ports:
      - "8001:8000"
    networks:
      - bandscore-network

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - BEATNET_URL=http://beatnet-service:8000
    depends_on:
      - beatnet-service
    networks:
      - bandscore-network

networks:
  bandscore-network:
    driver: bridge
```

## Verification

1. BeatNetサービス単体のテスト
2. UVERworld「just Melody」でBPM検出テスト（期待: 125 BPM）
3. 既存の曲（ガソリン0812.m4a）でテスト（期待: 162 BPM）

## Implementation Steps

### 1. BeatNet APIサービスの作成

**新規ファイル**: `backend/beatnet_service.py`

FastAPIを使用してBeatNetをHTTP APIとして公開するサービスを作成します。

### 2. Dockerfile.beatnetの更新

**更新ファイル**: `Dockerfile.beatnet`

既存のDockerfileはテスト用なので、APIサービス用に更新します。

### 3. docker-compose.ymlの作成

**新規ファイル**: `docker-compose.yml`

バックエンドとBeatNetサービスを連携させます。

### 4. バックエンドmain.pyの更新

**更新ファイル**: `backend/main.py`

BeatNet APIを呼び出す関数を追加し、BPM検出ロジックを更新します。

### 5. テストと検証

- BeatNetサービス単体のテスト
- UVERworld「just Melody」でBPM検出テスト（期待: 125 BPM）
- ガソリン0812.m4aでテスト（期待: 162 BPM）

## Critical Files

- `Dockerfile.beatnet` - BeatNetサービスのDockerイメージ（更新）
- `backend/beatnet_service.py` - BeatNet APIサービス（新規作成）
- `backend/main.py` - BeatNet呼び出しロジックの追加（更新）
- `docker-compose.yml` - サービス連携用（新規作成）

## Notes

- Docker Desktopを起動する必要があります
- 最初のビルドには時間がかかります（5-10分程度）
- BeatNetはPyTorchを使用するため、GPUがあると高速化できます
- Windows環境でもDockerコンテナで動作するため、madmomのインストール問題を回避できます
