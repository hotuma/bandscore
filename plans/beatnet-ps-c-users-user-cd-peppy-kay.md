# BeatNet-Plus ファインチューニング機能実装計画

## Context

現在の BeatNet は事前学習済みの固定モデルを使用しており、解析音源を増やすだけでは精度は向上しません。BeatNet-Plus に切り替え、ユーザーが独自のデータでファインチューニングを行えるようにすることで、特定のジャンルやスタイルに特化した高精度な BPM 検出を実現します。

## 実装方針

- **BeatNet-Plus への切り替え**: トレーニング機能をサポート
- **Web UI から実行**: `/lab/training` ページで管理
- **データセット管理**: 音声ファイルと `.beats` アノテーションファイルで構成

## 実装計画

### Phase 1: Docker インフラ

**新規ファイル**: `Dockerfile.beatnetplus`

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential libsndfile1 ffmpeg portaudio19-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.2.0 torchaudio==2.2.0
RUN pip install --no-cache-dir numpy scipy librosa soundfile

# BeatNet-Plus をインストール
RUN git clone https://github.com/mjhydri/BeatNet-Plus.git /tmp/beatnetplus && \
    cd /tmp/beatnetplus && pip install -e . && cd / && rm -rf /tmp/beatnetplus

RUN pip install --no-cache-dir fastapi uvicorn python-multipart pyyaml

COPY backend/beatnetplus_service.py /app/
RUN mkdir -p /app/models /app/training_data /app/training_jobs

EXPOSE 8000
CMD ["uvicorn", "beatnetplus_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

**修正**: `docker-compose.yml`

- `beatnet-service` を `beatnet-plus-service` に置き換え
- ボリュームを追加: `./beatnetplus_models:/app/models`, `./beatnetplus_training:/app/training_data`, `./beatnetplus_jobs:/app/training_jobs`

### Phase 2: BeatNet-Plus サービス

**新規ファイル**: `backend/beatnetplus_service.py`

主な機能:
- **BPM 検出**: `/detect_bpm` - モデル選択対応
- **モデル管理**: `/models` - 一覧、詳細、削除
- **データセット管理**: `/datasets` - 作成、一覧、ファイルアップロード
- **トレーニング管理**: `/training/start`, `/training/{job_id}`, `/training/jobs`

### Phase 3: バックエンド統合

**修正**: `backend/main.py`

追加するエンドポイント:
- `GET /beatnet/models` - 利用可能なモデル一覧
- `POST /beatnet/select-model` - アクティブモデルの選択
- `detect_bpm_with_beatnet()` 関数に `model_id` パラメータを追加

### Phase 4: フロントエンド実装

**新規ファイル**:
- `frontend/lib/beatnetApi.ts` - API クライアント関数
- `frontend/app/lab/training/page.tsx` - トレーニング管理 UI
- `frontend/components/BeatsEditor.tsx` - ビートアノテーションエディタ

**修正**: `frontend/app/lab/layout.tsx` - ナビゲーションに「Training」リンクを追加

### Phase 5: データ管理

**ディレクトリ構造**:
```
beatnetplus_models/
├── generic/best_model_weights.pt    # デフォルトモデル
├── finetuned_<job_id>.pt            # カスタムモデル
└── finetuned_<job_id>.json          # モデルメタデータ

beatnetplus_training/
└── <dataset_id>/
    ├── metadata.json
    ├── song1.wav
    └── song1.beats
```

**.beats ファイル形式**:
```
# <時間(秒)> <ビートタイプ(0=通常,1=ダウンビート)>
0.123456 1
0.483542 0
```

### Phase 6: 進捗追跡

- トレーニングジョブの進捗を 3 秒ごとにポーリング
- 進捗マイルストーン: キュー(0-5%) → データ読込(5-10%) → トレーニング(10-90%) → 保存(90-95%) → 完了(100%)

## 修正するファイル

| ファイル | 操作 |
|---------|------|
| `Dockerfile.beatnetplus` | 新規作成 |
| `docker-compose.yml` | 修正 |
| `backend/beatnetplus_service.py` | 新規作成 |
| `backend/main.py` | 修正 |
| `frontend/lib/beatnetApi.ts` | 新規作成 |
| `frontend/app/lab/training/page.tsx` | 新規作成 |
| `frontend/components/BeatsEditor.tsx` | 新規作成 |
| `frontend/app/lab/layout.tsx` | 修正 |

## 検証方法

1. **Docker ビルド**: `docker-compose build beatnet-plus-service`
2. **サービス起動**: `docker-compose up -d`
3. **ヘルスチェック**: `curl http://localhost:8001/health`
4. **モデル一覧**: `curl http://localhost:8001/models`
5. **データセット作成**: Web UI から作成してファイルをアップロード
6. **トレーニング実行**: Web UI からトレーニングを開始
7. **モデル選択**: トレーニング完了後、新しいモデルで BPM 検出をテスト

## 依存関係

- BeatNet-Plus GitHub リポジトリの可用性
- PyTorch と CUDA（GPU 使用の場合）
- 十分なディスク容量（モデルとトレーニングデータ用）
