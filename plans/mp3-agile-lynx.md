# MP3解析の空結果問題修正計画

## Context

クラウド環境でMP3ファイルをアップロードすると、解析結果が空の配列（`bars: []`）で返される問題が発生しています。

**問題の原因:**
- ローカル環境では `backend/bin/ffmpeg.exe` が存在し、MP3解析が正常に動作
- クラウド環境（Linux）では Windows用のffmpeg.exeは動作しない
- クラウド環境でFFmpegが正しくインストール・設定されていない可能性
- `librosa.load` 周りに適切なエラーハンドリングがないため、デコード失敗が静かに処理されている

## 修正内容

### 1. librosa.load のエラーハンドリング強化
**ファイル:** `backend/main.py` (行1989付近)

```python
# 現在のコード
y, sr = librosa.load(file_path, sr=22050, mono=True, offset=float(offset_sec), duration=float(load_dur))

# 修正後
try:
    y, sr = librosa.load(file_path, sr=22050, mono=True, offset=float(offset_sec), duration=float(load_dur))
    app_logger.info(f"mem after load: {mem_mb():.1f} MB")
    _progress(20) # Loaded

    app_logger.debug(f"Audio loaded. Size: {y.size}, SR: {sr}")
    if y.size == 0:
        raise ValueError("Audio file is empty or unreadable")
except Exception as e:
    app_logger.error(f"Failed to load audio file {file_path}: {e}")
    # FFmpeg関連のエラーを検出
    if "ffmpeg" in str(e).lower() or "decoder" in str(e).lower():
        raise RuntimeError(
            "Audio decoding failed. The server may be missing FFmpeg. "
            "Please contact support or try a WAV file instead."
        ) from e
    raise RuntimeError(f"Audio loading failed: {e}") from e
```

### 2. FFmpeg可用性チェックエンドポイントを追加
**ファイル:** `backend/main.py` (ヘルスチェックセクションに追加)

```python
@app.get("/health/ffmpeg")
def check_ffmpeg():
    """FFmpegが利用可能かチェックするエンドポイント"""
    try:
        # librosaがFFmpegを使用できるかテスト
        import tempfile
        import numpy as np
        import soundfile as sf

        # テスト用の短い音声を作成
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            test_file = f.name
            # 1秒のサイレント音声をWAVで保存
            test_audio = np.zeros(22050)
            sf.write(test_file.replace(".mp3", ".wav"), test_audio, 22050)

        # librosaでMP3として読み込みを試みる（FFmpegが必要）
        y, sr = librosa.load(test_file, sr=22050, duration=0.1)
        os.unlink(test_file.replace(".mp3", ".wav"))

        return {
            "status": "ok",
            "ffmpeg_available": True,
            "message": "FFmpeg is properly configured for MP3 decoding"
        }
    except Exception as e:
        return {
            "status": "error",
            "ffmpeg_available": False,
            "message": f"FFmpeg not available or misconfigured: {e}"
        }
```

### 3. バックエンド用Dockerfileの作成
**新規ファイル:** `backend/Dockerfile`

```dockerfile
FROM python:3.12.7-slim

WORKDIR /app

# システム依存関係をインストール（FFmpeg含む）
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 一時ファイル用ディレクトリを作成
RUN mkdir -p temp

# ポート8000を公開
EXPOSE 8000

# サーバー起動コマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. 分析開始前のFFmpegチェック
**ファイル:** `backend/main.py` (run_analysis_bg関数に追加)

```python
def run_analysis_bg(job_id: str, file_path: str, mode: AnalyzeMode = AnalyzeMode.PREVIEW, source: str = "upload"):
    app_logger.info(f"[run_analysis_bg] Starting analysis job {job_id}, mode={mode}, source={source}")
    cleanup_jobs()

    # MP3ファイルの場合、FFmpegが利用可能かチェック
    if file_path.lower().endswith('.mp3'):
        try:
            import soundfile as sf
            import tempfile
            import numpy as np

            # 簡易チェック
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                test_file = f.name
            test_audio = np.zeros(22050)
            sf.write(test_file.replace(".mp3", ".wav"), test_audio, 22050)

            y, sr = librosa.load(test_file.replace(".mp3", ".wav"), sr=22050, duration=0.1)
            os.unlink(test_file.replace(".mp3", ".wav"))
        except Exception as e:
            app_logger.error(f"FFmpeg check failed for MP3 processing: {e}")
            jobs[job_id] = {
                **jobs.get(job_id, {}),
                "status": "error",
                "error": "MP3 decoding is not available on this server. Please use WAV format or contact support."
            }
            return

    # ... 既存のコード ...
```

## 検証方法

1. **ローカルでのテスト**:
   ```bash
   cd backend
   uvicorn main:app --reload
   curl http://localhost:8000/health/ffmpeg
   ```

2. **MP3アップロードテスト**:
   - 既存のMP3ファイル（例: `tax.mp3`）で解析を試行
   - エラーが発生する場合、適切なエラーメッセージが返されることを確認

3. **クラウドデプロイ後の検証**:
   - `/health/ffmpeg` エンドポイントにアクセスしてFFmpeg可用性を確認
   - MP3ファイルをアップロードして、空の結果ではなく適切なエラーまたは正常な結果が返ることを確認

4. **ログ確認**:
   - クラウド環境のログでFFmpeg関連のエラーがないか確認
   - 音声読み込み時の詳細なログが出力されているか確認

## 関連ファイル

- `backend/main.py` - メインの解析ロジックとエラーハンドリング
- `backend/requirements.txt` - Python依存関係（既存）
- `backend/Dockerfile` - 新規作成（デプロイ用）
- `runtime.txt` - Pythonバージョン指定（既存）