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
# Install setuptools first to ensure pkg_resources is available
RUN pip install --no-cache-dir --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 一時ファイル用ディレクトリを作成
RUN mkdir -p temp

# ポート8000を公開
EXPOSE 8000

# サーバー起動コマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]