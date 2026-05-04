#!/usr/bin/env python
"""
特定のYouTube URLのダウンロードをテストするスクリプト
"""
import yt_dlp
import os
import sys

TEST_URL = "https://youtu.be/0Xd8v0JXmFo?si=IQEzn4UnKO-Q_UHn"

# Node.jsをJSランタイムとして設定
js_runtimes = {"node": {}}

# 一時ファイル用の設定
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
request_id = "test_download_2"
outtmpl = os.path.join(TEMP_DIR, f"{request_id}-%(id)s.%(ext)s")

ydl_opts = {
    "format": "bestaudio[ext=m4a]/bestaudio/best",
    "noplaylist": True,
    "socket_timeout": 30,
    "retries": 3,
    "quiet": False,
    "no_warnings": False,
    "js_runtimes": js_runtimes,
    "remote_components": ["ejs:github"],
    "outtmpl": outtmpl,
}

print(f"Testing URL: {TEST_URL}")
print(f"Output template: {outtmpl}")

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("\n=== Downloading audio ===")
        info = ydl.extract_info(TEST_URL, download=True)
        filename = ydl.prepare_filename(info)

        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"\n[SUCCESS] Download successful!")
            print(f"  Filename: {filename}")
            print(f"  File size: {file_size} bytes")

            # テスト用にダウンロードしたファイルを削除
            os.remove(filename)
            print(f"  Test file removed")
        else:
            print(f"\n[ERROR] Download failed - file not found: {filename}")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")
