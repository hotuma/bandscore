#!/usr/bin/env python
"""
特定のYouTube URLをテストするスクリプト
"""
import yt_dlp
import os
import sys

TEST_URL = "https://youtu.be/LaKp04a7hAM"

# Node.jsをJSランタイムとして設定
js_runtimes = {"node": {}}

ydl_opts = {
    "format": "bestaudio[ext=m4a]/bestaudio/best",
    "noplaylist": True,
    "socket_timeout": 30,
    "retries": 3,
    "quiet": False,
    "no_warnings": False,
    "js_runtimes": js_runtimes,
    "remote_components": ["ejs:github"],
}

print(f"Testing URL: {TEST_URL}")
print(f"Options: {ydl_opts}")

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print("\n=== Extracting info (no download) ===")
        info = ydl.extract_info(TEST_URL, download=False)
        print(f"\n[SUCCESS] Info extraction successful!")
        print(f"  Title: {info.get('title')}")
        print(f"  Duration: {info.get('duration')}s")
        print(f"  Available formats: {len(info.get('formats', []))}")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All tests passed!")
