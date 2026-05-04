"""Debug yt-dlp download to identify exact error"""
import yt_dlp
from yt_dlp.utils import DownloadError
import os
import sys

URL = "https://youtu.be/q7nFt0nkU3Y?si=WCYSGTh88EizQNiV"

# Check for cookies.txt
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cookies_path = os.path.join(project_root, "cookies.txt")
print(f"Cookies path: {cookies_path}")
print(f"Cookies exists: {os.path.exists(cookies_path)}")

base_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_bin_dir = os.path.join(base_dir, "bin")
ffmpeg_exe = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
ffmpeg_location = ffmpeg_bin_dir if os.path.exists(ffmpeg_exe) else None

# Test 1: Info extraction only (no download)
print("\n=== Test 1: Extract info (no download) ===")
try:
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "geo_bypass": True,
        "ffmpeg_location": ffmpeg_location,
    }
    if os.path.exists(cookies_path):
        opts["cookiefile"] = cookies_path
        print(f"Using cookies: {cookies_path}")

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(URL, download=False)
        print(f"Title: {info.get('title')}")
        print(f"Duration: {info.get('duration')}s")
        print(f"Format: {info.get('format')}")
        print(f"Ext: {info.get('ext')}")
        print(f"Formats available: {len(info.get('formats', []))}")
        for f in info.get('formats', [])[-5:]:
            print(f"  - {f.get('format_id')}: {f.get('ext')} {f.get('acodec')} {f.get('filesize', 'unknown')} bytes")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 2: Download with verbose logging
print("\n=== Test 2: Download attempt ===")
try:
    outtmpl = os.path.join("temp", "debug-test.%(ext)s")
    opts2 = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "geo_bypass": True,
        "nopart": True,
        "overwrites": True,
        "outtmpl": outtmpl,
        "quiet": False,
        "no_warnings": False,
        "verbose": True,
        "ffmpeg_location": ffmpeg_location,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        },
    }
    if os.path.exists(cookies_path):
        opts2["cookiefile"] = cookies_path

    with yt_dlp.YoutubeDL(opts2) as ydl:
        info = ydl.extract_info(URL, download=True)
        filename = ydl.prepare_filename(info)
        print(f"Output file: {filename}")
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"File size: {size} bytes")
            if size == 0:
                print("WARNING: File is 0 bytes!")
            else:
                print("SUCCESS: File downloaded successfully")
            # Clean up
            os.remove(filename)
        else:
            print("ERROR: File does not exist")
except DownloadError as e:
    print(f"DownloadError: {e}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
