#!/usr/bin/env python
"""
バックエンドの_process_analyze_url関数をテストするスクリプト
"""
import sys
import os
import asyncio

# バックエンドディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import _process_analyze_url
from main import AnalyzeMode

TEST_URL = "https://youtu.be/LaKp04a7hAM"

async def test_process_analyze_url():
    print(f"Testing _process_analyze_url with URL: {TEST_URL}")
    print(f"Mode: PREVIEW")

    try:
        result = await _process_analyze_url(TEST_URL, AnalyzeMode.PREVIEW, None)
        print(f"\n[SUCCESS] Result: {result}")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_process_analyze_url())
