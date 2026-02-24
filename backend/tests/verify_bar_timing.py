"""
Verify that analyze_audio_file produces correct bar timing for tax.mp3.
Tests the ACTUAL bars returned by the function.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import analyze_audio_file

# Find an mp3 file
TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp")
mp3_files = [f for f in os.listdir(TEMP_DIR) if f.endswith(".mp3")]
if not mp3_files:
    print("ERROR: No mp3 files found in temp/")
    sys.exit(1)

# Use the first file
test_file = os.path.join(TEMP_DIR, mp3_files[0])
print(f"Testing with: {mp3_files[0]}")

# Test 1: Single chunk analysis (60s)
print("\n=== Test 1: analyze_audio_file(0-60s) ===")
result = analyze_audio_file(test_file, offset_sec=0.0, duration_limit_sec=60.0)

bpm = result["bpm"]
phase = result.get("phase_offset_sec", "NOT_PRESENT")
bars = result["bars"]

print(f"BPM: {bpm}")
print(f"Phase offset: {phase}")
print(f"Total bars: {len(bars)}")

if len(bars) >= 2:
    bar0_start = float(bars[0]["start_sec"])
    bar0_end = float(bars[0]["end_sec"])
    bar1_start = float(bars[1]["start_sec"])
    bar1_end = float(bars[1]["end_sec"])

    bar_duration = bar1_start - bar0_start
    expected_duration = (60.0 / bpm) * 4  # 4 beats per bar

    print(f"Bar 0: {bar0_start:.4f} - {bar0_end:.4f}")
    print(f"Bar 1: {bar1_start:.4f} - {bar1_end:.4f}")
    print(f"Bar duration: {bar_duration:.4f}s")
    print(f"Expected (4 beats @ BPM {bpm}): {expected_duration:.4f}s")

    if abs(bar_duration - expected_duration) < 0.01:
        print("PASS: Bar duration matches expected!")
    else:
        print(f"FAIL: Bar duration mismatch! Got {bar_duration:.4f}, expected {expected_duration:.4f}")
        print(f"  This bar duration corresponds to BPM: {240.0/bar_duration:.1f}")

# Check several bar intervals
print("\nBar interval check (first 10 bars):")
for i in range(min(10, len(bars) - 1)):
    interval = float(bars[i+1]["start_sec"]) - float(bars[i]["start_sec"])
    print(f"  bar {i} -> {i+1}: {interval:.4f}s")

# Test 2: Check run_analysis_bg flow by simulating chunk merge
print("\n=== Test 2: Simulate chunk merge ===")
import math

MAX_ANALYSIS_SEC = 120.0
CHUNK_SEC = 30.0
FIRST_CHUNK_SEC = 60.0

all_bars = []
bpm = None
segment_duration = None
chunk_idx = 0
offset = 0.0

while offset < MAX_ANALYSIS_SEC:
    if chunk_idx == 0:
        dur = min(FIRST_CHUNK_SEC, MAX_ANALYSIS_SEC - offset)
    else:
        dur = min(CHUNK_SEC, MAX_ANALYSIS_SEC - offset)

    print(f"\n--- Chunk {chunk_idx}: offset={offset:.1f}, dur={dur:.1f}, forced_bpm={bpm} ---")
    raw = analyze_audio_file(
        test_file,
        offset_sec=offset,
        duration_limit_sec=dur,
        forced_bpm=bpm
    )

    actual_dur = raw["duration_sec"]
    chunk_bars = raw["bars"]

    if bpm is None:
        bpm = raw.get("bpm", 120.0)
        seconds_per_beat = 60.0 / bpm
        segment_duration = seconds_per_beat * 2
        print(f"  Detected BPM: {bpm:.1f}")

    if len(chunk_bars) >= 2:
        d = float(chunk_bars[1]["start_sec"]) - float(chunk_bars[0]["start_sec"])
        print(f"  Chunk bar duration: {d:.4f}s")
    print(f"  Chunk bars: {len(chunk_bars)}, actual_dur: {actual_dur:.1f}s")

    for i, bar in enumerate(chunk_bars):
        bar_start = bar.get("start_sec")
        bar_end = bar.get("end_sec")

        if bar_start is not None and bar_end is not None:
            abs_start = offset + float(bar_start)
            abs_end = offset + float(bar_end)
        else:
            abs_start = offset + i * segment_duration
            abs_end = offset + (i + 1) * segment_duration

        chunk_end_abs = offset + actual_dur
        if abs_start >= chunk_end_abs:
            break
        if abs_end > chunk_end_abs:
            abs_end = chunk_end_abs

        bar_obj = {
            "bar": len(all_bars) + 1,
            "chord": bar["chord"],
            "start_sec": abs_start,
            "end_sec": abs_end,
        }
        all_bars.append(bar_obj)

    if actual_dur < (dur - 0.5) or actual_dur <= 0.1:
        offset += actual_dur
        break

    offset += dur
    chunk_idx += 1

print(f"\n=== Before unified grid: {len(all_bars)} bars ===")
if len(all_bars) >= 2:
    d = all_bars[1]["start_sec"] - all_bars[0]["start_sec"]
    print(f"Bar 0-1 interval: {d:.4f}s")
    print(f"First bar start: {all_bars[0]['start_sec']:.4f}")

# Apply unified grid
if bpm is not None and len(all_bars) > 1:
    seg_duration = (60.0 / bpm) * 4
    first_start = all_bars[0].get("start_sec", 0.0)
    print(f"\nApplying unified grid: seg={seg_duration:.4f}s, phase={first_start:.4f}s")
    for i, bar in enumerate(all_bars):
        bar["start_sec"] = round(first_start + i * seg_duration, 4)
        bar["end_sec"] = round(first_start + (i + 1) * seg_duration, 4)

print(f"\n=== After unified grid: {len(all_bars)} bars ===")
if len(all_bars) >= 2:
    d = all_bars[1]["start_sec"] - all_bars[0]["start_sec"]
    print(f"Bar 0-1 interval: {d:.4f}s")
    print(f"First bar start: {all_bars[0]['start_sec']:.4f}")
    print(f"Last bar end: {all_bars[-1]['end_sec']:.4f}")

    expected = (60.0 / bpm) * 4
    if abs(d - expected) < 0.01:
        print(f"PASS: Unified grid bar duration = {d:.4f}s (expected {expected:.4f}s)")
    else:
        print(f"FAIL: Unified grid bar duration = {d:.4f}s (expected {expected:.4f}s)")
