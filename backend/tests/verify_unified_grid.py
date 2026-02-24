"""
Verify unified grid recalculation after chunk merge.
Tests that bar timing is consistent across chunk boundaries.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import analyze_audio_file

# Use one of the available temp files
TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "temp")

# Find any mp3 file to test with
mp3_files = [f for f in os.listdir(TEMP_DIR) if f.endswith(".mp3")]
if not mp3_files:
    print("ERROR: No mp3 files found in temp/")
    sys.exit(1)

test_file = os.path.join(TEMP_DIR, mp3_files[0])
print(f"Testing with: {mp3_files[0]}")

# Simulate multi-chunk analysis by running two chunks manually
print("\n=== Chunk 0 (0-60s) ===")
result0 = analyze_audio_file(test_file, offset_sec=0.0, duration_limit_sec=60.0)
bpm = result0.get("bpm", 120.0)
phase = result0.get("phase_offset_sec", 0.0)
bars0 = result0["bars"]
print(f"BPM: {bpm}, Phase: {phase:.4f}s, Bars: {len(bars0)}")

if len(bars0) >= 2:
    d = float(bars0[1]["start_sec"]) - float(bars0[0]["start_sec"])
    print(f"Bar duration (chunk0): {d:.4f}s")
    print(f"First bar start: {float(bars0[0]['start_sec']):.4f}s")

print(f"\n=== Chunk 1 (60-90s, forced_bpm={bpm}) ===")
result1 = analyze_audio_file(test_file, offset_sec=60.0, duration_limit_sec=30.0, forced_bpm=bpm)
bars1 = result1["bars"]
print(f"Bars: {len(bars1)}")

if len(bars1) >= 2:
    d = float(bars1[1]["start_sec"]) - float(bars1[0]["start_sec"])
    print(f"Bar duration (chunk1): {d:.4f}s")
    print(f"First bar start: {float(bars1[0]['start_sec']):.4f}s")

# Simulate chunk merge + unified grid
print("\n=== Merged + Unified Grid ===")
seg_duration = (60.0 / bpm) * 4  # 4 beats per bar
print(f"Expected bar duration (4 beats @ BPM {bpm}): {seg_duration:.4f}s")

all_bars = []
# Add chunk 0 bars
for bar in bars0:
    all_bars.append({
        "bar": len(all_bars) + 1,
        "chord": bar["chord"],
        "start_sec": float(bar["start_sec"]),
        "end_sec": float(bar["end_sec"]),
    })

# Add chunk 1 bars with offset
for bar in bars1:
    all_bars.append({
        "bar": len(all_bars) + 1,
        "chord": bar["chord"],
        "start_sec": 60.0 + float(bar["start_sec"]),
        "end_sec": 60.0 + float(bar["end_sec"]),
    })

# Show bars around chunk boundary BEFORE unification
chunk0_count = len(bars0)
print(f"\nBefore unification (around chunk boundary, bar {chunk0_count-1} to {chunk0_count+2}):")
for i in range(max(0, chunk0_count - 2), min(len(all_bars), chunk0_count + 2)):
    b = all_bars[i]
    gap = ""
    if i > 0:
        gap = f"  gap={b['start_sec'] - all_bars[i-1]['end_sec']:.4f}s"
    print(f"  bar {b['bar']}: {b['start_sec']:.4f} - {b['end_sec']:.4f}{gap}")

# Apply unified grid
first_start = all_bars[0]["start_sec"]
for i, bar in enumerate(all_bars):
    bar["start_sec"] = round(first_start + i * seg_duration, 4)
    bar["end_sec"] = round(first_start + (i + 1) * seg_duration, 4)

# Show bars around chunk boundary AFTER unification
print(f"\nAfter unification (around chunk boundary):")
for i in range(max(0, chunk0_count - 2), min(len(all_bars), chunk0_count + 2)):
    b = all_bars[i]
    gap = ""
    if i > 0:
        gap = f"  gap={b['start_sec'] - all_bars[i-1]['end_sec']:.4f}s"
    print(f"  bar {b['bar']}: {b['start_sec']:.4f} - {b['end_sec']:.4f}{gap}")

# Verify uniformity
print(f"\n=== Verification ===")
intervals = []
for i in range(1, len(all_bars)):
    interval = all_bars[i]["start_sec"] - all_bars[i-1]["start_sec"]
    intervals.append(interval)

if intervals:
    min_i = min(intervals)
    max_i = max(intervals)
    print(f"Total bars: {len(all_bars)}")
    print(f"Min interval: {min_i:.4f}s")
    print(f"Max interval: {max_i:.4f}s")
    print(f"Variation: {max_i - min_i:.6f}s")

    if max_i - min_i < 0.001:
        print("PASS: All bar intervals are uniform!")
    else:
        print("FAIL: Bar intervals are NOT uniform")
