"""
Chord detection accuracy verification script.
Tests the analysis pipeline with available audio files.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import (
    analyze_audio_file,
    build_chord_templates,
    CHORD_LABELS,
    CHORD_TO_TAB,
    get_diatonic_chords_for_key,
)
import time
import json
from collections import Counter
from itertools import groupby
import math

# ========================================
# Phase 1: Core Infrastructure
# ========================================

def load_json_analysis(json_path: str) -> dict:
    """
    Load and validate JSON analysis result.

    Args:
        json_path: Path to JSON file containing analysis results

    Returns:
        dict with keys: bpm, key, bars, duration_sec, etc.

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        ValueError: If required keys are missing
    """
    # Check file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate required keys
    required_keys = ['bpm', 'key', 'bars', 'duration_sec']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys in JSON: {missing_keys}")

    # Validate bars structure
    if not isinstance(data['bars'], list):
        raise ValueError("'bars' must be a list")

    for i, bar in enumerate(data['bars']):
        required_bar_keys = ['chord', 'start_sec', 'end_sec']
        missing_bar_keys = [k for k in required_bar_keys if k not in bar]
        if missing_bar_keys:
            raise ValueError(f"Bar {i} missing required keys: {missing_bar_keys}")

    return data

def parse_key(key_str: str) -> tuple[str, str]:
    """
    Parse key string into (root, mode).

    Args:
        key_str: Key notation (e.g., "Fm", "C", "G#", "A#m")

    Returns:
        Tuple of (root_name, mode) where mode is "" for major or "m" for minor

    Examples:
        "Fm" → ("F", "m")
        "C" → ("C", "")
        "G#" → ("G#", "")
        "A#m" → ("A#", "m")
    """
    key_str = key_str.strip()

    # Check if it ends with 'm' (minor)
    if key_str.endswith('m'):
        # Handle cases like "Fm", "A#m"
        root = key_str[:-1]
        mode = "m"
    else:
        # Major key (no suffix or ignored suffixes like "maj7")
        root = key_str
        mode = ""

    return (root, mode)

# ========================================
# Phase 2: Metrics Computation
# ========================================

def compute_stagnation_metrics(chords: list[str]) -> dict:
    """
    Analyze chord stagnation patterns.

    Args:
        chords: List of chord names in sequence

    Returns:
        {
            'max_consecutive': int,  # Longest consecutive run
            'avg_consecutive': float,  # Average run length
            'stagnation_runs': list[tuple],  # [(chord, count, start_bar)]
            'exceeds_threshold': bool,  # True if any run > 6 bars
            'total_runs': int  # Total number of distinct runs
        }
    """
    if not chords:
        return {
            'max_consecutive': 0,
            'avg_consecutive': 0.0,
            'stagnation_runs': [],
            'exceeds_threshold': False,
            'total_runs': 0
        }

    # Track consecutive runs
    runs = []
    for chord, group in groupby(chords):
        group_list = list(group)
        count = len(group_list)
        start_bar = len(runs) > 0 and sum(r[1] for r in runs) or 0
        runs.append((chord, count, start_bar + 1))  # +1 for 1-indexed bars

    # Calculate metrics
    max_consecutive = max(r[1] for r in runs)
    avg_consecutive = sum(r[1] for r in runs) / len(runs)
    exceeds_threshold = any(r[1] > 6 for r in runs)

    return {
        'max_consecutive': max_consecutive,
        'avg_consecutive': avg_consecutive,
        'stagnation_runs': runs,
        'exceeds_threshold': exceeds_threshold,
        'total_runs': len(runs)
    }

def compute_diatonic_metrics(chords: list[str], key_root: str, key_mode: str) -> dict:
    """
    Calculate diatonic compliance rate.

    Args:
        chords: List of chord names in sequence
        key_root: Root note of the key (e.g., "F", "C#")
        key_mode: Mode ("" for major, "m" for minor)

    Returns:
        {
            'diatonic_rate': float,  # 0.0 to 1.0
            'diatonic_chords': list[str],  # Chords used that are diatonic
            'non_diatonic_chords': list[str],  # Chords used that are NOT diatonic
            'expected_diatonic': list[str],  # All diatonic chords for this key
            'diatonic_count': int,
            'non_diatonic_count': int
        }
    """
    if not chords:
        return {
            'diatonic_rate': 0.0,
            'diatonic_chords': [],
            'non_diatonic_chords': [],
            'expected_diatonic': [],
            'diatonic_count': 0,
            'non_diatonic_count': 0
        }

    # Get expected diatonic chords for this key
    expected_diatonic = get_diatonic_chords_for_key(key_root, key_mode)

    # Count diatonic vs non-diatonic bars
    diatonic_count = sum(1 for c in chords if c in expected_diatonic)
    non_diatonic_count = len(chords) - diatonic_count

    # Get unique chords used
    unique_chords = set(chords)
    diatonic_chords = sorted([c for c in unique_chords if c in expected_diatonic])
    non_diatonic_chords = sorted([c for c in unique_chords if c not in expected_diatonic])

    # Calculate rate
    diatonic_rate = diatonic_count / len(chords) if chords else 0.0

    return {
        'diatonic_rate': diatonic_rate,
        'diatonic_chords': diatonic_chords,
        'non_diatonic_chords': non_diatonic_chords,
        'expected_diatonic': expected_diatonic,
        'diatonic_count': diatonic_count,
        'non_diatonic_count': non_diatonic_count
    }

def compute_diversity_metrics(chords: list[str]) -> dict:
    """
    Calculate chord diversity statistics.

    Args:
        chords: List of chord names in sequence

    Returns:
        {
            'unique_count': int,
            'distribution': dict[str, int],  # {chord: count}
            'simpson_index': float,  # 0.0 to 1.0, higher = more diverse
            'entropy': float,  # Shannon entropy
            'most_common': list[tuple]  # [(chord, count)] top 5
        }
    """
    if not chords:
        return {
            'unique_count': 0,
            'distribution': {},
            'simpson_index': 0.0,
            'entropy': 0.0,
            'most_common': []
        }

    # Count chord occurrences
    counter = Counter(chords)
    total = len(chords)

    # Calculate Simpson Diversity Index: D = 1 - Σ(ni/N)²
    simpson_index = 1.0 - sum((count / total) ** 2 for count in counter.values())

    # Calculate Shannon Entropy: H = -Σ(pi * log2(pi))
    entropy = -sum((count / total) * math.log2(count / total) for count in counter.values())

    # Get top 5 most common
    most_common = counter.most_common(5)

    return {
        'unique_count': len(counter),
        'distribution': dict(counter),
        'simpson_index': simpson_index,
        'entropy': entropy,
        'most_common': most_common
    }

def compute_timing_metrics(bars: list[dict], bpm: float) -> dict:
    """
    Verify bar timing consistency.

    Args:
        bars: List of bar dicts with 'start_sec' and 'end_sec'
        bpm: Detected BPM

    Returns:
        {
            'expected_duration': float,  # Expected bar duration from BPM
            'mean_duration': float,
            'std_duration': float,  # Standard deviation
            'min_duration': float,
            'max_duration': float,
            'anomalies': list[tuple],  # [(bar_idx, duration, deviation)]
            'timing_ok': bool  # True if std < 0.05s
        }
    """
    if not bars or bpm <= 0:
        return {
            'expected_duration': 0.0,
            'mean_duration': 0.0,
            'std_duration': 0.0,
            'min_duration': 0.0,
            'max_duration': 0.0,
            'anomalies': [],
            'timing_ok': False
        }

    # Expected bar duration: (60/BPM) * 4
    # Assuming 4/4 time, 2 beats per segment, 2 segments per bar
    expected_duration = (60.0 / bpm) * 4

    # Calculate actual durations
    durations = [bar['end_sec'] - bar['start_sec'] for bar in bars]

    # Calculate statistics
    mean_duration = sum(durations) / len(durations)
    variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
    std_duration = math.sqrt(variance)
    min_duration = min(durations)
    max_duration = max(durations)

    # Find anomalies: bars where |duration - expected| > 0.1s
    anomalies = []
    for i, duration in enumerate(durations):
        deviation = duration - expected_duration
        if abs(deviation) > 0.1:
            # Bar index is 1-indexed in output
            anomalies.append((i + 1, duration, deviation))

    # Timing is OK if std < 0.05s
    timing_ok = std_duration < 0.05

    return {
        'expected_duration': expected_duration,
        'mean_duration': mean_duration,
        'std_duration': std_duration,
        'min_duration': min_duration,
        'max_duration': max_duration,
        'anomalies': anomalies,
        'timing_ok': timing_ok
    }

# ========================================
# Phase 3: Visualization
# ========================================

def visualize_chord_progression(bars: list[dict], width: int = 120) -> str:
    """
    Generate ASCII timeline of chord progression.

    Args:
        bars: List of bar dicts with 'chord' key
        width: Display width in characters

    Returns:
        Multi-line string showing chord progression

    Format:
        Bar 1-10:   [Fm7 ][Fm7 ][C#  ][Fm7 ][Fm7 ][Fm7 ][Fm7 ][C#  ][C#  ][C#  ]
        Bar 11-20:  [C#  ][A#m7][A#m7][C#  ][C#  ][C#  ][C#  ][Fm  ][Fm  ][C#  ]
    """
    if not bars:
        return "No bars to display"

    lines = []
    bars_per_row = 10
    cell_width = 5  # [Fm7 ] = 5 chars

    for row_start in range(0, len(bars), bars_per_row):
        row_end = min(row_start + bars_per_row, len(bars))
        row_bars = bars[row_start:row_end]

        # Format bar range label
        label = f"Bar {row_start + 1}-{row_end}:".ljust(12)

        # Format chord cells
        cells = []
        for bar in row_bars:
            chord = bar['chord']
            # Truncate or pad to fit cell width (5 chars)
            if len(chord) > cell_width - 2:
                chord_display = chord[:cell_width - 2]
            else:
                chord_display = chord.ljust(cell_width - 2)
            cells.append(f"[{chord_display}]")

        line = label + "".join(cells)
        lines.append(line)

    return "\n".join(lines)

def visualize_stagnation_heatmap(chords: list[str], width: int = 80) -> str:
    """
    Show stagnation intensity as ASCII heatmap.

    Args:
        chords: List of chord names in sequence
        width: Display width in characters

    Returns:
        Multi-line string showing stagnation heatmap

    Format:
        Bar 1-80:   ##....................##########......
        Legend: # = 7+ bars, = = 4-6 bars, . = 1-3 bars
    """
    if not chords:
        return "No chords to display"

    # Calculate local stagnation count for each bar
    stagnation_levels = []
    for i, chord in enumerate(chords):
        # Count how many consecutive bars with same chord (including this one)
        count = 1
        # Look back
        j = i - 1
        while j >= 0 and chords[j] == chord:
            count += 1
            j -= 1
        # Look forward
        j = i + 1
        while j < len(chords) and chords[j] == chord:
            count += 1
            j += 1

        # Map to symbol (ASCII-safe for Windows)
        if count >= 7:
            symbol = '#'  # Heavy stagnation
        elif count >= 4:
            symbol = '='  # Medium stagnation
        else:
            symbol = '.'  # Light stagnation

        stagnation_levels.append(symbol)

    # Display in rows
    lines = []
    for row_start in range(0, len(stagnation_levels), width):
        row_end = min(row_start + width, len(stagnation_levels))
        row_symbols = stagnation_levels[row_start:row_end]

        label = f"Bar {row_start + 1}-{row_end}:".ljust(12)
        line = label + "".join(row_symbols)
        lines.append(line)

    # Add legend
    lines.append("Legend: # = 7+ bars, = = 4-6 bars, . = 1-3 bars")

    return "\n".join(lines)

def visualize_diatonic_marking(bars: list[dict], diatonic_chords: list[str], width: int = 80) -> str:
    """
    Mark diatonic vs non-diatonic chords.

    Args:
        bars: List of bar dicts with 'chord' key
        diatonic_chords: List of diatonic chord names
        width: Display width in characters

    Returns:
        Multi-line string showing diatonic marking

    Format:
        Bar 1-80:   oooooooooooxoooooooo...
        Legend: o = diatonic, x = non-diatonic
    """
    if not bars:
        return "No bars to display"

    # Mark each bar (ASCII-safe for Windows)
    marks = []
    diatonic_count = 0
    for bar in bars:
        chord = bar['chord']
        if chord in diatonic_chords:
            marks.append('o')  # Diatonic
            diatonic_count += 1
        else:
            marks.append('x')  # Non-diatonic

    # Calculate rates
    total = len(bars)
    diatonic_rate = (diatonic_count / total * 100) if total > 0 else 0
    non_diatonic_rate = 100 - diatonic_rate

    # Display in rows
    lines = []
    for row_start in range(0, len(marks), width):
        row_end = min(row_start + width, len(marks))
        row_marks = marks[row_start:row_end]

        label = f"Bar {row_start + 1}-{row_end}:".ljust(12)
        line = label + "".join(row_marks)
        lines.append(line)

    # Add legend
    legend = f"Legend: o = diatonic ({diatonic_rate:.1f}%), x = non-diatonic ({non_diatonic_rate:.1f}%)"
    lines.append(legend)

    return "\n".join(lines)

# ========================================
# Phase 4: Main Test Function
# ========================================

def test_json_analysis(json_path: str = None):
    """
    Verify chord detection accuracy from JSON analysis file.

    Args:
        json_path: Path to JSON file (default: search for analysis-*.json in project root)

    Prints:
        - Analysis metadata (BPM, key, duration, bar count)
        - Stagnation metrics
        - Diatonic compliance metrics
        - Chord diversity metrics
        - Timing consistency metrics
        - ASCII visualizations
        - Overall quality assessment
    """
    print("=" * 60)
    print("TEST 4: JSON Analysis Verification")
    print("=" * 60)

    # Auto-detect JSON file if not provided
    if json_path is None:
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        json_files = [f for f in os.listdir(project_root)
                      if f.startswith('analysis-') and f.endswith('.json')]
        if not json_files:
            print("  No JSON analysis files found. Skipping test.\n")
            return
        json_path = os.path.join(project_root, json_files[0])
        print(f"  Auto-detected: {os.path.basename(json_path)}\n")

    # Load and parse JSON
    try:
        data = load_json_analysis(json_path)
    except Exception as e:
        print(f"  FAILED to load JSON: {e}\n")
        return

    # Extract metadata
    bpm = data['bpm']
    key_str = data['key']
    duration_sec = data.get('duration_sec', data.get('analyzed_duration_sec', 0))
    bars = data['bars']

    # Parse key
    key_root, key_mode = parse_key(key_str)
    key_display = f"{key_str} ({key_root} {'minor' if key_mode == 'm' else 'major'})"

    # Print metadata
    print(f"File: {os.path.basename(json_path)}")
    print(f"BPM: {bpm}")
    print(f"Key: {key_display}")
    print(f"Duration: {duration_sec:.1f}s")
    print(f"Total Bars: {len(bars)}\n")

    # Extract chord sequence
    chords = [b['chord'] for b in bars]

    # ========================================
    # Compute Metrics
    # ========================================

    # Stagnation analysis
    stag_metrics = compute_stagnation_metrics(chords)

    # Diatonic compliance
    diat_metrics = compute_diatonic_metrics(chords, key_root, key_mode)

    # Chord diversity
    div_metrics = compute_diversity_metrics(chords)

    # Timing consistency
    timing_metrics = compute_timing_metrics(bars, bpm)

    # ========================================
    # Print Metrics
    # ========================================

    print("--- Stagnation Analysis ---")
    print(f"Max consecutive: {stag_metrics['max_consecutive']} bars")
    print(f"Average consecutive: {stag_metrics['avg_consecutive']:.1f} bars")
    print(f"Total runs: {stag_metrics['total_runs']}")
    if stag_metrics['exceeds_threshold']:
        long_runs = [r for r in stag_metrics['stagnation_runs'] if r[1] > 6]
        print(f"Exceeds threshold (>6 bars): YES - {len(long_runs)} occurrence(s)")
        for chord, count, start_bar in long_runs[:5]:  # Show first 5
            print(f"  Run: {chord} for {count} bars starting at bar {start_bar}")
    else:
        print("Exceeds threshold (>6 bars): NO")
    print()

    print("--- Diatonic Compliance ---")
    print(f"Expected diatonic chords: {', '.join(diat_metrics['expected_diatonic'][:10])}...")
    print(f"Diatonic rate: {diat_metrics['diatonic_rate'] * 100:.1f}% ({diat_metrics['diatonic_count']}/{len(bars)} bars)")
    print(f"Diatonic chords used: {', '.join(diat_metrics['diatonic_chords'])}")
    if diat_metrics['non_diatonic_chords']:
        print(f"Non-diatonic chords: {', '.join(diat_metrics['non_diatonic_chords'])}")
    print()

    print("--- Chord Diversity ---")
    print(f"Unique chords: {div_metrics['unique_count']}")
    print(f"Simpson diversity index: {div_metrics['simpson_index']:.2f}")
    print(f"Shannon entropy: {div_metrics['entropy']:.2f} bits")
    print(f"Top 5 chords:")
    for i, (chord, count) in enumerate(div_metrics['most_common'], 1):
        percentage = (count / len(chords)) * 100
        print(f"  {i}. {chord}: {count} bars ({percentage:.1f}%)")
    print()

    print("--- Timing Consistency ---")
    print(f"Expected bar duration ({bpm} BPM): {timing_metrics['expected_duration']:.3f}s")
    print(f"Mean duration: {timing_metrics['mean_duration']:.3f}s")
    print(f"Std deviation: {timing_metrics['std_duration']:.3f}s")
    print(f"Min: {timing_metrics['min_duration']:.3f}s, Max: {timing_metrics['max_duration']:.3f}s")
    if timing_metrics['anomalies']:
        print(f"Anomalies: {len(timing_metrics['anomalies'])} bars with unusual duration")
        # Show first 3 anomalies
        for bar_idx, duration, deviation in timing_metrics['anomalies'][:3]:
            print(f"  Bar {bar_idx}: {duration:.3f}s (deviation {deviation:+.3f}s)")
        if len(timing_metrics['anomalies']) > 3:
            print(f"  ... and {len(timing_metrics['anomalies']) - 3} more")
    print(f"Timing OK: {'YES' if timing_metrics['timing_ok'] else 'NO'} (std {'<' if timing_metrics['timing_ok'] else '>='} 0.05s)")
    print()

    # ========================================
    # Visualizations
    # ========================================

    print("--- Visualizations ---")
    print()
    print("Chord Progression Timeline:")
    print(visualize_chord_progression(bars[:40]))  # Show first 40 bars
    if len(bars) > 40:
        print(f"  ... (showing first 40 of {len(bars)} bars)")
    print()

    print("Stagnation Heatmap:")
    print(visualize_stagnation_heatmap(chords[:80]))  # Show first 80 bars
    if len(chords) > 80:
        print(f"  ... (showing first 80 of {len(chords)} bars)")
    print()

    print("Diatonic Marking:")
    print(visualize_diatonic_marking(bars[:80], diat_metrics['expected_diatonic']))
    if len(bars) > 80:
        print(f"  ... (showing first 80 of {len(bars)} bars)")
    print()

    # ========================================
    # Overall Assessment
    # ========================================

    print("--- Overall Assessment ---")

    assessments = []

    # Diatonic compliance
    if diat_metrics['diatonic_rate'] > 0.8:
        print(f"[PASS] Diatonic compliance > 80% ({diat_metrics['diatonic_rate'] * 100:.1f}%)")
        assessments.append(True)
    else:
        print(f"[FAIL] Diatonic compliance <= 80% ({diat_metrics['diatonic_rate'] * 100:.1f}%)")
        assessments.append(False)

    # Stagnation
    if stag_metrics['exceeds_threshold']:
        print(f"[WARN] Max stagnation ({stag_metrics['max_consecutive']} bars) exceeds threshold (6 bars)")
        assessments.append(None)  # Warning, not fail
    else:
        print(f"[PASS] Max stagnation ({stag_metrics['max_consecutive']} bars) within threshold (6 bars)")
        assessments.append(True)

    # Chord diversity
    if 5 <= div_metrics['unique_count'] <= 15:
        print(f"[PASS] Chord diversity acceptable ({div_metrics['unique_count']} unique chords)")
        assessments.append(True)
    else:
        print(f"[NOTE] Chord diversity {div_metrics['unique_count']} unique chords (expected 5-15)")
        assessments.append(None)

    # Timing
    if timing_metrics['timing_ok']:
        print(f"[PASS] Timing consistency OK (std = {timing_metrics['std_duration']:.3f}s)")
        assessments.append(True)
    else:
        print(f"[WARN] Timing consistency poor (std = {timing_metrics['std_duration']:.3f}s)")
        assessments.append(None)

    print()

    # Overall verdict
    has_warnings = None in assessments
    has_failures = False in assessments

    if has_failures:
        print("Overall: FAIL")
    elif has_warnings:
        print("Overall: PASS with WARNINGS")
    else:
        print("Overall: PASS")

    print()

# ========================================
# Existing Tests
# ========================================

def test_templates():
    """Verify template expansion is correct."""
    templates, labels, matrix = build_chord_templates()
    
    print("=" * 60)
    print("TEST 1: Template Expansion")
    print("=" * 60)
    
    expected_types = ["", "m", "7", "m7", "maj7", "sus4"]
    expected_count = 12 * len(expected_types)  # 72
    
    print(f"  Template count: {len(labels)} (expected {expected_count})")
    assert len(labels) == expected_count, f"Expected {expected_count} templates, got {len(labels)}"
    
    # Check matrix shape
    print(f"  Matrix shape: {matrix.shape} (expected ({expected_count}, 12))")
    assert matrix.shape == (expected_count, 12), f"Wrong matrix shape: {matrix.shape}"
    
    # Check all templates have tabs
    missing_tabs = [label for label in labels if label not in CHORD_TO_TAB]
    if missing_tabs:
        print(f"  WARNING: Missing tabs for: {missing_tabs}")
    else:
        print(f"  All {len(labels)} chord templates have tab entries OK")
    
    print("  PASSED OK\n")

def test_diatonic_expansion():
    """Verify diatonic chord list includes new types."""
    print("=" * 60)
    print("TEST 2: Diatonic Chord List Expansion")
    print("=" * 60)
    
    # C major key
    chords_c = get_diatonic_chords_for_key("C", "")
    print(f"  C major diatonic: {chords_c}")
    
    # Should include C, Cmaj7 (I), Dm, Dm7 (ii), G, G7 (V), etc.
    assert "C" in chords_c, "C should be diatonic in C major"
    assert "Cmaj7" in chords_c, "Cmaj7 should be diatonic in C major (I=maj7)"
    assert "Csus4" in chords_c, "Csus4 should be diatonic in C major"
    assert "Dm" in chords_c, "Dm should be diatonic in C major (ii)"
    assert "Dm7" in chords_c, "Dm7 should be diatonic in C major (ii=m7)"
    assert "G" in chords_c, "G should be diatonic in C major (V)"
    assert "G7" in chords_c, "G7 should be diatonic in C major (V=7)"

    # D7, Dmaj7, Gmaj7 should NOT be diatonic in C major
    assert "D7" not in chords_c, "D7 should NOT be diatonic in C major (ii is minor, not dominant)"
    assert "Dmaj7" not in chords_c, "Dmaj7 should NOT be diatonic in C major"
    assert "Gmaj7" not in chords_c, "Gmaj7 should NOT be diatonic in C major (V is 7, not maj7)"
    
    # Am minor key
    chords_am = get_diatonic_chords_for_key("A", "m")
    print(f"  A minor diatonic: {chords_am}")
    assert "Am" in chords_am, "Am should be diatonic in A minor"
    assert "Am7" in chords_am, "Am7 should be diatonic in A minor"
    
    print("  PASSED OK\n")

def test_audio_analysis():
    """Test full analysis pipeline with available audio."""
    print("=" * 60)
    print("TEST 3: Full Audio Analysis")
    print("=" * 60)
    
    # Find available audio files
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    audio_files = []
    
    for fname in os.listdir(project_root):
        if fname.endswith(('.mp3', '.m4a', '.wav')):
            audio_files.append(os.path.join(project_root, fname))
    
    if not audio_files:
        print("  No audio files found for testing, skipping")
        return
    
    # Test with first available file (likely tax.mp3)
    test_file = audio_files[0]
    print(f"  Testing with: {os.path.basename(test_file)}")
    
    start = time.time()
    try:
        result = analyze_audio_file(test_file, duration_limit_sec=30)
        elapsed = time.time() - start
        
        print(f"  Analysis time: {elapsed:.1f}s")
        print(f"  BPM: {result['bpm']}")
        print(f"  Key: {result['key']}")
        print(f"  Bars: {len(result['bars'])}")
        
        # Get chord sequence
        chords = [b['chord'] for b in result['bars']]
        unique_chords = sorted(set(chords))
        print(f"  Unique chords ({len(unique_chords)}): {unique_chords}")
        
        # Check tabs exist
        missing = [c for c in unique_chords if c not in CHORD_TO_TAB]
        if missing:
            print(f"  WARNING: Chords without tabs: {missing}")
        else:
            print(f"  All detected chords have tab entries OK")
        
        # Show first 20 chords
        print(f"  First 20 chords: {chords[:20]}")
        
        # Basic sanity checks
        assert len(result['bars']) > 0, "Should have at least 1 bar"
        assert result['bpm'] > 0, "BPM should be positive"
        assert result['key'], "Key should not be empty"
        
        print("  PASSED OK\n")
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        print()

if __name__ == "__main__":
    print("\nChord Detection Accuracy Verification\n")

    # Existing tests
    test_templates()
    test_diatonic_expansion()
    test_audio_analysis()

    # New JSON-based test
    if len(sys.argv) > 1:
        # JSON path provided as argument
        json_path = sys.argv[1]
        test_json_analysis(json_path)
    else:
        # Auto-detect JSON file
        test_json_analysis()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
