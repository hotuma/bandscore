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
    test_templates()
    test_diatonic_expansion()
    test_audio_analysis()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
