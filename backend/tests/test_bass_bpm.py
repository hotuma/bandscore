"""
バス帯域BPM検出のテストスクリプト
バス帯域に特化したBPM検出が正しく機能しているか確認
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

def lowpass_filter(y: np.ndarray, sr: int, cutoff_hz: int = 200) -> np.ndarray:
    """ローパスフィルタ"""
    sos = butter(10, cutoff_hz, 'low', fs=sr, output='sos')
    return sosfilt(sos, y)

def highpass_filter(y: np.ndarray, sr: int, cutoff_hz: int = 20) -> np.ndarray:
    """ハイパスフィルタ"""
    sos = butter(10, cutoff_hz, 'high', fs=sr, output='sos')
    return sosfilt(sos, y)

def test_bass_bpm_detection():
    """バス帯域BPM検出テスト"""
    print("=== バス帯域BPM検出テスト ===")

    # テスト用音声の読み込み
    test_audio = "c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a"

    if not os.path.exists(test_audio):
        print(f"テスト音声ファイルが見つかりません: {test_audio}")
        return None

    print(f"\n音声ファイル読み込み: {test_audio}")
    y, sr = librosa.load(test_audio, sr=22050, duration=30.0)

    hop_length = 512
    total_frames = len(y) // hop_length

    # === バス帯域BPM検出 ===
    print("\n=== バス帯域BPM検出 ===")
    bass_y = lowpass_filter(y, sr, cutoff_hz=150)
    bass_y = highpass_filter(bass_y, sr, cutoff_hz=40)
    bass_onset_env = librosa.onset.onset_strength(y=bass_y, sr=sr, hop_length=hop_length)
    bass_onset_frames = librosa.onset.onset_detect(
        onset_envelope=bass_onset_env, sr=sr, units='frames',
        backtrack=True, pre_max=3, post_max=3
    )

    print(f"検出されたバスオンセット数: {len(bass_onset_frames)}")

    if len(bass_onset_frames) >= 8:
        bass_bpm_candidates = []
        bass_tolerance = 4

        for c in range(60, 241):
            beat_period = 60.0 * sr / (c * hop_length)
            grid = np.arange(0, total_frames, beat_period)
            if len(grid) == 0:
                continue

            hits = 0
            bass_onset_set = set(int(o) for o in bass_onset_frames)
            for g in grid:
                g_int = int(round(g))
                for t in range(-bass_tolerance, bass_tolerance + 1):
                    if (g_int + t) in bass_onset_set:
                        hits += 1
                        break
            precision = hits / len(grid) if len(grid) > 0 else 0

            # Recall calculation
            grid_set = set()
            for g in grid:
                g_int = int(round(g))
                for t in range(-bass_tolerance, bass_tolerance + 1):
                    grid_set.add(g_int + t)
            bass_onset_hits = sum(1 for o in bass_onset_frames if int(o) in grid_set)
            recall = bass_onset_hits / max(1, len(bass_onset_frames))

            # F1 score
            if precision + recall > 0:
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                f1_score = 0.0

            bass_bpm_candidates.append((c, f1_score, precision, recall))

        # Sort by F1 score
        bass_bpm_candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"\nバス帯域上位10候補:")
        for i, (bpm, f1, p, r) in enumerate(bass_bpm_candidates[:10]):
            print(f"  {i+1}. {bpm} BPM (F1={f1:.3f}, P={p:.3f}, R={r:.3f})")

        bass_best_bpm, bass_best_score, _, _ = bass_bpm_candidates[0]
        print(f"\nバス帯域最適BPM: {bass_best_bpm} BPM (F1={bass_best_score:.3f})")

    # === 全帯域BPM検出（比較用） ===
    print("\n=== 全帯域BPM検出（比較用） ===")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units='frames',
        backtrack=True, pre_max=3, post_max=3
    )

    print(f"検出された全オンセット数: {len(onset_frames)}")

    full_band_candidates = []
    tolerance = 4

    for c in range(60, 241):
        beat_period = 60.0 * sr / (c * hop_length)
        grid = np.arange(0, total_frames, beat_period)
        if len(grid) == 0:
            continue

        hits = 0
        onset_set = set(int(o) for o in onset_frames)
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                if (g_int + t) in onset_set:
                    hits += 1
                    break
        precision = hits / len(grid) if len(grid) > 0 else 0

        grid_set = set()
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                grid_set.add(g_int + t)
        onset_hits = sum(1 for o in onset_frames if int(o) in grid_set)
        recall = onset_hits / max(1, len(onset_frames))

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        full_band_candidates.append((c, f1_score, precision, recall))

    full_band_candidates.sort(key=lambda x: x[1], reverse=True)

    print(f"\n全帯域上位10候補:")
    for i, (bpm, f1, p, r) in enumerate(full_band_candidates[:10]):
        print(f"  {i+1}. {bpm} BPM (F1={f1:.3f}, P={p:.3f}, R={r:.3f})")

    full_best_bpm, full_best_score, _, _ = full_band_candidates[0]
    print(f"\n全帯域最適BPM: {full_best_bpm} BPM (F1={full_best_score:.3f})")

    # === 結果の比較 ===
    print("\n=== 結果比較 ===")
    print(f"バス帯域: {bass_best_bpm} BPM (F1={bass_best_score:.3f})")
    print(f"全帯域:   {full_best_bpm} BPM (F1={full_best_score:.3f})")

    if abs(bass_best_bpm - full_best_bpm) > 5:
        print(f"\n[注意] 両手法で大きな差があります: {abs(bass_best_bpm - full_best_bpm):.1f} BPM")

        # スコア比較
        if bass_best_score > full_best_score * 1.1:
            print(f"[推奨] バス帯域BPMを採用: {bass_best_bpm} BPM (スコアが {bass_best_score/full_best_score:.2f}x 優れている)")
        elif full_best_score > bass_best_score * 1.1:
            print(f"[推奨] 全帯域BPMを採用: {full_best_bpm} BPM (スコアが {full_best_score/bass_best_score:.2f}x 優れている)")
        else:
            print(f"[推奨] バス帯域BPMを採用: {bass_best_bpm} BPM (スコアが類似しているがバス帯域の方が信頼性が高い)")
    else:
        print(f"\n[OK] 両手法で一致: {bass_best_bpm} BPM")

    return {
        'bass_bpm': bass_best_bpm,
        'bass_score': bass_best_score,
        'full_bpm': full_best_bpm,
        'full_score': full_best_score
    }

if __name__ == "__main__":
    result = test_bass_bpm_detection()
    if result:
        print("\n" + "=" * 50)
        print("テスト完了")
        print("=" * 50)
