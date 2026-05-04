"""
UVERworld「哀しみはきっと」のBPM検出テスト
本来のBPMは105だが、87.3と誤検出される問題の確認と修正
"""
import os
import sys
import numpy as np
import librosa
from collections import Counter

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    analyze_audio_file,
    highpass_filter,
    lowpass_filter,
    bandpass_filter,
    evaluate_tempo_prior,
    evaluate_bass_ac,
    verify_tempo_octave
)

# テストファイルパス
TEST_FILE = r"C:\Users\USER\.gemini\antigravity\url_transrater\downloads\UVERworld　『哀しみはきっと』.mp3"
EXPECTED_BPM = 105.0

def test_current_bpm_detection():
    """現在のBPM検出結果を確認"""
    print("=" * 60)
    print("現在のBPM検出結果")
    print("=" * 60)

    result = analyze_audio_file(TEST_FILE, duration_limit_sec=30)
    detected_bpm = result.get('bpm', 0)

    print(f"検出されたBPM: {detected_bpm:.1f}")
    print(f"期待されるBPM: {EXPECTED_BPM:.1f}")
    print(f"誤差: {abs(detected_bpm - EXPECTED_BPM):.1f} BPM")
    print(f"比率: {detected_bpm / EXPECTED_BPM:.3f}")

    return detected_bpm

def analyze_bpm_error(y, sr):
    """BPM誤検出の原因を分析"""
    print("\n" + "=" * 60)
    print("BPM誤検出原因の分析")
    print("=" * 60)

    # オンセット検出
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units='frames',
        backtrack=True, pre_max=3, post_max=3
    )
    onset_set = set(onset_frames.tolist())
    total_frames = len(onset_env)

    # 各BPM候補のスコアを計算
    tolerance = 4
    candidates = []

    for c in range(60, 241):
        beat_period = 60.0 * sr / (c * 512)
        grid = np.arange(0, total_frames, beat_period)
        if len(grid) == 0:
            continue

        hits = 0
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                if (g_int + t) in onset_set:
                    hits += 1
                    break
        precision = hits / len(grid)

        grid_set = set()
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                grid_set.add(g_int + t)
        onset_hits = sum(1 for o in onset_frames if int(o) in grid_set)
        recall = onset_hits / max(1, len(onset_frames))

        beta_sq = 0.64
        if precision + recall > 0:
            score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        else:
            score = 0.0

        candidates.append((c, score, precision, recall))

    # スコアでソート
    candidates.sort(key=lambda x: x[1], reverse=True)

    print(f"{'BPM':>6} {'Score':>8} {'Precision':>10} {'Recall':>10}")
    print("-" * 40)
    for c, s, p, r in candidates[:10]:
        marker = " <-- DETECTED" if abs(c - 87) < 1.5 else ""
        marker2 = " <-- EXPECTED" if abs(c - EXPECTED_BPM) < 1.5 else marker
        print(f"{c:>6} {s:>8.3f} {p:>10.3f} {r:>10.3f}{marker2}")

    # 期待BPMと検出BPMのスコアを比較
    detected_score = next((s for c, s, _, _ in candidates if abs(c - 87) < 1.5), 0)
    expected_score = next((s for c, s, _, _ in candidates if abs(c - EXPECTED_BPM) < 1.5), 0)

    print(f"\n87 BPM スコア: {detected_score:.3f}")
    print(f"105 BPM スコア: {expected_score:.3f}")

    return candidates

def check_bass_ac_correlation(y, sr):
    """バス帯域の自己相関で正しいBPMを確認"""
    print("\n" + "=" * 60)
    print("バス帯域自己相関分析")
    print("=" * 60)

    # バス帯域抽出
    y_bass = lowpass_filter(y, sr, cutoff_hz=200)
    y_bass = highpass_filter(y_bass, sr, cutoff_hz=20)
    bass_env = librosa.onset.onset_strength(y=y_bass, sr=sr, hop_length=512)
    ac = librosa.autocorrelate(bass_env)
    if ac[0] > 0:
        ac = ac / ac[0]

    # 各BPMのバスAC値を計算
    print(f"{'BPM':>6} {'Bass AC':>10}")
    print("-" * 20)
    for bpm in [70, 80, 87, 90, 100, 105, 110, 120, 130, 140, 150, 160, 170, 180]:
        lag = 60.0 * sr / (bpm * 512)
        lag_int = int(round(lag))
        if 0 < lag_int < len(ac) - 1:
            val = float((ac[lag_int - 1] + ac[lag_int] + ac[lag_int + 1]) / 3.0)
            marker = " <-- EXPECTED" if abs(bpm - EXPECTED_BPM) < 1.5 else ""
            marker2 = " <-- DETECTED" if abs(bpm - 87) < 1.5 else marker
            print(f"{bpm:>6} {val:>10.3f}{marker2}")

    del ac, bass_env, y_bass

def check_fractional_ratios():
    """1.2倍速（3/5）などの分数倍関係をチェック"""
    print("\n" + "=" * 60)
    print("分数倍関係のチェック")
    print("=" * 60)

    detected = 87.3
    expected = 105.0

    # 可能な分数倍関係
    ratios = {
        "半速 (1/2)": detected * 2,
        "1.5倍速 (3/2)": detected * 1.5,
        "2/3倍速": detected / (2/3),
        "3/5倍速 (1.2x)": detected * 1.2,
        "5/6倍速": detected / (5/6),
        "4/5倍速 (1.25x)": detected * 1.25,
        "5/4倍速": detected / (5/4),
        "3/4倍速": detected / (3/4),
        "4/3倍速": detected * (4/3),
    }

    print(f"検出BPM: {detected:.1f}")
    print(f"期待BPM: {expected:.1f}")
    print(f"\n検出BPMからの変換:")
    for name, converted in ratios.items():
        diff = abs(converted - expected)
        match = " <-- MATCH!" if diff < 3 else ""
        print(f"  {name}: {converted:.1f} BPM (差: {diff:.1f}){match}")

if __name__ == "__main__":
    # オーディオ読み込み
    print("オーディオファイルを読み込み中...")
    y, sr = librosa.load(TEST_FILE, sr=22050, mono=True, duration=60)
    print(f"読み込み完了: {len(y)} サンプル @ {sr} Hz\n")

    # テスト実行
    detected_bpm = test_current_bpm_detection()
    analyze_bpm_error(y, sr)
    check_bass_ac_correlation(y, sr)
    check_fractional_ratios()

    # ヒント
    print("\n" + "=" * 60)
    print("修正アプローチの提案")
    print("=" * 60)
    print("1. バス帯域自己相関の補正ロジックをBPM < 100でも適用")
    print("2. 1.2倍速（3/5）の分数倍補正を追加")
    print("3. onset_detectのパラメータ調整（pre_max, post_maxの増減）")
