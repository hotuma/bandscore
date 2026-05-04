"""
高度なlibrosaベースのBPM検出システム
マルチバンド分析とアンサンブル手法による精度向上
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
from scipy.signal import butter, sosfilt

def bandpass_filter(y: np.ndarray, sr: int, low_freq: int, high_freq: int) -> np.ndarray:
    """バンドパスフィルタ"""
    sos = butter(10, [low_freq, high_freq], 'band', fs=sr, output='sos')
    return sosfilt(sos, y)

def detect_bpm_band(y: np.ndarray, sr: int, band_name: str, bpm_range=(60, 240)) -> dict:
    """
    特定の周波数帯域でBPMを検出
    """
    hop_length = 512
    total_frames = len(y) // hop_length

    # オンセット検出
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units='frames',
        backtrack=True, pre_max=3, post_max=3
    )

    if len(onset_frames) < 8:
        return {
            'band': band_name,
            'bpm': None,
            'score': 0.0,
            'onsets': len(onset_frames)
        }

    # BPM検出（F1スコアベース）
    bpm_candidates = []
    tolerance = 4

    for c in range(bpm_range[0], bpm_range[1] + 1):
        beat_period = 60.0 * sr / (c * hop_length)
        grid = np.arange(0, total_frames, beat_period)
        if len(grid) == 0:
            continue

        # Precision
        hits = 0
        onset_set = set(int(o) for o in onset_frames)
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                if (g_int + t) in onset_set:
                    hits += 1
                    break
        precision = hits / len(grid) if len(grid) > 0 else 0

        # Recall
        grid_set = set()
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                grid_set.add(g_int + t)
        onset_hits = sum(1 for o in onset_frames if int(o) in grid_set)
        recall = onset_hits / max(1, len(onset_frames))

        # F1 score
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        bpm_candidates.append((c, f1_score, precision, recall))

    # F1スコアでソート
    bpm_candidates.sort(key=lambda x: x[1], reverse=True)

    if not bpm_candidates:
        return {
            'band': band_name,
            'bpm': None,
            'score': 0.0,
            'onsets': len(onset_frames)
        }

    best_bpm, best_score, precision, recall = bpm_candidates[0]

    return {
        'band': band_name,
        'bpm': best_bpm,
        'score': best_score,
        'precision': precision,
        'recall': recall,
        'onsets': len(onset_frames)
    }

def detect_bpm_multiband(y: np.ndarray, sr: int) -> dict:
    """
    マルチバンドBPM検出
    異なる周波数帯域でBPMを検出し、アンサンブルする
    """
    print("=== マルチバンドBPM検出 ===")

    # 周波数帯域の定義
    bands = [
        ('bass', 40, 150),      # バス帯域（40-150 Hz）
        ('mid', 150, 2000),     # 中域（150-2000 Hz）
        ('high', 2000, 8000),   # 高域（2000-8000 Hz）
        ('full', 20, 20000)     # 全帯域
    ]

    results = []

    for band_name, low_freq, high_freq in bands:
        print(f"\n{band_name}帯域 ({low_freq}-{high_freq} Hz) でBPM検出中...")

        # バンドパスフィルタ適用
        if band_name == 'full':
            y_band = y
        else:
            y_band = bandpass_filter(y, sr, low_freq, high_freq)

        # BPM検出
        result = detect_bpm_band(y_band, sr, band_name)
        results.append(result)

        if result['bpm']:
            print(f"  検出BPM: {result['bpm']:.1f} BPM, スコア: {result['score']:.3f}")
            print(f"  オンセット数: {result['onsets']}, 精度: {result['precision']:.3f}, 再現率: {result['recall']:.3f}")

    # 有効な結果のみフィルタリング
    valid_results = [r for r in results if r['bpm'] is not None]

    if not valid_results:
        return {
            'bpm': 120.0,
            'confidence': 0.0,
            'method': 'fallback',
            'details': valid_results
        }

    # スコアで重み付けされた平均BPMを計算
    total_weight = sum(r['score'] for r in valid_results)
    weighted_bpm = sum(r['bpm'] * r['score'] for r in valid_results) / total_weight

    # オクターブ誤り修正
    corrected_bpm = correct_octave_errors(weighted_bpm, valid_results)

    # 信頼度の計算
    confidence = calculate_confidence(valid_results, corrected_bpm)

    print(f"\n=== マルチバンドBPM検出結果 ===")
    print(f"加重平均BPM: {weighted_bpm:.1f} BPM")
    print(f"修正後BPM: {corrected_bpm:.1f} BPM")
    print(f"信頼度: {confidence:.3f}")

    return {
        'bpm': corrected_bpm,
        'confidence': confidence,
        'method': 'multiband',
        'details': valid_results,
        'weighted_bpm': weighted_bpm
    }

def correct_octave_errors(bpm: float, results: list) -> float:
    """
    オクターブ誤りを修正
    - 0.5倍速（2倍遅い）
    - 2倍速（2倍速い）
    - 1.5倍速（1.5倍速い）
    - 2/3速（1.5倍遅い）
    """
    candidates = [bpm]

    # オクターブ候補
    for result in results:
        original_bpm = result['bpm']
        candidates.extend([
            original_bpm * 0.5,   # 半分のBPM
            original_bpm * 2.0,   # 2倍のBPM
            original_bpm / 1.5,   # 2/3のBPM
            original_bpm * 1.5    # 1.5倍のBPM
        ])

    # 有効な範囲（60-240 BPM）にフィルタ
    valid_candidates = [c for c in candidates if 60 <= c <= 240]

    if not valid_candidates:
        return bpm

    # 最も頻度の高いBPMを選択
    from collections import Counter
    candidate_counts = Counter(int(round(c)) for c in valid_candidates)
    most_common_bpm = candidate_counts.most_common(1)[0][0]

    # 元のBPMに近い候補を優先
    closest_candidate = min(valid_candidates, key=lambda c: abs(c - most_common_bpm))

    # 元のBPMとの差が大きい場合のみ修正
    if abs(closest_candidate - bpm) < 5:
        return bpm
    else:
        return closest_candidate

def calculate_confidence(results: list, bpm: float) -> float:
    """
    信頼度を計算
    - 複数の帯域で一致しているか
    - スコアが高いか
    """
    if not results:
        return 0.0

    # BPMの標準偏差を計算
    bpms = [r['bpm'] for r in results if r['bpm']]
    if len(bpms) < 2:
        return max(r['score'] for r in results) if results else 0.0

    std_dev = np.std(bpms)
    avg_score = np.mean([r['score'] for r in results])

    # 標準偏差が小さいほど信頼度が高い
    consistency = max(0.0, 1.0 - std_dev / 30.0)

    # スコアも考慮
    confidence = (consistency * 0.6 + avg_score * 0.4)

    return min(1.0, max(0.0, confidence))

def test_advanced_librosa_bpm():
    """
    高度なlibrosa BPM検出のテスト
    """
    print("=" * 60)
    print("高度なlibrosa BPM検出テスト")
    print("=" * 60)

    # テスト1: 既存のテスト音声（ガソリン0812.m4a, 期待162 BPM）
    print("\n=== テスト1: ガソリン0812.m4a (期待162 BPM) ===")
    test_audio_1 = "c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a"

    y1, sr1 = librosa.load(test_audio_1, sr=22050, duration=30.0)
    result1 = detect_bpm_multiband(y1, sr1)

    expected_bpm_1 = 162.0
    error_1 = abs(result1['bpm'] - expected_bpm_1)

    print(f"\n期待値: {expected_bpm_1} BPM")
    print(f"検出BPM: {result1['bpm']:.1f} BPM")
    print(f"誤差: {error_1:.1f} BPM ({error_1/expected_bpm_1*100:.1f}%)")
    print(f"信頼度: {result1['confidence']:.3f}")

    if error_1 < 5:
        print(f"[EXCELLENT] 非常に正確！ 期待値 {expected_bpm_1} BPM に近いです")
    elif error_1 < 10:
        print(f"[GOOD] 良好です。期待値 {expected_bpm_1} BPM との差が {error_1:.1f} BPM")
    elif error_1 < 15:
        print(f"[ACCEPTABLE] 許容範囲です。期待値 {expected_bpm_1} BPM との差が {error_1:.1f} BPM")
    else:
        print(f"[WARNING] 改善の余地があります。期待値 {expected_bpm_1} BPM との差が {error_1:.1f} BPM")

    # テスト2: ターゲット楽曲（期待105 BPM）
    print("\n=== テスト2: ターゲット楽曲 (期待105 BPM) ===")
    target_audio = "c:/Users/USER/.gemini/antigravity/guitar-tab/backend/temp/05388afa7d224dbf846a1af4d61e2618-LaKp04a7hAM.m4a"

    if os.path.exists(target_audio):
        y2, sr2 = librosa.load(target_audio, sr=22050, duration=30.0)
        result2 = detect_bpm_multiband(y2, sr2)

        expected_bpm_2 = 105.0
        error_2 = abs(result2['bpm'] - expected_bpm_2)

        print(f"\n期待値: {expected_bpm_2} BPM")
        print(f"検出BPM: {result2['bpm']:.1f} BPM")
        print(f"誤差: {error_2:.1f} BPM ({error_2/expected_bpm_2*100:.1f}%)")
        print(f"信頼度: {result2['confidence']:.3f}")

        if error_2 < 5:
            print(f"[EXCELLENT] 非常に正確！ 期待値 {expected_bpm_2} BPM とほぼ一致")
        elif error_2 < 10:
            print(f"[GOOD] 良好です。期待値 {expected_bpm_2} BPM との差が {error_2:.1f} BPM")
        elif error_2 < 15:
            print(f"[ACCEPTABLE] 許容範囲です。期待値 {expected_bpm_2} BPM との差が {error_2:.1f} BPM")
        else:
            print(f"[POOR] 改善の余地があります。期待値 {expected_bpm_2} BPM との差が {error_2:.1f} BPM")
    else:
        print(f"ターゲット音声ファイルが見つかりません: {target_audio}")
        result2 = None

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

    return result1, result2

if __name__ == "__main__":
    test_advanced_librosa_bpm()
