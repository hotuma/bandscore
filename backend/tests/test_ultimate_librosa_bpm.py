"""
究極のlibrosaベースBPM検出システム
テンポトラッキングと高度なオクターブ誤り修正
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa

def detect_bpm_tempo_tracking(y: np.ndarray, sr: int, initial_bpm: float) -> float:
    """
    テンポトラッキングによるBPM検出
    初期BPMを基準に時間経過とともにBPMの変化を追跡
    """
    print(f"テンポトラッキング（初期BPM: {initial_bpm:.1f} BPM）...")

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # 小節ごとのBPM変化を追跡
    beats_per_second = initial_bpm / 60.0
    frames_per_beat = sr / (hop_length * beats_per_second)

    # スライディングウィンドウでBPMを推定
    window_size_beats = 8  # 8拍分のウィンドウ
    window_size_frames = int(window_size_beats * frames_per_beat)

    bpm_estimates = []
    frame_start = 0

    while frame_start + window_size_frames < len(onset_env):
        frame_end = frame_start + window_size_frames
        window_env = onset_env[frame_start:frame_end]

        # ウィンドウ内でBPMを推定
        try:
            tempo, _ = librosa.beat.beat_track(onset_envelope=window_env, sr=sr,
                                                  hop_length=hop_length, bpm=initial_bpm)

            if isinstance(tempo, (int, float)):
                bpm_estimates.append(tempo)
            elif hasattr(tempo, '__len__') and len(tempo) > 0:
                bpm_estimates.append(tempo[0])
        except:
            # エラーが発生した場合は初期BPMを使用
            bpm_estimates.append(initial_bpm)

        frame_start += window_size_frames // 2  # 50%オーバーラップ

    if not bpm_estimates:
        return initial_bpm

    # 中央値を採用（外れ値を無視）
    final_bpm = np.median(bpm_estimates)

    print(f"  テンポトラッキング結果: {final_bpm:.1f} BPM")
    print(f"  BPM推定回数: {len(bpm_estimates)}")

    return final_bpm

def detect_bpm_with_tempo_grid(y: np.ndarray, sr: int, bpm_range=(60, 240)) -> dict:
    """
    テンポグリッド法によるBPM検出
    """
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units='frames',
        backtrack=True, pre_max=3, post_max=3
    )

    if len(onset_frames) < 8:
        return {'bpm': 120.0, 'confidence': 0.0, 'method': 'tempo_grid_fallback'}

    total_frames = len(y) // hop_length
    bpm_candidates = []

    for c in range(bpm_range[0], bpm_range[1] + 1):
        beat_period = 60.0 * sr / (c * hop_length)
        grid = np.arange(0, total_frames, beat_period)

        if len(grid) == 0:
            continue

        # ヒット数と正確性を計算
        hits = 0
        misses = 0
        tolerance = 3

        onset_set = set(int(o) for o in onset_frames)

        for g in grid:
            g_int = int(round(g))
            hit = False
            for t in range(-tolerance, tolerance + 1):
                if (g_int + t) in onset_set:
                    hit = True
                    break

            if hit:
                hits += 1
            else:
                misses += 1

        # スコア計算（ヒット率とグリッドの整合性）
        precision = hits / len(grid) if len(grid) > 0 else 0
        recall = hits / len(onset_frames) if len(onset_frames) > 0 else 0

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        # グリッドの規則性ボーナス
        regularity_bonus = 1.0 - (misses / len(grid)) if len(grid) > 0 else 0.0

        # 総合スコア
        total_score = f1_score * 0.7 + regularity_bonus * 0.3

        bpm_candidates.append((c, total_score, precision, recall))

    bpm_candidates.sort(key=lambda x: x[1], reverse=True)

    if not bpm_candidates:
        return {'bpm': 120.0, 'confidence': 0.0, 'method': 'tempo_grid_fallback'}

    best_bpm, best_score, precision, recall = bpm_candidates[0]

    return {
        'bpm': best_bpm,
        'confidence': best_score,
        'precision': precision,
        'recall': recall,
        'method': 'tempo_grid'
    }

def advanced_octave_correction(bpm: float, onset_env: np.ndarray, sr: int) -> float:
    """
    高度なオクターブ誤り修正
    """
    # オンセット間隔の分布を分析
    hop_length = 512
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units='frames',
        backtrack=True, pre_max=3, post_max=3
    )

    if len(onset_frames) < 3:
        return bpm

    # オンセット間隔を計算
    intervals = np.diff(onset_frames) * hop_length / sr  # 秒単位
    median_interval = np.median(intervals)
    estimated_bpm = 60.0 / median_interval

    # オクターブ候補
    candidates = [
        estimated_bpm,
        estimated_bpm * 0.5,
        estimated_bpm * 2.0,
        estimated_bpm / 1.5,
        estimated_bpm * 1.5,
        bpm,  # 元のBPM
        bpm * 0.5,
        bpm * 2.0,
        bpm / 1.5,
        bpm * 1.5
    ]

    # 有効範囲にフィルタ
    valid_candidates = [c for c in candidates if 60 <= c <= 240]

    if not valid_candidates:
        return bpm

    # ヒストグラムで最も頻度の高いBPMを特定
    from collections import Counter
    rounded_candidates = [int(round(c)) for c in valid_candidates]
    candidate_counts = Counter(rounded_candidates)

    # 最頻値とその近くの候補を採用
    most_common = candidate_counts.most_common(5)
    most_common_bpms = [bpm for bpm, count in most_common]

    # 元の推定BPMに最も近いものを選択
    closest_to_estimated = min(most_common_bpms, key=lambda b: abs(b - estimated_bpm))

    print(f"  オンセット間隔から推定: {estimated_bpm:.1f} BPM")
    print(f"  修正後BPM: {closest_to_estimated:.1f} BPM")

    return closest_to_estimated

def ultimate_bpm_detection(y: np.ndarray, sr: int) -> dict:
    """
    究極のBPM検出システム
    テンポグリッド法 + テンポトラッキング + 高度なオクターブ修正
    """
    print("=== 究極BPM検出システム ===")

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # ステップ1: テンポグリッド法
    print("\nステップ1: テンポグリッド法")
    grid_result = detect_bpm_with_tempo_grid(y, sr)

    # ステップ2: テンポトラッキング
    print("\nステップ2: テンポトラッキング")
    tempo_tracking_bpm = detect_bpm_tempo_tracking(y, sr, grid_result['bpm'])

    # ステップ3: 高度なオクターブ修正
    print("\nステップ3: 高度なオクターブ修正")
    final_bpm = advanced_octave_correction(tempo_tracking_bpm, onset_env, sr)

    # 信頼度計算
    confidence = max(grid_result['confidence'], 0.5)

    print(f"\n=== 究極BPM検出結果 ===")
    print(f"テンポグリッド: {grid_result['bpm']:.1f} BPM (信頼度: {grid_result['confidence']:.3f})")
    print(f"テンポトラッキング: {tempo_tracking_bpm:.1f} BPM")
    print(f"最終BPM: {final_bpm:.1f} BPM")
    print(f"最終信頼度: {confidence:.3f}")

    return {
        'bpm': final_bpm,
        'confidence': confidence,
        'method': 'ultimate',
        'grid_bpm': grid_result['bpm'],
        'tempo_tracking_bpm': tempo_tracking_bpm
    }

def test_ultimate_bpm_detection():
    """
    究極BPM検出システムのテスト
    """
    print("=" * 60)
    print("究極librosa BPM検出テスト")
    print("=" * 60)

    # テスト1: 既存のテスト音声（ガソリン0812.m4a, 期待162 BPM）
    print("\n=== テスト1: ガソリン0812.m4a (期待162 BPM) ===")
    test_audio_1 = "c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a"

    y1, sr1 = librosa.load(test_audio_1, sr=22050, duration=30.0)
    result1 = ultimate_bpm_detection(y1, sr1)

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
        result2 = ultimate_bpm_detection(y2, sr2)

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
    test_ultimate_bpm_detection()
