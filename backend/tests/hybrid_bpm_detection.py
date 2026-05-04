"""
ハイブリッド BPM 検出の実装
第1段階: 初期 BPM 推定 (高速)
第2段階: 詳細スコアリング (中精度)
第3段階: オクターブ検証 (高精度)
"""
import math
import numpy as np
import librosa
from scipy.signal import butter, sosfilt
from typing import List, Tuple


def lowpass_filter(y: np.ndarray, sr: int, cutoff_hz: int = 200) -> np.ndarray:
    """ローパスフィルタ（バス帯域抽出）"""
    sos = butter(10, cutoff_hz, 'low', fs=sr, output='sos')
    return sosfilt(sos, y)


def evaluate_bass_ac(y: np.ndarray, sr: int, bpm: float, hop_length: int = 512) -> float:
    """
    バス帯域での自己相関評価

    Args:
        y: 音声信号
        sr: サンプリングレート
        bpm: 候補 BPM
        hop_length: ホップ長

    Returns:
        バス帯域自己相関スコア (0-1)
    """
    y_bass = lowpass_filter(y, sr, cutoff_hz=200)
    onset_env = librosa.onset.onset_strength(y=y_bass, sr=sr, hop_length=hop_length)
    expected_lag = int(round(60.0 * sr / (bpm * hop_length)))

    if expected_lag <= 0:
        return 0.0
    expected_lag = max(expected_lag, 1)

    ac = librosa.autocorrelate(onset_env)
    if expected_lag >= len(ac):
        return 0.0

    return float(ac[expected_lag]) / max(1.0, float(ac[0]))


def evaluate_fullband_ac(onset_env: np.ndarray, sr: int, bpm: float,
                       hop_length: int = 512) -> float:
    """
    全帯域での自己相関評価

    Args:
        onset_env: オンセットエンベロープ
        sr: サンプリングレート
        bpm: 候補 BPM
        hop_length: ホップ長

    Returns:
        全帯域自己相関スコア (0-1)
    """
    expected_lag = int(round(60.0 * sr / (bpm * hop_length)))

    if expected_lag <= 0:
        return 0.0
    expected_lag = max(expected_lag, 1)

    ac = librosa.autocorrelate(onset_env)
    if expected_lag >= len(ac):
        return 0.0

    return float(ac[expected_lag]) / max(1.0, float(ac[0]))


def evaluate_phase_concentration(onset_env: np.ndarray, sr: int, bpm: float,
                               n_bins: int = 36) -> float:
    """
    ビート位相エネルギー集中度評価

    Args:
        onset_env: オンセットエンベロープ
        sr: サンプリングレート
        bpm: 候補 BPM
        n_bins: 位相ビン数

    Returns:
        位相集中度スコア (0-1)
    """
    beat_interval = 60.0 / bpm
    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    phases = (times % beat_interval) / beat_interval

    bins = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i, phase in enumerate(phases):
        bin_idx = min(int(phase * n_bins), n_bins - 1)
        bins[bin_idx] += onset_env[i]
        counts[bin_idx] += 1

    avg_bins = bins / np.maximum(counts, 1)
    mean_val = np.mean(avg_bins)

    if mean_val > 0:
        cv = np.std(avg_bins) / mean_val
        return min(cv, 1.0)
    return 0.0


def evaluate_onset_consistency(onset_frames: np.ndarray, sr: int,
                              bpm: float, tolerance: int = 4) -> float:
    """
    オンセット整合性評価

    Args:
        onset_frames: オンセットフレーム配列
        sr: サンプリングレート
        bpm: 候補 BPM
        tolerance: 許容フレーム数

    Returns:
        オンセット整合性スコア (0-1)
    """
    total_frames = len(onset_frames) * 20  # 推定総フレーム数
    beat_period = 60.0 * sr / (bpm * 512)
    grid = np.arange(0, total_frames, beat_period)

    if len(grid) == 0:
        return 0.0

    onset_set = set(int(o) for o in onset_frames)

    # グリッド上のオンセット数
    hits = 0
    for g in grid:
        g_int = int(round(g))
        for t in range(-tolerance, tolerance + 1):
            if (g_int + t) in onset_set:
                hits += 1
                break

    # 整合性スコア
    consistency = hits / len(grid) if len(grid) > 0 else 0.0

    return consistency


def evaluate_tempo_prior(bpm: float, target_range: Tuple[float, float] = (140, 170)) -> float:
    """
    テンポ事前確率評価

    Args:
        bpm: 候補 BPM
        target_range: 目標範囲 (min, max)

    Returns:
    テンポ事前確率スコア (0-1)
    """
    min_bpm, max_bpm = target_range

    # 目標範囲内であれば高いスコア
    if min_bpm <= bpm <= max_bpm:
        mean = (min_bpm + max_bpm) / 2.0
        std = (max_bpm - min_bpm) / 3.0  # より狭い分布で中速を強調
        return math.exp(-0.5 * ((bpm - mean) / std) ** 2)
    else:
        # 範囲外の場合、距離に応じて減少
        dist = min(abs(bpm - min_bpm), abs(bpm - max_bpm))
        return math.exp(-0.5 * (dist / 20.0) ** 2)  # より急激な減衰


def initial_bpm_estimation(y: np.ndarray, sr: int,
                          tolerance: int = 4,
                          bpm_range: Tuple[int, int] = (140, 170),
                          top_k: int = 10) -> List[Tuple[float, float]]:
    """
    第1段階: 初期 BPM 推定 (高速)

    Args:
        y: 音声信号
        sr: サンプリングレート
        tolerance: 許容フレーム数
        bpm_range: BPM スキャン範囲
        top_k: 返す候補数

    Returns:
        [(BPM, スコア), ...] (top_k 個)
    """
    print("[Stage 1] 初期 BPM 推定中...")
    print(f"[Stage 1] BPM 範囲: {bpm_range[0]}-{bpm_range[1]} BPM")

    # オンセット検出
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=True,
        pre_max=3,
        post_max=3
    )

    print(f"[Stage 1] 検出されたオンセット数: {len(onset_frames)}")

    if len(onset_frames) < 2:
        return [(120.0, 0.5)]  # デフォルト

    # BPM スキャン
    total_frames = len(onset_env)
    num_onsets = len(onset_frames)
    onset_set = set(onset_frames)

    candidates = []
    BEAT_F_BETA = 0.8
    beta_sq = BEAT_F_BETA ** 2

    for c in range(bpm_range[0], bpm_range[1] + 1):
        beat_period = 60.0 * sr / (c * 512)
        grid = np.arange(0, total_frames, beat_period)

        if len(grid) == 0:
            continue

        # Precision
        hits = 0
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                if (g_int + t) in onset_set:
                    hits += 1
                    break
        precision = hits / len(grid)

        # Recall
        grid_set = set()
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                grid_set.add(g_int + t)
        onset_hits = sum(1 for o in onset_frames if int(o) in grid_set)
        recall = onset_hits / max(1, num_onsets)

        # F_beta スコア
        if precision + recall > 0:
            fb_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        else:
            fb_score = 0.0

        # テンポ事前確率 (強化)
        tempo_prior = evaluate_tempo_prior(c)

        # 統合スコア (テンポ事前確率の重みを増強)
        score = fb_score * (0.5 + 0.5 * tempo_prior)
        candidates.append((c, score))

    # 上位候補を返す
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:top_k]

    print(f"[Stage 1] 上位{len(top_candidates)}候補:")
    for i, (bpm, score) in enumerate(top_candidates):
        print(f"  {i+1}. {bpm} BPM (score={score:.3f})")

    return top_candidates


def detailed_scoring(y: np.ndarray, sr: int,
                   candidates: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    第2段階: 詳細スコアリング (中精度)

    Args:
        y: 音声信号
        sr: サンプリングレート
        candidates: 候補 BPM リスト [(BPM, 初期スコア), ...]

    Returns:
        (最適 BPM, 統合スコア)
    """
    print("[Stage 2] 詳細スコアリング中...")

    # オンセットエンベロープ（全帯域）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

    # オンセットフレーム
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=True,
        pre_max=3,
        post_max=3
    )

    scored_candidates = []

    for bpm, initial_score in candidates:
        # 各評価指標を計算
        bass_ac = evaluate_bass_ac(y, sr, bpm)
        full_ac = evaluate_fullband_ac(onset_env, sr, bpm)
        phase = evaluate_phase_concentration(onset_env, sr, bpm)
        onset_consistency = evaluate_onset_consistency(onset_frames, sr, bpm)
        tempo_prior = evaluate_tempo_prior(bpm)

        # 重み付け統合スコア
        # バス自己相関: 0.20 (低下: 過剰な影響を抑制)
        # 全帯域自己相関: 0.15 (低下)
        # 位相集中度: 0.20
        # オンセット整合性: 0.15
        # テンポ事前確率: 0.30 (増強: 中速を優先)
        combined_score = (
            bass_ac * 0.20 +
            full_ac * 0.15 +
            phase * 0.20 +
            onset_consistency * 0.15 +
            tempo_prior * 0.30
        )

        scored_candidates.append((bpm, combined_score))

        print(f"  {bpm} BPM: bass={bass_ac:.3f}, full={full_ac:.3f}, "
              f"phase={phase:.3f}, onset={onset_consistency:.3f}, "
              f"prior={tempo_prior:.3f}, combined={combined_score:.3f}")

    # 最も信頼性の高い BPM を選択
    best_bpm, best_score = max(scored_candidates, key=lambda x: x[1])

    print(f"[Stage 2] 選択された BPM: {best_bpm:.1f} (combined_score={best_score:.3f})")

    return best_bpm, best_score


def octave_verification(y: np.ndarray, sr: int, bpm: float) -> float:
    """
    第3段階: オクターブ検証 (高精度)

    Args:
        y: 音声信号
        sr: サンプリングレート
        bpm: 候補 BPM

    Returns:
        最終 BPM (オクターブ補正済み)
    """
    print("[Stage 3] オクターブ検証中...")

    # オンセットエンベロープ
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

    # オクターブ候補のスコアを計算
    def get_score(target_bpm):
        bass_ac = evaluate_bass_ac(y, sr, target_bpm)
        full_ac = evaluate_fullband_ac(onset_env, sr, target_bpm)
        phase = evaluate_phase_concentration(onset_env, sr, target_bpm)
        prior = evaluate_tempo_prior(target_bpm)

        return bass_ac * 0.40 + full_ac * 0.25 + phase * 0.20 + prior * 0.15

    # 現在の BPM のスコア
    score_full = get_score(bpm)

    # 半速候補
    half_bpm = bpm * 0.5
    if half_bpm >= 60:
        score_half = get_score(half_bpm)
    else:
        score_half = 0.0

    # 倍速候補
    double_bpm = bpm * 2.0
    if double_bpm <= 240:
        score_double = get_score(double_bpm)
    else:
        score_double = 0.0

    print(f"[Stage 3] {bpm:.1f} BPM: score={score_full:.3f}")
    if half_bpm >= 60:
        print(f"[Stage 3] {half_bpm:.1f} BPM (×0.5): score={score_half:.3f}")
    if double_bpm <= 240:
        print(f"[Stage 3] {double_bpm:.1f} BPM (×2.0): score={score_double:.3f}")

    # ゲート条件
    PHASE_GATE = 0.25
    SCORE_OVERRIDE_RATIO = 1.15

    # 半速補正のチェック
    if half_bpm >= 60:
        half_phase = evaluate_phase_concentration(onset_env, sr, half_bpm)
        score_ratio = score_half / max(score_full, 1e-9)

        if half_phase >= PHASE_GATE or score_ratio >= SCORE_OVERRIDE_RATIO:
            print(f"[Stage 3] 半速補正適用: {bpm:.1f} → {half_bpm:.1f} BPM "
                  f"(phase={half_phase:.3f}, ratio={score_ratio:.2f})")
            return half_bpm

    # 倍速補正のチェック
    if double_bpm <= 240:
        double_phase = evaluate_phase_concentration(onset_env, sr, double_bpm)
        score_ratio = score_double / max(score_full, 1e-9)

        if double_phase >= PHASE_GATE and score_ratio >= SCORE_OVERRIDE_RATIO:
            print(f"[Stage 3] 倍速補正適用: {bpm:.1f} → {double_bpm:.1f} BPM "
                  f"(phase={double_phase:.3f}, ratio={score_ratio:.2f})")
            return double_bpm

    # 補正なし
    print(f"[Stage 3] 補正なし: {bpm:.1f} BPM")
    return bpm


def hybrid_bpm_detection(y: np.ndarray, sr: int) -> float:
    """
    ハイブリッド BPM 検出 (メイン関数)

    Args:
        y: 音声信号
        sr: サンプリングレート

    Returns:
        最終 BPM
    """
    print("=" * 60)
    print("ハイブリッド BPM 検出開始")
    print("=" * 60)

    # 第1段階: 初期推定
    candidates = initial_bpm_estimation(y, sr)

    if not candidates:
        print("[Hybrid] 候補なし、デフォルト 120 BPM を返す")
        return 120.0

    # 第2段階: 詳細スコアリング
    best_bpm, score = detailed_scoring(y, sr, candidates)

    # 第3段階: オクターブ検証
    final_bpm = octave_verification(y, sr, best_bpm)

    print("=" * 60)
    print(f"ハイブリッド BPM 検出完了: {final_bpm:.1f} BPM")
    print("=" * 60)

    return final_bpm


if __name__ == "__main__":
    # テストコード
    print("ハイブリッド BPM 検出のテスト")
    print("使用方法: テスト音声ファイルを指定して実行してください")
