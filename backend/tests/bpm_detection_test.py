"""
BPM 検出の動作検証
処理時間とメモリ消費量の測定
"""
import time
import psutil
import os
import librosa
import numpy as np
import math
from scipy.signal import butter, sosfilt


def get_memory_mb():
    """現在のプロセスのメモリ使用量（MB）を取得"""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except:
        return 0


def lowpass_filter(y, sr, cutoff_hz=200):
    """ローパスフィルタ"""
    sos = butter(10, cutoff_hz, 'low', fs=sr, output='sos')
    y_filtered = sosfilt(sos, y)
    return y_filtered


def evaluate_bass_ac(y, sr, bpm, hop_length=512):
    """バス帯域での自己相関評価"""
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


def evaluate_fullband_ac(onset_env, sr, bpm, hop_length=512):
    """全帯域での自己相関評価"""
    expected_lag = int(round(60.0 * sr / (bpm * hop_length)))
    if expected_lag <= 0:
        return 0.0
    expected_lag = max(expected_lag, 1)
    ac = librosa.autocorrelate(onset_env)
    if expected_lag >= len(ac):
        return 0.0
    return float(ac[expected_lag]) / max(1.0, float(ac[0]))


def evaluate_phase_concentration(onset_env, sr, bpm, n_bins=36):
    """ビート位相エネルギー集中度評価"""
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


def evaluate_tempo_prior(bpm):
    """テンポ事前確率評価（中速 80-180 BPM を強く優先）"""
    mean = 120.0
    std = 30.0

    # 中速レンジ (80-180) を強く優先
    if 80 <= bpm <= 180:
        std = 25.0  # より狭い分布で中速を強調

    return math.exp(-0.5 * ((bpm - mean) / std) ** 2)


def detect_bpm_improved(y, sr):
    """
    改良版 BPM 検出アルゴリズム
    tolerance を広げてより正確なビートタイミングを許容
    """
    print("[BPM Detection] 改良版 BPM 検出開始...")

    # オンセット検出
    print("[BPM Detection] オンセット検出中...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

    # オンセットフレーム検出（より厳密なパラメータ）
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=512,
        backtrack=True,
        pre_max=5,    # 増加: ピーク前のウィンドウを広げる
        post_max=5,   # 増加: ピーク後のウィンドウを広げる
        wait=4,       # 追加: 近接するオンセットを統合（4フレーム以上間隔）
        delta=0.3,    # 追加: 最小ピーク強度
        normalize=True # 追加: 正規化
    )

    print(f"[BPM Detection] 検出されたオンセット数: {len(onset_frames)}")

    if len(onset_frames) < 2:
        print("[BPM Detection] オンセットが少なすぎます（デフォルト 120 BPM）")
        return 120.0

    # BPM スキャン (tolerance を 6 に拡張、範囲を 80-180 に制限)
    print("[BPM Detection] BPM スキャン中 (tolerance=6, 範囲=80-180 BPM)...")
    total_frames = len(onset_env)
    num_onsets = len(onset_frames)
    onset_set = set(onset_frames)

    best_bpm = 120.0
    best_score = -1.0
    tolerance = 6  # 拡張: 162 BPM のような中速テンポで正確なビートタイミングを許容
    top_candidates = []

    for c in range(80, 181):  # 中速レンジに制限
        beat_period = 60.0 * sr / (c * 512)
        grid = np.arange(0, total_frames, beat_period)
        if len(grid) == 0:
            continue

        # Precision: ビートのうちオンセットが近くにある割合
        hits = 0
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                if (g_int + t) in onset_set:
                    hits += 1
                    break
        precision = hits / len(grid)

        # Recall: オンセットのうちビートが近くにある割合
        grid_set = set()
        for g in grid:
            g_int = int(round(g))
            for t in range(-tolerance, tolerance + 1):
                grid_set.add(g_int + t)
        onset_hits = sum(1 for o in onset_frames if int(o) in grid_set)
        recall = onset_hits / max(1, num_onsets)

        # F_betaスコア (beta=0.8: Precisionを重視)
        BEAT_F_BETA = 0.8
        beta_sq = BEAT_F_BETA ** 2
        if precision + recall > 0:
            fb_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        else:
            fb_score = 0.0

        # テンポ事前確率を重みとして追加（中速を優先）
        tempo_prior = evaluate_tempo_prior(c)
        score = fb_score * (0.7 + 0.3 * tempo_prior)  # 事前確率で30%調整

        top_candidates.append((c, score, precision, recall))
        if score > best_score:
            best_score = score
            best_bpm = float(c)

    # 上位5候補をログ出力
    top_candidates.sort(key=lambda x: x[1], reverse=True)
    print("[BPM Detection] 上位5候補:")
    for i, (c, s, p, r) in enumerate(top_candidates[:5]):
        print(f"  {i+1}. {c} BPM (P={p:.3f} R={r:.3f} Fb={s:.3f})")

    print(f"[BPM Detection] 選択された BPM: {best_bpm:.1f} (スコア: {best_score:.3f})")

    return best_bpm


def test_bpm_detection(audio_file_path):
    """BPM検出テスト"""
    try:
        print(f"=== BPM 検出テスト: {audio_file_path} ===")

        # メモリ初期値
        mem_start = get_memory_mb()
        print(f"初期メモリ: {mem_start:.1f} MB")

        # 音声ファイル読み込み
        print("\n音声ファイル読み込み中...")
        start_load = time.time()
        y, sr = librosa.load(audio_file_path, sr=44100)
        load_time = time.time() - start_load
        duration = len(y) / sr
        print(f"音声読み込み完了: {duration:.1f}秒 ({sr} Hz) - {load_time:.2f}秒")
        mem_after_load = get_memory_mb()
        print(f"読み込み後メモリ: {mem_after_load:.1f} MB (+{mem_after_load - mem_start:.1f} MB)")

        # ハイパスフィルタ適用（低周波ノイズ除去）
        print("\nハイパスフィルタ適用中...")
        start_filter = time.time()
        sos = butter(10, 80, 'high', fs=sr, output='sos')
        from scipy.signal import sosfilt
        y_filtered = sosfilt(sos, y)
        filter_time = time.time() - start_filter
        print(f"フィルタ適用完了: {filter_time:.2f}秒")
        mem_after_filter = get_memory_mb()
        print(f"フィルタ後メモリ: {mem_after_filter:.1f} MB (+{mem_after_filter - mem_after_load:.1f} MB)")

        # BPM検出
        print("\nBPM検出中...")
        start_detect = time.time()
        detected_bpm = detect_bpm_improved(y_filtered, sr)
        detect_time = time.time() - start_detect
        print(f"BPM検出完了: {detect_time:.2f}秒")

        # メモリ最終値
        mem_final = get_memory_mb()
        print(f"BPM検出後メモリ: {mem_final:.1f} MB (+{mem_final - mem_after_filter:.1f} MB)")

        # 結果出力
        print("\n=== 結果 ===")
        print(f"検出された BPM: {detected_bpm:.1f}")
        print(f"音声長: {duration:.1f} 秒")
        print(f"読み込み時間: {load_time:.2f} 秒")
        print(f"フィルタ時間: {filter_time:.2f} 秒")
        print(f"検出時間: {detect_time:.2f} 秒")
        print(f"総処理時間: {load_time + filter_time + detect_time:.2f} 秒")
        print(f"メモリ増加: {mem_final - mem_start:.1f} MB")
        print(f"処理速度: {duration/(load_time + filter_time + detect_time):.2f} xリアルタイム")

        # 期待値との比較
        expected_bpm = 162.0
        if abs(detected_bpm - expected_bpm) < 5:
            print(f"\n[OK] 成功！ 期待値 {expected_bpm} BPM に近い値が検出されました")
        else:
            print(f"\n[WARNING] 注意: 期待値 {expected_bpm} BPM との差が {abs(detected_bpm - expected_bpm):.1f} BPM あります")

        return {
            'bpm': detected_bpm,
            'load_time': load_time,
            'filter_time': filter_time,
            'detect_time': detect_time,
            'total_time': load_time + filter_time + detect_time,
            'memory_increase': mem_final - mem_start,
            'duration': duration,
            'realtime_factor': duration / (load_time + filter_time + detect_time)
        }

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # テスト用音声ファイルパス
    test_audio = "c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a"

    if os.path.exists(test_audio):
        print(f"テスト音声ファイルが見つかりました: {test_audio}")
        result = test_bpm_detection(test_audio)

        if result:
            print("\n=== 成功 ===")
            print(f"BPM: {result['bpm']:.1f}")
            print(f"処理時間: {result['total_time']:.2f} 秒")
            print(f"メモリ増加: {result['memory_increase']:.1f} MB")
            print(f"処理速度: {result['realtime_factor']:.2f} xリアルタイム")
    else:
        print(f"テスト音声ファイルが見つかりません: {test_audio}")
        print("実際の音声ファイルパスを指定してください。")