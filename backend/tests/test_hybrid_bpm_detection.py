"""
ハイブリッド BPM 検出のテストスクリプト
処理時間とメモリ消費量の測定
"""
import time
import psutil
import os
import librosa
from scipy.signal import butter, sosfilt


def get_memory_mb():
    """現在のプロセスのメモリ使用量（MB）を取得"""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except:
        return 0


def test_hybrid_bpm_detection(audio_file_path):
    """ハイブリッド BPM 検出テスト"""
    try:
        print(f"=== ハイブリッド BPM 検出テスト: {audio_file_path} ===")

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
        y_filtered = sosfilt(sos, y)
        filter_time = time.time() - start_filter
        print(f"フィルタ適用完了: {filter_time:.2f}秒")
        mem_after_filter = get_memory_mb()
        print(f"フィルタ後メモリ: {mem_after_filter:.1f} MB (+{mem_after_filter - mem_after_load:.1f} MB)")

        # ハイブリッド BPM 検出
        print("\nハイブリッド BPM 検出中...")
        start_detect = time.time()

        # ハイブリッド BPM 検出モジュールをインポート
        from hybrid_bpm_detection import hybrid_bpm_detection

        detected_bpm = hybrid_bpm_detection(y_filtered, sr)
        detect_time = time.time() - start_detect
        print(f"\nBPM検出完了: {detect_time:.2f}秒")

        # メモリ最終値
        mem_final = get_memory_mb()
        print(f"BPM検出後メモリ: {mem_final:.1f} MB (+{mem_final - mem_after_filter:.1f} MB)")

        # 結果出力
        print("\n" + "=" * 60)
        print("=== 最終結果 ===")
        print("=" * 60)
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
        error = abs(detected_bpm - expected_bpm)
        if error < 5:
            print(f"\n[OK] 成功！ 期待値 {expected_bpm} BPM に近い値が検出されました (差: {error:.1f} BPM)")
        elif error < 10:
            print(f"\n[GOOD] 良好です。期待値 {expected_bpm} BPM との差が {error:.1f} BPM です")
        else:
            print(f"\n[WARNING] 注意: 期待値 {expected_bpm} BPM との差が {error:.1f} BPM あります")

        return {
            'bpm': detected_bpm,
            'load_time': load_time,
            'filter_time': filter_time,
            'detect_time': detect_time,
            'total_time': load_time + filter_time + detect_time,
            'memory_increase': mem_final - mem_start,
            'duration': duration,
            'realtime_factor': duration / (load_time + filter_time + detect_time),
            'error': error
        }

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_old_method(audio_file_path):
    """既存の方法と比較テスト"""
    print("\n" + "=" * 60)
    print("=== 既存の方法との比較 ===")
    print("=" * 60)

    # 既存の方法でテスト
    print("\n既存の方法で検出中...")
    start_old = time.time()
    from hybrid_bpm_detection import initial_bpm_estimation

    # 音声読み込み
    y, sr = librosa.load(audio_file_path, sr=44100)
    sos = butter(10, 80, 'high', fs=sr, output='sos')
    y_filtered = sosfilt(sos, y)

    # 既存の方法（第1段階のみ）
    old_candidates = initial_bpm_estimation(y_filtered, sr, tolerance=6, bpm_range=(140, 170))
    old_bpm = old_candidates[0][0] if old_candidates else 120.0
    old_time = time.time() - start_old

    print(f"既存の方法: {old_bpm:.1f} BPM (処理時間: {old_time:.2f}秒)")

    # ハイブリッド方法でテスト
    print("\nハイブリッド方法で検出中...")
    start_hybrid = time.time()
    from hybrid_bpm_detection import hybrid_bpm_detection

    hybrid_bpm = hybrid_bpm_detection(y_filtered, sr)
    hybrid_time = time.time() - start_hybrid

    print(f"ハイブリッド方法: {hybrid_bpm:.1f} BPM (処理時間: {hybrid_time:.2f}秒)")

    # 比較
    expected_bpm = 162.0
    old_error = abs(old_bpm - expected_bpm)
    hybrid_error = abs(hybrid_bpm - expected_bpm)

    print("\n比較結果:")
    print(f"  既存方法: {old_bpm:.1f} BPM (誤差: {old_error:.1f} BPM, 時間: {old_time:.2f}s)")
    print(f"  ハイブリッド: {hybrid_bpm:.1f} BPM (誤差: {hybrid_error:.1f} BPM, 時間: {hybrid_time:.2f}s)")
    print(f"  精度改善: {old_error - hybrid_error:.1f} BPM")
    print(f"  時間増加: {hybrid_time - old_time:.2f}秒")

    return {
        'old_bpm': old_bpm,
        'hybrid_bpm': hybrid_bpm,
        'old_time': old_time,
        'hybrid_time': hybrid_time,
        'old_error': old_error,
        'hybrid_error': hybrid_error
    }


if __name__ == "__main__":
    # テスト用音声ファイルパス
    test_audio = "c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a"

    if os.path.exists(test_audio):
        print(f"テスト音声ファイルが見つかりました: {test_audio}")

        # ハイブリッド BPM 検出テスト
        result = test_hybrid_bpm_detection(test_audio)

        if result:
            print("\n" + "=" * 60)
            print("=== 成功 ===")
            print(f"BPM: {result['bpm']:.1f}")
            print(f"処理時間: {result['total_time']:.2f} 秒")
            print(f"メモリ増加: {result['memory_increase']:.1f} MB")
            print(f"処理速度: {result['realtime_factor']:.2f} xリアルタイム")

        # 既存の方法との比較
        print("\n")
        compare_result = compare_with_old_method(test_audio)

    else:
        print(f"テスト音声ファイルが見つかりません: {test_audio}")
        print("実際の音声ファイルパスを指定してください。")