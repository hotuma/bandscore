"""
BeatNet ライブラリの動作検証
BPM 検出と処理時間・メモリ消費量の測定
"""
import time
import psutil
import os
import librosa
import numpy as np


def get_memory_mb():
    """現在のプロセスのメモリ使用量（MB）を取得"""
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except:
        return 0


def test_beatnet(audio_file_path):
    """BeatNetでBPM検出テスト"""
    try:
        print(f"=== BeatNet テスト: {audio_file_path} ===")

        # メモリ初期値
        mem_start = get_memory_mb()
        print(f"初期メモリ: {mem_start:.1f} MB")

        # 音声ファイル読み込み
        print("音声ファイル読み込み中...")
        y, sr = librosa.load(audio_file_path, sr=44100)
        duration = len(y) / sr
        print(f"音声読み込み完了: {duration:.1f}秒 ({sr} Hz)")
        mem_after_load = get_memory_mb()
        print(f"読み込み後メモリ: {mem_after_load:.1f} MB (+{mem_after_load - mem_start:.1f} MB)")

        # BeatNet インポートと初期化
        print("\nBeatNet ロード中...")
        from BeatNet import BeatNet
        mem_before_beatnet = get_memory_mb()
        print(f"BeatNet インポート後メモリ: {mem_before_beatnet:.1f} MB (+{mem_before_beatnet - mem_after_load:.1f} MB)")

        # BeatNet 初期化
        print("BeatNet モデル初期化中...")
        processor = BeatNet(
            mode='offline',
            inference_model='DBN',
            plot=None
        )
        mem_after_init = get_memory_mb()
        print(f"BeatNet 初期化後メモリ: {mem_after_init:.1f} MB (+{mem_after_init - mem_before_beatnet:.1f} MB)")

        # BPM検出と処理時間測定
        print("\nBPM 検出中...")
        start_time = time.time()

        # BeatNetでビートトラッキング
        output = processor.process(y, sr)

        end_time = time.time()
        processing_time = end_time - start_time

        # メモリ最終値
        mem_final = get_memory_mb()
        print(f"BPM検出後メモリ: {mem_final:.1f} MB (+{mem_final - mem_after_init:.1f} MB)")

        # 結果出力
        print("\n=== 結果 ===")
        print(f"処理時間: {processing_time:.2f} 秒")
        print(f"メモリ増加: {mem_final - mem_start:.1f} MB")
        print(f"音声長: {duration:.1f} 秒")
        print(f"処理速度: {duration/processing_time:.2f} xリアルタイム")

        # 出力フォーマットを確認
        print(f"\n出力タイプ: {type(output)}")
        if isinstance(output, np.ndarray):
            print(f"出力形状: {output.shape}")
            print(f"出力サンプル: {output[:5]}")
        else:
            print(f"出力内容: {output}")

        return {
            'processing_time': processing_time,
            'memory_increase': mem_final - mem_start,
            'duration': duration,
            'realtime_factor': duration / processing_time,
            'output': output
        }

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # テスト用音声ファイルパス
    # ユーザーの環境にある実際のファイルを指定してください
    test_audio = "c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a"

    if os.path.exists(test_audio):
        print(f"テスト音声ファイルが見つかりました: {test_audio}")
        result = test_beatnet(test_audio)

        if result:
            print("\n=== 成功 ===")
            print(f"BPM 検出に成功しました！")
            print(f"処理時間: {result['processing_time']:.2f} 秒")
            print(f"メモリ増加: {result['memory_increase']:.1f} MB")
            print(f"処理速度: {result['realtime_factor']:.2f} xリアルタイム")
    else:
        print(f"テスト音声ファイルが見つかりません: {test_audio}")
        print("実際の音声ファイルパスを指定してください。")