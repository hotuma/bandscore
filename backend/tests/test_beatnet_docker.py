"""
BeatNet BPM検出のDocker用テストスクリプト
BeatNetを使用したBPM検出が正しく機能しているか確認
"""
import numpy as np
import librosa

def test_beatnet_bpm_detection():
    """BeatNet BPM検出テスト"""
    print("=== BeatNet BPM検出テスト ===")

    try:
        from BeatNet.BeatNet import BeatNet
        print(f"BeatNet imported successfully")

    except ImportError as e:
        print(f"[ERROR] BeatNet import failed: {e}")
        return None

    # テスト用音声の読み込み
    test_audio = "/app/audio/ガソリン0812.m4a"

    import os
    if not os.path.exists(test_audio):
        print(f"テスト音声ファイルが見つかりません: {test_audio}")
        return None

    print(f"\n音声ファイル読み込み: {test_audio}")
    y, sr = librosa.load(test_audio, sr=44100, duration=30.0)
    duration = len(y) / sr
    print(f"音声長: {duration:.1f} 秒, サンプリングレート: {sr} Hz")

    # BeatNetモデルの初期化
    print("\n=== BeatNetモデルの初期化 ===")

    try:
        # BeatNetの使用: model=3 (Rock_corpus), mode='offline', inference_model='DBN'
        tracker = BeatNet(model=3, mode='offline', inference_model='DBN')
        print("BeatNetモデル初期化完了 (Model 3: Rock_corpus, DBNモード)")

    except Exception as e:
        print(f"[ERROR] BeatNetモデル初期化失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

    # BPM検出実行
    print("\n=== BPM検出実行 ===")

    try:
        # ステレオ音声をモノラルに変換
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # BPM検出
        print("BPM検出中...")
        # resultは (beat_time, downbeat) の配列
        result = tracker.process(y)
        beats = result[:, 0]
        downbeats = result[result[:, 1] == 1, 0]  # ダウンビートのみ抽出

        print(f"検出されたビート数: {len(beats)}")
        print(f"検出されたダウンビート数: {len(downbeats)}")

        # BPM計算
        if len(beats) >= 2:
            intervals = np.diff(beats)
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval
            print(f"\n=== BeatNet検出結果 ===")
            print(f"ビート間隔の中央値: {median_interval:.4f} 秒")
            print(f"検出されたBPM: {bpm:.1f} BPM")
            print(f"検出されたビット数: {len(beats)}")

            # 期待値との比較（ガソリン0812.m4aは162 BPM）
            expected_bpm = 162.0
            error = abs(bpm - expected_bpm)
            print(f"\n期待値: {expected_bpm} BPM")
            print(f"誤差: {error:.1f} BPM ({error/expected_bpm*100:.1f}%)")

            if error < 5:
                print(f"[OK] 成功！ 期待値 {expected_bpm} BPM に非常に近いです")
            elif error < 10:
                print(f"[GOOD] 良好です。期待値 {expected_bpm} BPM との差が {error:.1f} BPM")
            elif error < 15:
                print(f"[ACCEPTABLE] 許容範囲です。期待値 {expected_bpm} BPM との差が {error:.1f} BPM")
            else:
                print(f"[WARNING] 注意: 期待値 {expected_bpm} BPM との差が {error:.1f} BPM あります")

            # 最初の数ビートを表示
            print(f"\n最初の10ビット:")
            for i, beat in enumerate(beats[:10]):
                print(f"  ビート {i+1}: {beat:.3f} 秒")

            if len(downbeats) > 0:
                print(f"\n最初の5ダウンビート:")
                for i, downbeat in enumerate(downbeats[:5]):
                    print(f"  ダウンビート {i+1}: {downbeat:.3f} 秒")

            return {
                'bpm': bpm,
                'beats_count': len(beats),
                'downbeats_count': len(downbeats),
                'error': error,
                'expected_bpm': expected_bpm
            }

        else:
            print("[ERROR] ビート検出に失敗（検出されたビート数が不足）")
            return None

    except Exception as e:
        print(f"[ERROR] BPM検出実行失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_beatnet_with_target_audio():
    """ターゲット楽曲（105 BPM）でのBeatNetテスト"""
    print("\n=== ターゲット楽曲でのBeatNetテスト ===")

    # ターゲット楽曲の音声ファイル
    target_audio = "/app/audio/05388afa7d224dbf846a1af4d61e2618-LaKp04a7hAM.m4a"

    import os
    if not os.path.exists(target_audio):
        print(f"ターゲット音声ファイルが見つかりません: {target_audio}")
        return None

    print(f"\nターゲット音声ファイル: {os.path.basename(target_audio)}")

    try:
        from BeatNet.BeatNet import BeatNet

        # BeatNetの使用: model=3 (Rock_corpus), mode='offline', inference_model='DBN'
        tracker = BeatNet(model=3, mode='offline', inference_model='DBN')

        # 音声読み込み
        y, sr = librosa.load(target_audio, sr=44100, duration=30.0)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        print(f"音声長: {len(y)/sr:.1f} 秒")

        # BPM検出
        print("BPM検出中...")
        result = tracker.process(y)
        beats = result[:, 0]

        if len(beats) >= 2:
            intervals = np.diff(beats)
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval

            print(f"\n=== ターゲット楽曲検出結果 ===")
            print(f"検出されたBPM: {bpm:.1f} BPM")
            print(f"検出されたビット数: {len(beats)}")

            # 期待値: 105 BPM
            expected_bpm = 105.0
            error = abs(bpm - expected_bpm)
            print(f"\n期待値: {expected_bpm} BPM")
            print(f"誤差: {error:.1f} BPM ({error/expected_bpm*100:.1f}%)")

            if error < 5:
                print(f"[EXCELLENT] 非常に正確！ 期待値 {expected_bpm} BPM とほぼ一致")
            elif error < 10:
                print(f"[GOOD] 良好です。期待値 {expected_bpm} BPM との差が {error:.1f} BPM")
            elif error < 15:
                print(f"[ACCEPTABLE] 許容範囲です。期待値 {expected_bpm} BPM との差が {error:.1f} BPM")
            else:
                print(f"[POOR] 改善の余地があります。期待値 {expected_bpm} BPM との差が {error:.1f} BPM")

            return bpm

        else:
            print("[ERROR] ビート検出に失敗")
            return None

    except Exception as e:
        print(f"[ERROR] ターゲット楽曲テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("BeatNet BPM検出テストスイート")
    print("=" * 60)

    # テスト1: 既存のテスト音声（ガソリン0812.m4a, 期待162 BPM）
    result1 = test_beatnet_bpm_detection()

    # テスト2: ターゲット楽曲（期待105 BPM）
    result2 = test_beatnet_with_target_audio()

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

    if result1:
        print(f"\nテスト1（ガソリン0812.m4a）: {result1['bpm']:.1f} BPM (誤差: {result1['error']:.1f} BPM)")

    if result2:
        print(f"テスト2（ターゲット楽曲）: {result2:.1f} BPM")
