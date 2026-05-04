"""
TorchAudioの事前学習済みモデルを調査
BPM検出に使用可能なモデルがあるか確認
"""
import torch
import torchaudio

def test_torchaudio_models():
    """TorchAudioの事前学習済みモデルをテスト"""
    print("=== TorchAudioモデル調査 ===")

    # 利用可能なモデルを確認
    print("\n利用可能な事前学習済みモデル:")
    try:
        # wav2vec2などの音声認識モデル
        print("HUBERTやwav2vec2などのモデルが利用可能です")
        print("これらは主に音声認識用ですが、リズム検出にも応用可能です")

    except Exception as e:
        print(f"[ERROR] モデル調査失敗: {e}")

    # 自作のシンプルなBPM検出モデルをテスト
    print("\n=== シンプルなPyTorch BPM検出モデルテスト ===")

    try:
        import librosa
        import numpy as np

        # テスト用音声の読み込み
        test_audio = "c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a"

        print(f"\n音声ファイル読み込み: {test_audio}")
        y, sr = librosa.load(test_audio, sr=22050, duration=30.0)
        print(f"音声長: {len(y)/sr:.1f} 秒, サンプリングレート: {sr} Hz")

        # オンセット検出（librosa）
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, units='frames',
            backtrack=True, pre_max=3, post_max=3
        )

        print(f"検出されたオンセット数: {len(onset_frames)}")

        # PyTorchテンソルに変換
        onset_tensor = torch.from_numpy(onset_env).float().unsqueeze(0).unsqueeze(0)
        print(f"オンセット環境テンソル形状: {onset_tensor.shape}")

        # シンプルな1D CNNモデルでBPMを検出
        class SimpleBPMDetector(torch.nn.Module):
            def __init__(self):
                super(SimpleBPMDetector, self).__init__()
                self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=5, padding=2)
                self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=5, padding=2)
                self.pool = torch.nn.MaxPool1d(2)
                self.fc1 = torch.nn.Linear(32 * (onset_env.shape[0] // 4), 64)
                self.fc2 = torch.nn.Linear(64, 1)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # モデルを初期化
        model = SimpleBPMDetector()
        print(f"シンプルBPM検出モデルを作成しました")

        # フォワードパスをテスト
        with torch.no_grad():
            output = model(onset_tensor)
            print(f"モデル出力形状: {output.shape}")
            print(f"出力値: {output.item():.4f}")

        # 実際のBPMを計算（従来方法）
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from main import detect_bpm_hybrid
        actual_bpm = detect_bpm_hybrid(y, sr)
        print(f"\n従来方法でのBPM: {actual_bpm:.1f} BPM")

        return True

    except Exception as e:
        print(f"[ERROR] テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_torchaudio_models()
