"""
高度なPyTorchベースのBPM検出モデル
オンセット強度データを用いてBPMを直接検出するモデル
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class AdvancedBPMDetector(nn.Module):
    """
    高度なBPM検出モデル
    オンセット強度からBPMを予測
    """
    def __init__(self, input_size, num_bpm_classes=181):  # 60-240 BPM
        super(AdvancedBPMDetector, self).__init__()

        # 1D CNN for feature extraction
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Dense layers
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_bpm_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

def bpm_to_class(bpm, min_bpm=60, max_bpm=240):
    """BPMをクラスインデックスに変換"""
    return int(bpm - min_bpm)

def class_to_bpm(class_idx, min_bpm=60):
    """クラスインデックスをBPMに変換"""
    return min_bpm + class_idx

def detect_bpm_pytorch(audio_path, model=None, sr=22050, duration=30.0):
    """
    PyTorchモデルを使用してBPMを検出
    """
    print(f"音声ファイル読み込み: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    print(f"音声長: {len(y)/sr:.1f} 秒, サンプリングレート: {sr} Hz")

    # オンセット検出
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, units='frames',
        backtrack=True, pre_max=3, post_max=3
    )

    print(f"検出されたオンセット数: {len(onset_frames)}")

    # オンセット強度を正規化
    onset_env_normalized = (onset_env - np.mean(onset_env)) / (np.std(onset_env) + 1e-8)

    # PyTorchテンソルに変換
    onset_tensor = torch.from_numpy(onset_env_normalized).float().unsqueeze(0).unsqueeze(0)
    print(f"入力テンソル形状: {onset_tensor.shape}")

    # モデルがなければ作成
    if model is None:
        model = AdvancedBPMDetector(onset_tensor.shape[2])
        print("新しいモデルを作成しました")

    # 予測
    model.eval()
    with torch.no_grad():
        output = model(onset_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_bpm = class_to_bpm(predicted_class)
        confidence = probabilities[0][predicted_class].item()

    print(f"\n=== PyTorch BPM検出結果 ===")
    print(f"検出されたBPM: {predicted_bpm:.1f} BPM")
    print(f"信頼度: {confidence:.3f}")

    # Top 5候補を表示
    top5_probs, top5_indices = torch.topk(probabilities, 5)
    print(f"\nTop 5候補:")
    for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
        bpm = class_to_bpm(idx.item())
        print(f"  {i+1}. {bpm:.0f} BPM (信頼度: {prob:.3f})")

    return predicted_bpm, confidence, model

def test_pytorch_bpm_detection():
    """PyTorch BPM検出のテスト"""
    print("=" * 60)
    print("PyTorch BPM検出テスト")
    print("=" * 60)

    # テスト1: 既存のテスト音声（ガソリン0812.m4a, 期待162 BPM）
    print("\n=== テスト1: ガソリン0812.m4a (期待162 BPM) ===")
    test_audio_1 = "c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a"
    bpm_1, conf_1, model_1 = detect_bpm_pytorch(test_audio_1)

    if bpm_1:
        expected_bpm_1 = 162.0
        error_1 = abs(bpm_1 - expected_bpm_1)
        print(f"\n期待値: {expected_bpm_1} BPM")
        print(f"誤差: {error_1:.1f} BPM ({error_1/expected_bpm_1*100:.1f}%)")

        if error_1 < 10:
            print(f"[EXCELLENT] 非常に正確！ 期待値 {expected_bpm_1} BPM に近いです")
        elif error_1 < 20:
            print(f"[GOOD] 良好です。期待値 {expected_bpm_1} BPM との差が {error_1:.1f} BPM")
        else:
            print(f"[WARNING] 改善の余地があります。期待値 {expected_bpm_1} BPM との差が {error_1:.1f} BPM")

    # テスト2: ターゲット楽曲（期待105 BPM）
    print("\n=== テスト2: ターゲット楽曲 (期待105 BPM) ===")
    target_audio = "c:/Users/USER/.gemini/antigravity/guitar-tab/backend/temp/05388afa7d224dbf846a1af4d61e2618-LaKp04a7hAM.m4a"

    if os.path.exists(target_audio):
        bpm_2, conf_2, model_2 = detect_bpm_pytorch(target_audio, model=model_1)

        if bpm_2:
            expected_bpm_2 = 105.0
            error_2 = abs(bpm_2 - expected_bpm_2)
            print(f"\n期待値: {expected_bpm_2} BPM")
            print(f"誤差: {error_2:.1f} BPM ({error_2/expected_bpm_2*100:.1f}%)")

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

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)

    return bpm_1, bpm_2

if __name__ == "__main__":
    test_pytorch_bpm_detection()
