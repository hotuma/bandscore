"""
拡張データセットを使用したPyTorch BPM検出モデルの訓練
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from audio_data_augmentation import AudioAugmentation, augment_dataset
from train_pytorch_bpm import AdvancedBPMDetector, train_model, evaluate_model

class AugmentedBPMDataset(Dataset):
    """拡張BPM検出用データセット"""
    def __init__(self, augmented_data, hop_length=512):
        self.data = augmented_data
        self.hop_length = hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        y = item['y']
        sr = item['sr']
        true_bpm = item['bpm']

        # オンセット強度の計算
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # 正規化
        onset_env_normalized = (onset_env - np.mean(onset_env)) / (np.std(onset_env) + 1e-8)

        # PyTorchテンソルに変換
        onset_tensor = torch.from_numpy(onset_env_normalized).float()

        # BPMをクラスインデックスに変換（60-240 BPM）
        bpm_class = int(true_bpm) - 60
        bpm_class = max(0, min(bpm_class, 180))  # 0-180の範囲に制限

        return onset_tensor, bpm_class

def main():
    """拡張データセットでの訓練"""
    print("=" * 60)
    print("拡張データセットでのBPM検出モデル訓練")
    print("=" * 60)

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # オリジナルデータ
    original_data = [
        ("c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a", 162.0),
        ("c:/Users/USER/.gemini/antigravity/guitar-tab/backend/temp/05388afa7d224dbf846a1af4d61e2618-LaKp04a7hAM.m4a", 105.0),
    ]

    # データ拡張（各ファイルにつき5つの拡張を作成）
    print("\n=== データ拡張中 ===")
    augmentation_result = augment_dataset(original_data, augmentations_per_file=5)

    # データセットの作成
    print("\n=== データセットの作成 ===")
    dataset = AugmentedBPMDataset(augmentation_result['data'])

    print(f"拡張データセットサイズ: {len(dataset)}")

    if len(dataset) < 4:
        print("警告: データセットが小さいです")
        train_dataset = dataset
        val_dataset = dataset
        test_dataset = dataset
    else:
        # データを分割
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.85 * total_size)

        train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
        val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, val_size)))
        test_dataset = torch.utils.data.Subset(dataset, list(range(val_size, total_size)))

    print(f"訓練セット: {len(train_dataset)} 検証セット: {len(val_dataset)} テストセット: {len(test_dataset)}")

    # データローダーの作成
    batch_size = min(4, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # モデルの作成
    sample_data = dataset[0][0]
    input_size = sample_data.shape[0]
    model = AdvancedBPMDetector(input_size).to(device)

    print(f"入力サイズ: {input_size}")
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # 訓練
    print("\n=== モデル訓練開始 ===")
    train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.0005, device=device)

    # ベストモデルの読み込み
    model.load_state_dict(torch.load('best_bpm_model.pth'))

    # 評価
    print("\n=== モデル評価 ===")
    results = evaluate_model(model, test_loader, device)

    # モデルの保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'results': results,
        'augmentation_info': augmentation_result
    }, 'bpm_model_augmented.pth')

    print(f"\n拡張モデルを保存しました: bpm_model_augmented.pth")

    # テスト推論（オリジナルデータのみ）
    print("\n=== テスト推論（オリジナルデータ）===")

    def predict_bpm(model, y, sr, device='cpu'):
        """学習済みモデルでのBPM予測"""
        model.eval()

        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_env_normalized = (onset_env - np.mean(onset_env)) / (np.std(onset_env) + 1e-8)

        onset_tensor = torch.from_numpy(onset_env_normalized).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(onset_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_bpm = predicted_class + 60
            confidence = probabilities[0][predicted_class].item()

        return predicted_bpm, confidence

    for audio_file, true_bpm in original_data:
        y, sr = librosa.load(audio_file, sr=22050, duration=30.0)
        predicted_bpm, confidence = predict_bpm(model, y, sr, device)
        error = abs(predicted_bpm - true_bpm)
        print(f"ファイル: {os.path.basename(audio_file)}")
        print(f"  正解BPM: {true_bpm:.1f} BPM")
        print(f"  予測BPM: {predicted_bpm:.1f} BPM (誤差: {error:.1f} BPM, 信頼度: {confidence:.3f})")

    print("\n" + "=" * 60)
    print("拡張データセット訓練完了")
    print("=" * 60)

if __name__ == "__main__":
    main()