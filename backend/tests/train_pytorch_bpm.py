"""
PyTorch BPM検出モデルの訓練スクリプト
既存の音声データセットを使用してモデルを訓練
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
import json

class BPMDataset(Dataset):
    """BPM検出用データセット"""
    def __init__(self, data_list, sr=22050, duration=30.0, hop_length=512):
        self.data_list = data_list
        self.sr = sr
        self.duration = duration
        self.hop_length = hop_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_path, true_bpm = self.data_list[idx]

        # 音声読み込み
        y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)

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

class AdvancedBPMDetector(nn.Module):
    """高度なBPM検出モデル"""
    def __init__(self, input_size, num_bpm_classes=181):
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
        # データ次元を追加 (batch_size, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

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

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, device='cpu'):
    """
    モデルの訓練
    """
    print(f"訓練開始（デバイス: {device}）")

    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # 訓練モード
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 勾配をゼロ化
            optimizer.zero_grad()

            # フォワードパス
            output = model(data)
            loss = criterion(output, target)

            # バックワードパスと最適化
            loss.backward()
            optimizer.step()

            # 統計
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        # 訓練統計
        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / train_total

        # 検証モード
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        # 学習率スケジューラ
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # ベストモデルを保存
            torch.save(model.state_dict(), 'best_bpm_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    print('訓練完了')
    return model

def evaluate_model(model, test_loader, device='cpu'):
    """
    モデルの評価
    """
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_accuracy = 100. * test_correct / test_total

    # BPM誤差の計算
    bpm_errors = []
    for pred, target in zip(all_predictions, all_targets):
        pred_bpm = pred + 60
        target_bpm = target + 60
        error = abs(pred_bpm - target_bpm)
        bpm_errors.append(error)

    mean_error = np.mean(bpm_errors)
    median_error = np.median(bpm_errors)

    print(f'\n=== テスト結果 ===')
    print(f'正解率: {test_accuracy:.2f}%')
    print(f'平均BPM誤差: {mean_error:.1f} BPM')
    print(f'中央値BPM誤差: {median_error:.1f} BPM')

    return {
        'accuracy': test_accuracy,
        'mean_error': mean_error,
        'median_error': median_error
    }

def prepare_training_data():
    """
    訓練用データセットの準備
    既存の音声ファイルと既知のBPMを使用
    """
    print("=== 訓練データ準備 ===")

    # データリスト（音声ファイルパス, 正解BPM）
    data_list = [
        # 既存のテストデータ
        ("c:/Users/USER/.gemini/antigravity/guitar-tab/ガソリン0812.m4a", 162.0),
        ("c:/Users/USER/.gemini/antigravity/guitar-tab/backend/temp/05388afa7d224dbf846a1af4d61e2618-LaKp04a7hAM.m4a", 105.0),
    ]

    # その他の音声ファイルを追加（正解BPMが分かっているもの）
    temp_dir = "c:/Users/USER/.gemini/antigravity/guitar-tab/backend/temp"
    if os.path.exists(temp_dir):
        import glob
        m4a_files = glob.glob(os.path.join(temp_dir, "*.m4a"))
        print(f"見つかった音声ファイル: {len(m4a_files)}個")

        # ここでは正解BPMが分からないため、実際の訓練では手動でBPMを指定する必要があります
        # デモ用にランダムなBPMを割り当て（実際には正確なBPMを指定してください）
        # for audio_file in m4a_files[:3]:  # 最初の3つだけ使用
        #     random_bpm = np.random.randint(80, 180)
        #     data_list.append((audio_file, float(random_bpm)))

    print(f"訓練データ数: {len(data_list)}")

    # データセットの分割
    if len(data_list) < 4:
        # データが少ない場合、交差検証用に全データ使用
        print("警告: 訓練データが少ないため、評価の信頼性が低いです")
        train_list = data_list
        val_list = data_list
        test_list = data_list
    else:
        # データを分割
        np.random.shuffle(data_list)
        split1 = int(len(data_list) * 0.7)
        split2 = int(len(data_list) * 0.85)

        train_list = data_list[:split1]
        val_list = data_list[split1:split2]
        test_list = data_list[split2:]

    print(f"訓練セット: {len(train_list)} 検証セット: {len(val_list)} テストセット: {len(test_list)}")

    return train_list, val_list, test_list

def main():
    """メイン訓練関数"""
    print("=" * 60)
    print("BPM検出モデルの訓練")
    print("=" * 60)

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # データの準備
    train_list, val_list, test_list = prepare_training_data()

    if len(train_list) == 0:
        print("エラー: 訓練データがありません")
        return

    # データセットの作成
    train_dataset = BPMDataset(train_list)
    val_dataset = BPMDataset(val_list)
    test_dataset = BPMDataset(test_list)

    # データローダーの作成
    batch_size = min(2, len(train_list))  # データが少ない場合は小さなバッチサイズ
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # モデルの作成
    sample_data = train_dataset[0][0]
    input_size = sample_data.shape[0]
    model = AdvancedBPMDetector(input_size).to(device)

    print(f"入力サイズ: {input_size}")
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # 訓練
    train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, device=device)

    # ベストモデルの読み込み
    model.load_state_dict(torch.load('best_bpm_model.pth'))

    # 評価
    results = evaluate_model(model, test_loader, device)

    # モデルの保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'results': results
    }, 'bpm_model_final.pth')

    print(f"\nモデルを保存しました: bpm_model_final.pth")

    # テスト用の推論関数
    def predict_bpm(model, audio_path, device='cpu'):
        """学習済みモデルでのBPM予測"""
        model.eval()

        # 音声読み込み
        y, sr = librosa.load(audio_path, sr=22050, duration=30.0)

        # オンセット強度の計算
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_env_normalized = (onset_env - np.mean(onset_env)) / (np.std(onset_env) + 1e-8)

        # テンソルに変換
        onset_tensor = torch.from_numpy(onset_env_normalized).float().unsqueeze(0).to(device)

        # 予測
        with torch.no_grad():
            output = model(onset_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_bpm = predicted_class + 60
            confidence = probabilities[0][predicted_class].item()

        return predicted_bpm, confidence

    # テスト推論
    print("\n=== テスト推論 ===")
    for audio_file, true_bpm in test_list:
        predicted_bpm, confidence = predict_bpm(model, audio_file, device)
        error = abs(predicted_bpm - true_bpm)
        print(f"ファイル: {os.path.basename(audio_file)}")
        print(f"  正解BPM: {true_bpm:.1f} BPM")
        print(f"  予測BPM: {predicted_bpm:.1f} BPM (誤差: {error:.1f} BPM, 信頼度: {confidence:.3f})")

    print("\n" + "=" * 60)
    print("訓練完了")
    print("=" * 60)

if __name__ == "__main__":
    main()