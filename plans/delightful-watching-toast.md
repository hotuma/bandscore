# BPM検出精度改善 + 小節アライメント修正

## Context

2つの問題を修正:
1. **BPM精度**: F1スコアのStage 2を自己相関リファインメントに置換
2. **小節頭ズレ**: ビートグリッドの位相オフセット検出を追加し、小節が音楽の拍に合うようにする

## 実装済みの変更（backend/main.py）

### 変更1: onset計算の前出し
- onset_env/onset_frames_detectedをif/elseブロックの前に移動
- BPM検出と位相検出の両方で使用可能に

### 変更2: Stage 2 → 自己相関リファインメント（hop=128）
- 高解像度onset_envで自己相関を計算
- 放物線補間でサブフレーム精度
- **1 BPM未満の差はcoarse BPM（整数）を維持** → 184→183.5になる問題を解消

### 変更3: Stage 3 位相オフセット検出（新規追加）
- BPM確定後、全位相（0 ～ beat_period）をsweep
- **precision（割合）**で最適位相を選択（hit数比較だとphase=0が勝つバグを修正済み）
- tax.mp3で264.7msのオフセットを検出

### 変更4: beat_times生成の修正
- 旧: `librosa.frames_to_time(i * frames_per_segment, ...)` → フレーム量子化で1.115s間隔（本来0.652s）
- 新: `np.arange(phase_offset_sec, total_duration, target_segment_duration)` → 正確な時間ベース生成

## テスト結果（バックエンド単体）

```
BPM: 184.0（正確に維持）
Beat phase offset: 264.7ms
Bar 1: 0.265s - 1.569s (interval=1.304s = 4 beats at 184 BPM)
```

## 未検証事項

ユーザーのコンソールログは `barStart: 0.000, 間隔1.115s` を示しており、変更前のコードの結果。

### 確認すべきこと
1. バックエンドサーバーを再起動して変更を反映
2. フロントエンドで再分析を実行
3. コンソールログで barStart ≈ 0.265, 間隔 ≈ 1.304s を確認

## ファイル
- **backend/main.py**: 全変更済み（onset前出し、AC refinement、位相検出、beat_times修正）
- **frontend/**: 変更不要（L158のbadRate<0.3チェックにより、正常なbarはそのまま使用される）
