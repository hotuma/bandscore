# ハイブリッド BPM 検出の設計

## 目的
librosa ベースの BPM 検出の精度を向上させるため、複数の検出手法を組み合わせるハイブリッドアプローチを実装する。

## アーキテクチャ

### 第1段階: 初期 BPM 推定 (高速)
**目的**: 効率的な初期推定で候補 BPM を絞り込む

**手法**:
1. オンセット検出 (標準パラメータ)
2. BPM スキャン (60-180 BPM, tolerance=4)
3. F_beta スコアによる上位候補選択 (top 10)
4. テンポ事前確率による重み付け

**パラメータ**:
- tolerance: 4 (バランス重視)
- 範囲: 60-180 BPM
- top_k: 10
- F_beta: 0.8 (Precision 重視)

**出力**: 上位10候補 BPM + スコア

### 第2段階: 詳細スコアリング (中精度)
**目的**: 候補 BPM の信頼性を詳細評価

**手法**:
1. 各候補 BPM について、複数の評価指標を計算
2. 評価指標の統合スコアを計算
3. 最も信頼性の高い BPM を選択

**評価指標**:
1. **バス帯域自己相関** (weight=0.30)
   - バスドラム/ベースのビートパターン評価
   - 200 Hz 以下の低周波数帯域

2. **全帯域自己相関** (weight=0.20)
   - 全体的なリズム構造の評価
   - 440 Hz 以下の中高周波数帯域

3. **位相エネルギー集中度** (weight=0.20)
   - ビート位相でのエネルギー集中度
   - 明確な拍構造の存在を確認

4. **オンセット整合性** (weight=0.15)
   - グリッド上のオンセット分布の均一性
   - 外れ値オンセットの割合

5. **テンポ事前確率** (weight=0.15)
   - 一般的な BPM (80-140) の優先
   - ユーザー調整を考慮

**パラメータ**:
- バス帯域カットオフ: 200 Hz
- 位相ビン数: 36 (10度刻み)

**出力**: 最も信頼性の高い BPM + 統合スコア

### 第3段階: オクターブ検証 (高精度)
**目的**: 倍速/半速誤検出を修正

**手法**:
1. 選択された BPM のオクターブ候補を生成
2. 各候補について、バス帯域でのスコアを再計算
3. ゲート条件により補正を制御

**オクターブ候補**:
- 原則: BPM × 1.0
- 半速: BPM × 0.5 (80 BPM 以上)
- 倍速: BPM × 2.0 (240 BPM 以下)

**ゲート条件**:
1. **位相エネルギー集中度**: > 0.25
   - 半速候補に明確な拍構造がある場合のみ補正

2. **スコアオーバーライド**: ratio >= 1.15
   - 半速スコアが元スコアより 15% 以上高い場合

**補正ルール**:
- ゲート条件を満たす場合: 半速に補正
- 満たさない場合: 元の BPM を維持

**出力**: 最終 BPM (オクターブ補正済み)

## 統合アルゴリズム

```
入力: 音声信号 y, サンプリングレート sr

# 第1段階: 初期推定
candidates = initial_bpm_estimation(y, sr)
# 出力: [(BPM, score), ...] (top 10)

# 第2段階: 詳細スコアリング
best_bpm = detailed_scoring(y, sr, candidates)
# 出力: best_bpm + 統合スコア

# 第3段階: オクターブ検証
final_bpm = octave_verification(y, sr, best_bpm)
# 出力: final_bpm

出力: final_bpm
```

## 実装詳細

### 関数構成

```python
def initial_bpm_estimation(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """
    第1段階: 初期 BPM 推定
    入力: 音声信号, サンプリングレート
    出力: [(BPM, スコア), ...] (top 10)
    """

def detailed_scoring(y: np.ndarray, sr: int,
                   candidates: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    第2段階: 詳細スコアリング
    入力: 音声信号, サンプリングレート, 候補 BPM リスト
    出力: (最適 BPM, 統合スコア)
    """

def octave_verification(y: np.ndarray, sr: int, bpm: float) -> float:
    """
    第3段階: オクターブ検証
    入力: 音声信号, サンプリングレート, 候補 BPM
    出力: 最終 BPM (オクターブ補正済み)
    """

def hybrid_bpm_detection(y: np.ndarray, sr: int) -> float:
    """
    ハイブリッド BPM 検出 (メイン関数)
    入力: 音声信号, サンプリングレート
    出力: 最終 BPM
    """
```

### 補助関数

```python
def evaluate_bass_ac(y: np.ndarray, sr: int, bpm: float, hop_length: int = 512) -> float:
    """バス帯域自己相関評価"""

def evaluate_fullband_ac(onset_env: np.ndarray, sr: int, bpm: float,
                       hop_length: int = 512) -> float:
    """全帯域自己相関評価"""

def evaluate_phase_concentration(onset_env: np.ndarray, sr: int, bpm: float,
                               n_bins: int = 36) -> float:
    """位相エネルギー集中度評価"""

def evaluate_onset_consistency(onset_frames: np.ndarray, sr: int,
                              bpm: float, tolerance: int = 4) -> float:
    """オンセット整合性評価"""

def evaluate_tempo_prior(bpm: float, target_range: Tuple[float, float] = (80, 140)) -> float:
    """テンポ事前確率評価"""

def lowpass_filter(y: np.ndarray, sr: int, cutoff_hz: int = 200) -> np.ndarray:
    """ローパスフィルタ"""
```

## パラメータ最適化

### 初期推定のパラメータ
- tolerance: 4 (バランス重視)
- BPM 範囲: 60-180
- top_k: 10
- F_beta: 0.8

### 詳細スコアリングのパラメータ
- バス帯域カットオフ: 200 Hz
- 位相ビン数: 36
- 重みバランス:
  - バス自己相関: 0.30
  - 全帯域自己相関: 0.20
  - 位相集中度: 0.20
  - オンセット整合性: 0.15
  - テンポ事前確率: 0.15

### オクターブ検証のパラメータ
- 位相ゲート: 0.25
- スコアオーバーライド: 1.15

## 性能目標

### 処理時間
- 初期推定: < 2秒 (120秒音声)
- 詳細スコアリング: < 3秒
- オクターブ検証: < 1秒
- **合計**: < 6秒 (120秒音声)

### 精度目標
- 162 BPM 検出: ±5 BPM 以内
- オクターブ誤検出: < 5%
- 一般的な BPM (80-140): ±3 BPM 以内

### メモリ消費
- 増加量: < 50 MB (既存に追加)

## テスト計画

### 単体テスト
1. 各段階の個別テスト
2. 補助関数の単体テスト

### 統合テスト
1. サンプル音声でのテスト
2. 異なる BPM (80, 120, 162) でのテスト
3. 異なるジャンルでのテスト

### 性能テスト
1. 処理時間の測定
2. メモリ消費量の測定
3. 精度の評価

## 進捗

- [x] 設計ドキュメントの作成
- [ ] 第1段階の実装
- [ ] 第2段階の実装
- [ ] 第3段階の実装
- [ ] main.py への統合
- [ ] テストと検証