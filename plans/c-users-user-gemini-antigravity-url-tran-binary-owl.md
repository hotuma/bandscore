# BPM検出精度改善計画

## Context

UVERworld「just Melody」を解析した際、本来125 BPMであるべきところが162 BPMと誤検出された問題を修正する。

**問題の分析**:
- 解析結果: 161.99997445595014 BPM（実際は約162 BPM）
- 正解: 125 BPM
- 誤差率: 約30%の過大評価

**根本原因**:

1. **バス帯域チェックが動作していない**（main.py:2155）
   - 現在: `if bpm is not None and bpm > 200:` の条件で実行
   - 162 BPMは200以下なので、バス帯域チェックがスキップされる
   - バスドラム/ベースの周期性チェックが実行されず、誤検出が修正されない

2. **ハイブリッドBPM検出のバイアス**（main.py:3171）
   - `evaluate_tempo_prior_hybrid` で140-170 BPMの範囲を優先
   - 162 BPMがこの範囲内のため、過度に高評価されている可能性

3. **オクターブ補正の限界**（main.py:1526）
   - 2倍、0.5倍、1.5倍の補正のみ対応
   - 162 BPM → 125 BPM（約1.3倍）は補正対象外

## Implementation Plan

### 修正1: バス帯域チェックの閾値を下げる

**ファイル**: `backend/main.py`

**変更箇所**: 行2155付近

```python
# 変更前
if bpm is not None and bpm > 200:

# 変更後
if bpm is not None and bpm > 140:
```

**理由**: 140 BPMを超える場合にバス帯域チェックを実行することで、162 BPMのような誤検出を修正できるようにする。

### 修正2: ハイブリッドBPM検出のターゲット範囲を調整

**ファイル**: `backend/main.py`

**変更箇所**: 行3171付近

```python
# 変更前
def evaluate_tempo_prior_hybrid(bpm: float, target_range: tuple = (140, 170)) -> float:

# 変更後
def evaluate_tempo_prior_hybrid(bpm: float, target_range: tuple = (110, 150)) -> float:
```

**理由**: 140-170 BPMへの強いバイアスを削減し、より中速（110-150 BPM）を優先するようにする。

### 修正3: オクターブ補正の範囲を拡張（オプション）

**ファイル**: `backend/main.py`

**変更箇所**: 行1532付近

```python
# 変更前
if detected_bpm < 80 or detected_bpm > 240:

# 変更後
if detected_bpm < 60 or detected_bpm > 240:
```

**理由**: 80 BPM未満の補正制限を緩和し、より広範なオクターブ補正を可能にする。

### 修正4: テンポ事前確率スコアの調整（オプション）

**ファイル**: `backend/main.py`

**変更箇所**: 行1411-1433

```python
# 変更前
if 60 <= bpm <= 100:
    return 0.7
elif 100 < bpm <= 140:
    return 0.7
elif 140 < bpm <= 180:
    return 0.6
elif 180 < bpm <= 240:
    return 0.5

# 変更後（中速をより強調）
if 60 <= bpm <= 100:
    return 0.7
elif 100 < bpm <= 140:
    return 0.8  # 中速を強調
elif 140 < bpm <= 180:
    return 0.5  # やや高速を抑制
elif 180 < bpm <= 240:
    return 0.4  # 高速を抑制
```

## Verification

1. **テストファイルの準備**:
   - UVERworld「just Melody」のMP3ファイルを使用

2. **テスト実行**:
   ```bash
   cd backend
   python -c "
   from main import analyze_audio_file
   result = analyze_audio_file('C:/Users/USER/.gemini/antigravity/url_transrater/downloads/UVERworld　『just Melody』.mp3')
   print(f'Detected BPM: {result[\"metadata\"][\"bpm\"]}')
   print(f'Expected BPM: 125')
   print(f'Error: {abs(result[\"metadata\"][\"bpm\"] - 125)} BPM')
   "
   ```

3. **成功基準**:
   - 検出BPMが125 BPM ±5 BPM（120-130 BPM）の範囲内であること

## Critical Files

- `backend/main.py:2155` - バス帯域チェックの閾値
- `backend/main.py:3171` - ハイブリッドBPM検出のターゲット範囲
- `backend/main.py:1411` - テンポ事前確率スコア

## Notes

- ユーザーが選択したテキスト「madmom/BeatNet」は、より高度なBPM検出ライブラリを示唆している可能性があります。将来的にはBeatNetの導入も検討可能ですが、まずは既存のロジック改善から始めます。
- BeatNetを使用する場合、Dockerfile.beatnetが既に存在しているため、その実装も検討可能です。
