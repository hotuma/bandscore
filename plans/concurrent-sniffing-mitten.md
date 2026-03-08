# Phase 4: 倍速検出（Double-Tempo Detection）対策

## Context

### 発見された問題
「いんちき和尚のブルース」（遅いブルース曲）の解析で、BPM 186.2が検出されたが、実際のテンポは約93 BPM（半分）と推測される。これにより：
- ギタータブのハイライトが音楽の2倍速で進む
- タイミングが完全に不一致

### 根本原因
現在の`estimate_bpm_multi_candidate()`関数は：
1. オンセット検出で裏拍/装飾音も拾う
2. テンポ推定で高周波ピークを優先（F1スコアで評価）
3. 倍速/半分速の検証機能がない
4. ジャンル特有のテンポ範囲を考慮しない

ブルース/バラード/ジャズでは：
- シャッフルビート、オフビート、三連符が多用される
- オンセット検出が8分音符や16分音符を主拍と誤認識
- 結果：真のBPM（60-120）の倍速（120-240）を検出

### 既存の実装（Phase 1-3）の状態
✅ Phase 1: BPM検出精度向上（F1スコア、位相検出、統一グリッド）
✅ Phase 2: 累積ドリフト対策（AC refinement常時適用）
✅ Phase 3: 位相統一（forced_phase実装）→ 正常動作確認済み

しかし、いずれも**BPMが正しい前提**での改善。倍速検出には対応していない。

---

## 実装アプローチ

### Strategy 1: Octave Error Correction（推奨）

BPM検出後に倍速/半分速を検証する後処理を追加。

#### 実装箇所
`backend/main.py` の`estimate_bpm_multi_candidate()`関数（line ~536-609）の後に、新関数`verify_tempo_octave()`を追加。

#### アルゴリズム
```python
def verify_tempo_octave(y, sr, detected_bpm, onset_env, tempo_candidates):
    """
    倍速/半分速検出を検証し、最適なオクターブ（×1, ×2, ×0.5）を選択

    Args:
        y: audio signal
        sr: sample rate
        detected_bpm: 検出されたBPM
        onset_env: オンセットエンベロープ
        tempo_candidates: BPM候補リスト[(bpm, precision, recall, f1), ...]

    Returns:
        corrected_bpm: 修正されたBPM
        octave_factor: 適用された倍率（0.5, 1.0, 2.0）
    """
    candidates_to_test = [
        (detected_bpm * 0.5, 0.5),  # 半分速
        (detected_bpm * 1.0, 1.0),  # 現在の値
        (detected_bpm * 2.0, 2.0),  # 倍速
    ]

    # 範囲外のものを除外（40-240 BPM）
    candidates_to_test = [(bpm, f) for bpm, f in candidates_to_test if 40 <= bpm <= 240]

    scores = []
    for candidate_bpm, factor in candidates_to_test:
        # 1. ダウンビート一致度（低音域のオンセットとの一致）
        downbeat_score = evaluate_downbeat_alignment(y, sr, candidate_bpm)

        # 2. オンセット強度の分布（強拍vs弱拍の差）
        regularity_score = evaluate_onset_regularity(onset_env, sr, candidate_bpm)

        # 3. 音楽理論的な妥当性（ジャンル推定から期待されるBPM範囲）
        prior_score = evaluate_tempo_prior(candidate_bpm)

        total_score = (downbeat_score * 0.4 +
                       regularity_score * 0.4 +
                       prior_score * 0.2)

        scores.append((candidate_bpm, factor, total_score))

    # 最高スコアを選択
    best_bpm, best_factor, best_score = max(scores, key=lambda x: x[2])

    print(f"[OctaveVerify] Candidates: {scores}")
    print(f"[OctaveVerify] Selected: {best_bpm:.1f} BPM (factor={best_factor})")

    return best_bpm, best_factor
```

#### サブ関数の実装

**1. ダウンビート一致度**
```python
def evaluate_downbeat_alignment(y, sr, candidate_bpm):
    """低音域のビート（バスドラム）との一致度を評価"""
    # 低音域（20-200Hz）を抽出
    y_bass = librosa.effects.preemphasis(y, coef=-0.97)  # 低域強調
    y_bass = apply_highpass_filter(y, sr, cutoff=20)
    y_bass = apply_lowpass_filter(y_bass, sr, cutoff=200)

    # 低音域のオンセット
    bass_onset_env = librosa.onset.onset_strength(y=y_bass, sr=sr)
    bass_onset_times = librosa.frames_to_time(
        librosa.util.peak_pick(bass_onset_env, 3, 3, 3, 5, 0.5, 10),
        sr=sr
    )

    # candidate_bpmのグリッドとの一致度
    beat_interval = 60.0 / candidate_bpm
    tolerance = beat_interval * 0.15  # ±15%

    matches = 0
    for onset_time in bass_onset_times:
        # 最も近いビート位置との距離
        beat_position = onset_time % beat_interval
        distance = min(beat_position, beat_interval - beat_position)
        if distance < tolerance:
            matches += 1

    return matches / len(bass_onset_times) if len(bass_onset_times) > 0 else 0.0
```

**2. オンセット規則性**
```python
def evaluate_onset_regularity(onset_env, sr, candidate_bpm):
    """
    強拍位置のオンセット強度 vs 弱拍位置の強度を比較
    真のテンポなら強拍が強く、倍速なら差が小さい
    """
    beat_interval = 60.0 / candidate_bpm
    hop_length = 512

    strong_beat_strengths = []
    weak_beat_strengths = []

    for i in range(len(onset_env)):
        time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
        beat_phase = (time % beat_interval) / beat_interval

        # 強拍（0%, 25%, 50%, 75%付近）vs 弱拍（その間）
        if beat_phase < 0.1 or (0.24 < beat_phase < 0.26) or \
           (0.49 < beat_phase < 0.51) or (0.74 < beat_phase < 0.76):
            strong_beat_strengths.append(onset_env[i])
        elif 0.15 < beat_phase < 0.2 or 0.4 < beat_phase < 0.45:
            weak_beat_strengths.append(onset_env[i])

    if not strong_beat_strengths or not weak_beat_strengths:
        return 0.5

    # 強拍の平均強度 vs 弱拍の平均強度
    strong_mean = np.mean(strong_beat_strengths)
    weak_mean = np.mean(weak_beat_strengths)

    # 比率（1.0以上が理想、真のテンポなら大きい）
    ratio = strong_mean / (weak_mean + 1e-6)

    # 1.0-2.0の範囲にスケール
    score = min(ratio / 2.0, 1.0)

    return score
```

**3. テンポ事前分布**
```python
def evaluate_tempo_prior(bpm):
    """
    音楽理論的な妥当性スコア
    一般的な音楽の多くは60-140 BPMに集中
    """
    if 60 <= bpm <= 100:
        return 1.0  # バラード/ブルース/R&B
    elif 100 <= bpm <= 140:
        return 0.9  # ポップ/ロック
    elif 140 <= bpm <= 180:
        return 0.6  # アップテンポロック/EDM
    elif 180 <= bpm <= 240:
        return 0.3  # 倍速検出の可能性高い
    elif 40 <= bpm < 60:
        return 0.5  # バラード（遅め）
    else:
        return 0.1  # 異常値
```

#### 統合箇所
`analyze_audio_file()`関数内（line ~890-920付近）:

```python
# 既存のBPM検出（Phase 1実装）
if forced_bpm is None:
    raw_bpm, candidates = estimate_bpm_multi_candidate(y, sr, onset_env)

    # 自己相関リファインメント（Phase 2実装）
    if raw_bpm is not None:
        refined_bpm, lag, ac_val = autocorr_refine_bpm(y, sr, raw_bpm)
        delta = abs(refined_bpm - raw_bpm)
        if delta < 1:  # Phase 2: 1BPM未満の差でも採用
            bpm = refined_bpm
            print(f"[DEBUG] AC refined={refined_bpm:.1f} (delta<1), using refined {bpm:.1f}")
        else:
            bpm = raw_bpm

    # ★NEW: オクターブ検証（Phase 4）
    if bpm is not None:
        corrected_bpm, octave_factor = verify_tempo_octave(
            y, sr, bpm, onset_env, candidates
        )
        if octave_factor != 1.0:
            print(f"[OctaveCorrection] {bpm:.1f} → {corrected_bpm:.1f} BPM (×{octave_factor})")
            bpm = corrected_bpm
```

---

### Strategy 2: Multi-Scale Onset Detection（代替案）

複数の時間スケールでオンセット検出を行い、最も安定したスケールを選択。

**利点**: より根本的な解決
**欠点**: 計算コスト増加、実装複雑

Phase 4では**Strategy 1**を採用し、Strategy 2は将来の改善として保留。

---

## Critical Files

### 修正対象
- `backend/main.py`
  - Line ~610: `verify_tempo_octave()`関数を追加
  - Line ~640: `evaluate_downbeat_alignment()`関数を追加
  - Line ~670: `evaluate_onset_regularity()`関数を追加
  - Line ~700: `evaluate_tempo_prior()`関数を追加
  - Line ~920: `analyze_audio_file()`内でオクターブ検証を呼び出し

### 既存ユーティリティの再利用
- `apply_highpass_filter()` (line ~233) - 低域抽出に再利用
- 低域通過フィルタ: 新規実装が必要（`apply_lowpass_filter()`）

---

## Verification Plan

### 1. ユニットテスト
`backend/tests/verify_octave_detection.py`を作成：

```python
import librosa
import numpy as np
from main import verify_tempo_octave, estimate_bpm_multi_candidate

# テストケース1: 倍速検出の補正
def test_double_tempo_blues():
    y, sr = librosa.load("いんちき和尚のブルース.mp3", duration=60, sr=22050)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    raw_bpm, candidates = estimate_bpm_multi_candidate(y, sr, onset_env)
    print(f"Raw BPM: {raw_bpm}")

    corrected_bpm, factor = verify_tempo_octave(y, sr, raw_bpm, onset_env, candidates)
    print(f"Corrected BPM: {corrected_bpm} (factor={factor})")

    # 期待: 186 → 93（factor=0.5）
    assert 85 <= corrected_bpm <= 100, f"Expected ~93 BPM, got {corrected_bpm}"
    assert factor == 0.5

# テストケース2: 正常検出の維持（tax.mp3）
def test_normal_tempo_rock():
    y, sr = librosa.load("tax.mp3", duration=60, sr=22050)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    raw_bpm, candidates = estimate_bpm_multi_candidate(y, sr, onset_env)
    corrected_bpm, factor = verify_tempo_octave(y, sr, raw_bpm, onset_env, candidates)

    print(f"Raw: {raw_bpm}, Corrected: {corrected_bpm}, Factor: {factor}")

    # 期待: 183.5 → 183.5（factor=1.0、変更なし）
    assert factor == 1.0
    assert abs(corrected_bpm - raw_bpm) < 1
```

### 2. エンドツーエンド検証

#### ステップ1: バックエンド再起動
```bash
cd backend
uvicorn main:app --reload
```

#### ステップ2: 新規アップロード
- 「いんちき和尚のブルース」を再アップロード
- バックエンドログで以下を確認：
  ```
  [DEBUG] BPM candidate: 186 (P=0.749 R=0.665 F1=0.705)
  [DEBUG] AC refined=186.2
  [OctaveVerify] Candidates: [(93.1, 0.5, 0.85), (186.2, 1.0, 0.62), ...]
  [OctaveVerify] Selected: 93.1 BPM (factor=0.5)
  [OctaveCorrection] 186.2 → 93.1 BPM (×0.5)
  ```

#### ステップ3: フロントエンドで再生
- バー持続時間が 1.289s → **2.578s** に変更されることを確認
- ハイライトが音楽と同期することを確認

#### ステップ4: tax.mp3で回帰テスト
- tax.mp3を再アップロード
- BPM 183.5のまま変更されないことを確認（factor=1.0）

---

## Implementation Notes

### 1. 低域通過フィルタの実装
`apply_highpass_filter()`を参考に、`apply_lowpass_filter()`を追加：

```python
def apply_lowpass_filter(y, sr, cutoff=200, order=5):
    """Apply Butterworth low-pass filter"""
    sos = butter(order, cutoff, btype='low', fs=sr, output='sos')
    return sosfilt(sos, y)
```

### 2. メモリ考慮
- 低音域抽出は元の信号をコピーするため、メモリ使用量が一時的に増加
- 処理後は`y_bass`を即座に削除（`del y_bass`）
- 60秒チャンクでの処理なら問題なし（既存のメモリ管理で対応可能）

### 3. 診断タグの更新
`_build`タグに`build-v5`を追加：

```python
result["_build"] = "build-v5-octave-correction"
```

### 4. エッジケース処理
- BPM候補が40未満または240以上の場合：
  - オクターブ検証をスキップ
  - ログに警告を出力

```python
if bpm < 40 or bpm > 240:
    print(f"[WARNING] Extreme BPM detected: {bpm:.1f}, skipping octave verification")
    # そのまま使用
```

---

## Expected Outcomes

### Before (Phase 3まで)
- **いんちき和尚のブルース**: BPM 186.2、バー持続時間 1.289s → タイミング不一致
- **tax.mp3**: BPM 183.5、バー持続時間 1.308s → 正常

### After (Phase 4)
- **いんちき和尚のブルース**: BPM **93.1**、バー持続時間 **2.578s** → タイミング一致
- **tax.mp3**: BPM 183.5（変更なし）、バー持続時間 1.308s → 正常維持

### 汎用性向上
- ブルース、バラード、ジャズなど幅広いジャンルに対応
- 倍速/半分速検出の自動補正により、手動調整不要

---

## Alternative: Manual Override UI（将来の拡張）

もしアルゴリズムでの完全な解決が困難な場合、UIに以下を追加：

### フロントエンド（ResultDisplay.tsx）
```tsx
<button onClick={() => setBpm(bpm * 0.5)}>BPM ÷2 (現在倍速の場合)</button>
<button onClick={() => setBpm(bpm * 2.0)}>BPM ×2 (現在半分速の場合)</button>
```

しかし、これは応急処置であり、Phase 4の自動検出が優先される。

---

## Summary

Phase 4では、倍速検出（Double-Tempo Detection）問題を解決するため：

1. **オクターブ検証アルゴリズム**を実装（ダウンビート一致度、オンセット規則性、テンポ事前分布）
2. BPM検出後に×0.5、×1.0、×2.0の3候補を評価
3. 最適なテンポを自動選択
4. 「いんちき和尚のブルース」で BPM 186 → 93 への補正を確認
5. tax.mp3で回帰がないことを確認

これにより、幅広いジャンル・テンポの曲に対応できる。

---

## Phase 4.1: 実装後の問題修正

### 発見された問題

初期実装（build-v5）をテストした結果、以下の問題が判明：

#### 問題1: librosa API互換性エラー（致命的）
```
[WARNING] evaluate_downbeat_alignment failed: peak_pick() takes 1 positional argument but 7 were given
```

**原因**: `librosa.util.peak_pick()`のAPIが変更された。古いバージョンでは位置引数で7個渡していたが、新しいバージョンではキーワード引数のみを受け付ける。

**影響**: `evaluate_downbeat_alignment`が常に失敗し、`downbeat_score`が常に0.5（デフォルト値）になっている。ダウンビート一致度が正しく評価されていない。

#### 問題2: prior_scoreの偏りが大きすぎる

**実際のスコア**（バックエンドログより）:

| 曲 | 正解BPM | 半分速BPM | 半分速スコア | 元の値スコア | 選択 |
|----|---------|----------|------------|------------|------|
| いんちき和尚 | 93.1 | 93.1 | **0.458** | 0.392 | ✅ 正解 |
| tax | 183.5 | 91.8 | **0.593** | 0.486 | ❌ 誤り |

**原因分析**:
- `evaluate_tempo_prior(91.8)` → 0.9（バラード/ブルース範囲）
- `evaluate_tempo_prior(183.5)` → 0.3（倍速検出の可能性）
- 差が**0.6**と大きすぎる（重み0.2 × 0.6 = 0.12の差）
- tax（183.5 BPM）のような速い曲でも、prior_scoreが低すぎて半分速が選ばれてしまう

#### 問題3: 位相のずれ

ユーザーレポート:
> 小節頭には合っていないが、一定テンポでハイライトが進んでいる。曲の頭から合っていない。

**原因**: BPMが誤って半分になると、位相検出も誤った基準で行われるため、小節頭が合わなくなる。

---

### 修正内容

#### 修正1: peak_pick API修正（必須）

**ファイル**: `backend/main.py`
**関数**: `evaluate_downbeat_alignment` (line ~738)

```python
# 修正前（エラー）
bass_onset_times = librosa.frames_to_time(
    librosa.util.peak_pick(bass_onset_env, 3, 3, 3, 5, 0.5, 10),
    sr=sr
)

# 修正後
peaks = librosa.util.peak_pick(
    bass_onset_env,
    pre_max=3,
    post_max=3,
    pre_avg=3,
    post_avg=5,
    delta=0.5,
    wait=10
)
bass_onset_times = librosa.frames_to_time(peaks, sr=sr)
```

**期待される効果**: `downbeat_score`が正しく計算され、低音域のビートとの一致度が反映される。

#### 修正2: prior_scoreの調整

**オプションA: スコア範囲を狭める（推奨）**

```python
def evaluate_tempo_prior(bpm: float) -> float:
    """
    音楽理論的な妥当性スコア。
    範囲を狭めて、極端な偏りを防ぐ。
    """
    if 60 <= bpm <= 100:
        return 0.7  # 1.0 → 0.7（下げる）
    elif 100 < bpm <= 140:
        return 0.7  # 0.9 → 0.7
    elif 140 < bpm <= 180:
        return 0.6  # 0.6 → 0.6（維持）
    elif 180 < bpm <= 240:
        return 0.5  # 0.3 → 0.5（大幅に上げる）
    elif 40 <= bpm < 60:
        return 0.5  # 0.5 → 0.5（維持）
    else:
        return 0.1  # 0.1（維持）
```

**変更点**:
- 60-140 BPM: 0.7（以前は0.9-1.0）
- 180-240 BPM: 0.5（以前は0.3）
- **最大差**: 0.7 - 0.5 = **0.2**（以前は0.7）

**オプションB: 重みを下げる（代替案）**

```python
# verify_tempo_octave関数内
total_score = (downbeat_score * 0.5 +    # 0.4 → 0.5（上げる）
               regularity_score * 0.4 +  # 0.4（維持）
               prior_score * 0.1)        # 0.2 → 0.1（下げる）
```

**推奨**: **オプションA**（スコア範囲を狭める）
- より柔軟
- 速い曲（180+ BPM）への偏見を減らす
- ダウンビート一致度が修正されれば、より正確な判定が可能

#### 修正3: ビルドタグ更新

```python
"_build": "build-v5.1-octave-fix"
```

診断用に、修正版であることを明示。

---

### 修正後の期待結果

#### いんちき和尚のブルース
- BPM: **93.1**（半分速、正解）
- スコア: 半分速が元の値を上回る
- 位相: 正しく検出される
- ハイライト: 小節頭と一致

#### tax.mp3
- BPM: **183.5**（元の値維持、正解）
- スコア:
  - 半分速 (91.8): downbeat + regularity + prior(0.7) = 約0.5
  - 元の値 (183.5): downbeat + regularity + prior(0.5) = 約0.5-0.6
  - **元の値が選ばれる**（downbeat/regularityで差がつく）
- 位相: 正しく検出される
- ハイライト: 小節頭と一致

---

### 実装ステップ

1. ✅ `evaluate_downbeat_alignment`の`peak_pick`呼び出しを修正
2. ✅ `evaluate_tempo_prior`のスコア範囲を調整
3. ✅ ビルドタグを`build-v5.1-octave-fix`に更新
4. 🔄 いんちき和尚のブルースで検証（BPM 93.1維持、位相一致）
5. 🔄 tax.mp3で検証（BPM 183.5に戻る、位相一致）

---

## Final Summary

Phase 4.1では、Phase 4初期実装の問題を修正：

1. **librosa API互換性**: `peak_pick()`の呼び出し方を修正
2. **prior_scoreの偏り**: スコア範囲を0.5-0.7に狭める（以前は0.3-1.0）
3. **位相検出**: BPMが正しくなることで、位相も正しく検出される

これにより、**いんちき和尚のブルース**と**tax.mp3**の両方で正しいBPMと位相が検出され、ハイライトが小節頭と一致する。
