# Fix: ハイライト半拍ズレ + ガソリンBPM誤検出

## Context

「はいよろこんで」(BPM 146.9) のハイライトが小節頭ではなく4拍目の裏に表示される問題。

### 試行1: 半拍シフト候補追加 → 効果なし
半拍シフト [0.5,1.5,2.5,3.5] を追加したが、スコアリング自体が間違った位置を選ぶ:
```
shift=0.0: bass=0.2855, snare=0.2673, score=0.1518
shift=0.5: bass=0.1792, snare=0.2007, score=0.0788  ← 半拍は低スコア
shift=3.0: bass=0.2985, snare=0.2841, score=0.1565  ← 依然最高(間違い)
shift=3.5: bass=0.1605, snare=0.1657, score=0.0776  ← 正しい位置だが最低スコア
```

### 根本原因
曲のバスドラムがシンコペーションで裏拍に打っているため、
`score = bass - 0.5*snare`（ダウンビートのみ評価）では裏拍の方が高スコアになる。
**バー内のバックビートパターン（2&4拍目のスネア）を検出しないと補正不可能。**

## 変更ファイル

- [backend/main.py](backend/main.py) — `refine_bar_phase()` の Step B スコアリング

## 変更内容

### Step B のスコア式にバックビートスネア検出を追加

ポップ・ロックではスネアが2拍目と4拍目に入る（バックビートパターン）。
各候補のバー内で2&4拍目のスネアエネルギーを計測し、正の重みで加算する。

```python
# 変更前:
score = bass_avg - ALPHA * snare_avg

# 変更後:
score = bass_avg - ALPHA * snare_avg + BETA * backbeat_snare_avg
```

**`backbeat_snare_avg`**: 各バーのbeat 2 (bar_start + 1拍) と beat 4 (bar_start + 3拍) の
スネアエネルギー平均。BETA = 0.5。

### なぜこれで直るか

正しいアラインメント（バーが実際の beat 1 から開始）:
- bar_start (beat 1): バス低い（シンコペーション）→ 既存スコア低い
- beat 2, 4: **スネア高い（バックビート）** → 新スコア成分が**高い** ✓

間違ったアラインメント（バーが4拍目の裏から開始）:
- bar_start (and-of-4): バス高い（シンコペーション）→ 既存スコア高い
- "beat 2, 4"位置: 実際のbeat 1, 3 → スネア低い → 新スコア成分が**低い** ✗

バックビート成分が加わることで、正しい位置のスコアが逆転する。

### 半拍シフト候補（前回の変更）: 維持
8候補テストは引き続き有効。スコアリング改善と組み合わせて機能する。

---

## Issue 2: 「ガソリン」BPM誤検出 (231.4 → 正しくは ~178)

### 経緯
最初「BPMが早すぎる」をオクターブ倍速と誤解 → PHASE_GATEバイパスで115.7に → 遅すぎ。
正しいBPMは ~178。これはオクターブエラーではなく、検出精度の問題。

### 根本原因
BPMグリッドサーチ (60-240, lines 1170-1213) が234 BPMをTop候補に選ぶ:
- 全帯域オンセットがハイハット等の高速アーティキュレーションを拾う
- F-beta=0.8 (Precision重視) で密なグリッドほど有利
- 180 BPMはTop 5に入らない (低スコア)
- AC精緻化は±5%しか検索しない → 234から180に到達不可能

### 修正

#### Step 1: PHASE_GATEバイパスをリバート
前回追加したスコア差バイパスを削除し、元のPHASE_GATEロジックに戻す。
ファイル: [main.py:819-835](backend/main.py#L819-L835)

#### Step 2: バス自己相関によるテンポ補正を追加
`analyze_audio_file()` 内、AC精緻化とオクターブ検証の後に新ステップを追加。
バス帯域 (20-200Hz) のオンセットエンベロープの自己相関を計算し、
60-200 BPMの範囲で最大ピークを探索する。

```python
# --- Bass AC tempo correction ---
# 全帯域オンセットはハイハット等で高速テンポを検出しやすい。
# バスドラム (20-200Hz) の自己相関ピークが真のテンポを示す。
y_bass_temp = lowpass_filter(y, sr, cutoff_hz=200)
y_bass_temp = highpass_filter(y_bass_temp, sr, cutoff_hz=20)
bass_env_temp = librosa.onset.onset_strength(y=y_bass_temp, sr=sr, hop_length=512)
del y_bass_temp
ac_bass = librosa.autocorrelate(bass_env_temp)
if ac_bass[0] > 0:
    ac_bass = ac_bass / ac_bass[0]

best_bass_bpm = bpm
best_bass_val = 0.0
for candidate in range(60, 201, 2):  # 71 candidates
    lag = 60.0 * sr / (candidate * 512)
    lag_int = int(round(lag))
    if 0 < lag_int < len(ac_bass) - 1:
        val = (ac_bass[lag_int-1] + ac_bass[lag_int] + ac_bass[lag_int+1]) / 3.0
        if val > best_bass_val:
            best_bass_val = val
            best_bass_bpm = float(candidate)

# 検出BPMとバスBPMが大きくずれている場合、バスBPMを優先
if abs(best_bass_bpm - bpm) > 10:
    prior_bass = evaluate_tempo_prior(best_bass_bpm)
    prior_detected = evaluate_tempo_prior(bpm)
    bass_ac_detected_val = # ac_bass at detected bpm lag
    score_bass = best_bass_val * 0.6 + prior_bass * 0.4
    score_detected = bass_ac_detected_val * 0.6 + prior_detected * 0.4
    if score_bass > score_detected * 1.05:  # 5% advantage required
        print(f"[BassTempoCorrection] {bpm:.1f} → {best_bass_bpm:.1f} BPM")
        bpm = best_bass_bpm
```

**ガソリンの場合**:
- バスドラムは ~178 BPM で打っている
- バス自己相関のピークは ~178 BPM付近 (lag ≈ 14.6 frames)
- best_bass_bpm ≈ 178, detected bpm = 231.4 → 差 > 10
- bass_ac at 178 > bass_ac at 231 + prior加算 → 178 BPMに補正

### 安全性
- バスBPMと検出BPMの差が10以下なら何もしない
- 5%のスコア差が必要なので、微差では補正されない
- バス帯域のみを使うので、ハイハット等の影響を受けない
- メモリ: y_bass_temp は del で即解放

## 検証

1. バックエンドを再起動
2. 「はいよろこんで」: ログで `bb_snare` 確認、ハイライトが小節頭に合うか目視確認
3. 「ガソリン」: BPMが ~178 に検出されるか、`[BassTempoCorrection]` ログを確認
4. 既存の正しいBPM検出が壊れていないか確認
