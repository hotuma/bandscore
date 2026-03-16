# Context

ユーザーが YouTube 動画（ZIEQDjrAdwE）を解析した際、コードが変わるべきタイミングで変わらない（同じコードが連続しすぎる）という問題が発生している。

バックエンドログで以下の異常が確認された：
```
Raw chords max stagnation: 3 bars
After stagnation-aware smoothing: 6 bars  ← スムージングで悪化
After break_long_stagnation_runs: 6 bars  ← 改善されていない
```

# 問題の根本原因

## Root Cause 1: `min_hold_segments=2` が変化を抑制
- `detect_chords_matrix` のデフォルト `min_hold_segments=2`（`backend/main.py` ~行527）
- 低flux時は直近の切り替え後 **最低2セグメント** は同じコードを維持する
- 128BPM・4beats/seg → 1セグメント = 1.875s → 最低 **3.75秒** は変化しない
- 実際の楽曲が1バー(1.875s)ごとにコードが変わる曲の場合、全変化が抑制される

## Root Cause 2: スムージングの `max_run=6` > 検出の `max_repeat_segments=4`
- `smooth_chord_sequence_stagnation_aware(raw_chords, passes=2, max_run=6)` （~行1874）
- この関数は「A-B-A → A-A-A」「A-B-B-A → A-A-A-A」の平滑化をする
- `max_run=6` なので、6バー以下ならどんな同コード連続も平滑化が許可される
- 結果：隣接する同コードのセグメントが合体し、3バーが6バーに膨れ上がる

## Root Cause 3: `break_long_stagnation_runs` の閾値が緩すぎる
- `break_long_stagnation_runs(smoothed_chords, max_consecutive=6)` （~行1880）
- 判定条件: `if run_length > max_consecutive` （**厳格な `>`**）
- 6バーの連続は `6 > 6 = False` → **検出されず何も変更されない**
- `max_consecutive=6` の閾値自体も `max_repeat_segments=4` と不整合

# 修正方針

3つのパラメータを `max_repeat_segments=4`（実績のある閾値）に揃えることで整合性を取る。

## 変更箇所

### 変更1: `min_hold_segments: 2 → 1`
- **ファイル**: [backend/main.py](backend/main.py)（~行527）
- **変更前**: `min_hold_segments: int = 2`
- **変更後**: `min_hold_segments: int = 1`
- **効果**: 1バー(1.875s)ごとにコード変化を許可。`high_flux_threshold` チェックがあるため極端なちらつきは防止される。

### 変更2: スムージングの `max_run: 6 → 4`
- **ファイル**: [backend/main.py](backend/main.py)（~行1874付近）
- **変更前**: `smooth_chord_sequence_stagnation_aware(raw_chords, passes=2, max_run=6)`
- **変更後**: `smooth_chord_sequence_stagnation_aware(raw_chords, passes=2, max_run=4)`
- **効果**: スムージングが4バー超の連続を生成しなくなる（`max_repeat_segments=4` と整合）

### 変更3: `break_long_stagnation_runs` の `max_consecutive: 6 → 4`
- **ファイル**: [backend/main.py](backend/main.py)（~行1880付近）
- **変更前**: `break_long_stagnation_runs(smoothed_chords, max_consecutive=6)`
- **変更後**: `break_long_stagnation_runs(smoothed_chords, max_consecutive=4)`
- **効果**: 4バー超の連続を実際に検出・破壊できる（`6 > 4 = True`）

# 影響範囲

- 変更は `backend/main.py` のみ
- フロントエンドの変更は不要（タイミング計算・ハイライトロジックは問題なし）
- 副作用の懸念：`min_hold_segments=1` でちらつきが増える可能性があるが、後段のスムージングが吸収する

# 検証方法

1. バックエンドサーバーを再起動
2. 同じ YouTube URL (`ZIEQDjrAdwE`) を再度解析
3. バックエンドログで以下を確認:
   - `After stagnation-aware smoothing: N bars` が 4 以下になること
   - `After break_long_stagnation_runs: N bars` が 4 以下になること
4. フロントエンドで再生し、コードが以前より頻繁に変わることを確認
