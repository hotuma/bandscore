# BPM検出誤補正・音声同期ズレ修正プラン（+ stale closure 修正）

## Context

YouTube URL (https://youtu.be/fWHnghPgg4Q) の解析時、ハイライトとコード音再生が原曲とズレている。
デバッグログから、BPM検出パイプラインに2段階の誤補正が発生していることを確認。

## 根本原因

### 障害連鎖

```
onset検出: BPM候補 216-238 BPM (実際の正解 ~110 BPM の 2倍速を検出)
  ↓
OctaveVerify: score_half(0.499) > score_full(0.400) → 半テンポが明確に優位
  ↓
しかし phase_half(0.044) < PHASE_GATE(0.25) でブロック → 219.9 BPM を維持 ← 問題1
  ↓
BassTempoCheck が実行 (bpm=219.9 > 200 なので起動)
  ↓
bass peak = 158 BPM、ratio=158/220=0.718、|0.718-0.75|=0.032 < TOLERANCE(0.06) → valid ← 問題2
  ↓
158 BPM に誤補正 (正解 ~110 BPM からさらに遠ざかる)
```

### 問題1: `OctaveVerify` の PHASE_GATE が厳しすぎる
- `backend/main.py:1431-1436`
- スコア比較で 110 BPM が明確に優位 (0.499 vs 0.400、差25%) なのにブロックされる
- phase_half の絶対値ではなく、スコア差で判断するべき

### 問題2: `BassTempoCheck` が ratio=0.75 を許容
- `backend/main.py:1935`
- VALID_RATIOS に 0.75 (3/4倍速) が含まれているが、音楽的に非常に稀
- ratio=0.718 が 0.75 に近いとして不正に valid 判定される

### 問題3: フロントエンドのデフォルトオフセット未実装
- `frontend/components/ResultDisplay.tsx:38-40`
- CLAUDE.md に「YouTube時 +0.2s」と記載されているが実装されていない

---

## 修正内容

### Fix A: OctaveVerify にスコアオーバーライドを追加 (最重要)

**ファイル**: `backend/main.py`
**行**: 1431-1436

```python
# 変更前
PHASE_GATE = 0.25
if phase_half < PHASE_GATE:
    print(f"[OctaveVerify] Half-tempo phase={phase_half:.3f} < {PHASE_GATE} "
          f"(weak beat structure) -> keeping {detected_bpm:.1f}")
    print(f"[OctaveVerify] Selected: {detected_bpm:.1f} BPM (factor=x1.0, score={score_full:.3f})")
    return detected_bpm, 1.0
```

```python
# 変更後
PHASE_GATE = 0.25
SCORE_OVERRIDE_RATIO = 1.2  # スコア差が20%以上ならPHASE_GATEをバイパス
score_ratio = score_half / max(score_full, 1e-9)
score_override = score_ratio >= SCORE_OVERRIDE_RATIO

if phase_half < PHASE_GATE and not score_override:
    print(f"[OctaveVerify] Half-tempo phase={phase_half:.3f} < {PHASE_GATE} "
          f"(weak beat structure, score_ratio={score_ratio:.2f} < {SCORE_OVERRIDE_RATIO}) "
          f"-> keeping {detected_bpm:.1f}")
    print(f"[OctaveVerify] Selected: {detected_bpm:.1f} BPM (factor=x1.0, score={score_full:.3f})")
    return detected_bpm, 1.0
elif phase_half < PHASE_GATE and score_override:
    print(f"[OctaveVerify] Half-tempo phase={phase_half:.3f} < {PHASE_GATE} "
          f"BUT score_ratio={score_ratio:.2f} >= {SCORE_OVERRIDE_RATIO} "
          f"(score strongly favors half-tempo) -> overriding gate")
```

**効果**: score_half=0.499 / score_full=0.400 = 1.25 >= 1.2 → オーバーライドして 110 BPM を選択。
BassTempoCheck は bpm=110 < 200 のため実行されなくなる。

---

### Fix B: BassTempoCheck から ratio=0.75 を除外 (補助防衛)

**ファイル**: `backend/main.py`
**行**: 1935

```python
# 変更前
VALID_RATIOS = [0.5, 1.0/3, 0.25, 2.0/3, 0.75]

# 変更後
VALID_RATIOS = [0.5, 1.0/3, 0.25, 2.0/3]  # 0.75 (3/4倍速) を除外: 音楽的に非常に稀
```

**効果**: Fix A が他の曲で効かないケースでも、ratio≈0.75 の誤補正を防ぐ二重防衛。

---

### Fix C: YouTube URL時のデフォルトオフセット +0.2s

**ファイル**: `frontend/components/ResultDisplay.tsx`
**行**: 36-40

```typescript
// 変更前
// M4A直接配信により、librosaとブラウザが同一フォーマットをデコードするため
// 系統的なオフセット補償は不要。ユーザーは手動で微調整可能（±0.5sスライダー）
useEffect(() => {
    setOffsetSec(0);
}, [audioUrl]);

// 変更後
// http/https URL (バックエンド配信 = yt-dlp 経由) の場合は librosa と
// ブラウザのデコード差を補償するため +0.2s をデフォルトとして適用。
// blob: URL はローカルファイル直接再生のため補正不要。
useEffect(() => {
    const isRemoteAudio = audioUrl?.startsWith('http://') || audioUrl?.startsWith('https://');
    setOffsetSec(isRemoteAudio ? 0.2 : 0);
}, [audioUrl]);
```

---

### Fix D: スケジューラー tick の `offsetSec` stale closure 修正 (コード音停止の修正)

**ファイル**: `frontend/components/ResultDisplay.tsx`
**問題**: Fix C で `offsetSec = 0.2` が設定されたが、スケジューラー `tick` 関数は依存配列 `[autoChord, safeBars]` のみのため、`offsetSec = 0` のまま stale closure になる。

**症状の連鎖**:
1. `onPlay`/`onSeeked`: フレッシュな `offsetSec = 0.2` で `schedulingBarRef` を初期化
   - 例: audio = 5.0s → `t = 5.0 - 0.2 = 4.8` → bar1 (4.188-6.371) → `schedulingBarRef = 1`
2. `tick`: stale `offsetSec = 0` → `tAnalysis = 5.0 - 0 = 5.0`
3. `delta = 4.188 - 5.0 = -0.812 < -0.1` → **bar1 スキップ！**
4. bar2 (6.371s): `delta > 0.2` → break → 次バーまで無音

**修正**: `offsetSec` を `useRef` に同期し、`tick` がリアルタイム値を参照できるようにする。

**追加する行** (行25の`offsetSec` state定義の直後):
```typescript
const offsetSecRef = useRef<number>(0);
useEffect(() => { offsetSecRef.current = offsetSec; }, [offsetSec]);
```

**変更する行** (行528):
```typescript
// 変更前
const tAnalysis = tBrowser - offsetSec;

// 変更後
const tAnalysis = tBrowser - offsetSecRef.current;
```

---

### Fix E: BarPhaseRefine Step B に shift-magnitude ペナルティを追加

**ファイル**: `backend/main.py`
**行**: 1613-1619

**問題**: Step B の「ステータスクオ保護」閾値が一律 2% のため、大きなシフト（3.5ビート=1.9秒）でも 5.74% の改善率で採用されてしまう。

**このケースの計算**:
- shift=3.5 beats (→2005ms): improvement = (0.6619-0.6260)/0.6260 = **5.74%** > 2% → 採用
- shift=0 (→95.2ms): 0% improvement
- 結果: 2秒近く遅れたフェーズが選ばれ「バーが遅れている」症状が発生

**修正**: シフト量に比例した必要改善率 (+2%/ビート) を要求

```python
# 変更前 (行1613-1619)
# ステータスクオ保護: 2%未満の改善なら変更しない
if best_idx != 0:
    improvement = (scores_b[best_idx] - scores_b[0]) / (abs(scores_b[0]) + 1e-8)
    if improvement < 0.02:
        print(f"[BarPhaseRefine] Step B: shift={best_shift} "
              f"improvement={improvement:.3f} < 0.02, keeping shift=0")
        best_shift = 0.0

# 変更後
# ステータスクオ保護: 大きなシフトほど高い改善率を要求 (基準2% + シフト量×2%/beat)
if best_idx != 0:
    improvement = (scores_b[best_idx] - scores_b[0]) / (abs(scores_b[0]) + 1e-8)
    SHIFT_PENALTY_PER_BEAT = 0.02  # 大きなシフトを抑制: 1ビートあたり追加2%の改善を要求
    required_improvement = 0.02 + abs(best_shift) * SHIFT_PENALTY_PER_BEAT
    if improvement < required_improvement:
        print(f"[BarPhaseRefine] Step B: shift={best_shift} "
              f"improvement={improvement:.3f} < required={required_improvement:.3f}, keeping shift=0")
        best_shift = 0.0
```

**修正後の計算**:
- shift=3.5: required = 0.02 + 3.5×0.02 = **9%**, actual 5.74% < 9% → 却下 ✓
- shift=0.5: required = 0.02 + 0.5×0.02 = 3%, 1.3% < 3% → 却下
- → shift=0 (phase=95.2ms) が使われ、最初のバーが曲の冒頭に近くなる

---

### Fix F: 30sチャンクの target_segment_duration 計算バグ修正

**ファイル**: `backend/main.py`
**行**: 2097-2098

**問題**: forced_phase パス（2チャンク目以降）で `target_segment_duration` が常に「2beat分」で計算される。`beats_per_seg` が4でも seg_dur は 2 beat 相当になるため、ログと実際の計算が不一致になる。

```python
# 変更前 (行2097-2098)
beat_duration = 60.0 / bpm
target_segment_duration = beat_duration * 2  # 常に2beat固定！
```

```python
# 変更後
beat_duration = 60.0 / bpm
target_segment_duration = beat_duration * beats_per_seg  # beats_per_segに合わせて計算
```

**効果**:
- 30sチャンクの seg_dur が 1.091s（2beat）→ 2.183s（4beat）に修正される
- ログと実際の計算が一致する
- ChunkMerge と同じ計算結果になる

---

### Fix G: BarPhaseRefine Step B の改善率閾値を緩和

**ファイル**: `backend/main.py`
**行**: 1613-1621

**問題**: Fix E では `SHIFT_PENALTY_PER_BEAT = 0.02` に設定したが、これでは正当な shift=3.5（2005ms）も却下されてしまう。

```python
# Fix E の計算（現状）
required_improvement = 0.02 + 3.5 × 0.02 = 0.09 (9%)
actual_improvement = 5.74%
→ 5.74% < 9% → 却下
```

**修正**: ハイブリッドな改善率計算を使用（相対改善率3%以上 または 絶対改善率0.01以上）

```python
# 変更前 (行1613-1621)
if best_idx != 0:
    improvement = (scores_b[best_idx] - scores_b[0]) / (abs(scores_b[0]) + 1e-8)
    SHIFT_PENALTY_PER_BEAT = 0.02
    required_improvement = 0.02 + abs(best_shift) * SHIFT_PENALTY_PER_BEAT
    if improvement < required_improvement:
        print(f"[BarPhaseRefine] Step B: shift={best_shift} "
              f"improvement={improvement:.3f} < required={required_improvement:.3f}, keeping shift=0")
        best_shift = 0.0

# 変更後
if best_idx != 0:
    # ハイブリッド改善率: 相対改善率（相対的な上昇率）または絶対改善率（実数値差）の大きい方
    improvement = (scores_b[best_idx] - scores_b[0]) / (abs(scores_b[0]) + 1e-8)
    improvement_abs = scores_b[best_idx] - scores_b[0]

    # ステータスクオ保護: シフト量に応じて相対改善率の要求を上げるが、絶対改善率が十分なら通す
    # 3%以上の相対改善率 または 0.01以上の絶対改善率が必要
    REQUIRED_RELATIVE_IMPROVEMENT = 0.03
    REQUIRED_ABSOLUTE_IMPROVEMENT = 0.01

    if improvement < REQUIRED_RELATIVE_IMPROVEMENT and improvement_abs < REQUIRED_ABSOLUTE_IMPROVEMENT:
        print(f"[BarPhaseRefine] Step B: shift={best_shift} "
              f"improvement={improvement:.3f} < {REQUIRED_RELATIVE_IMPROVEMENT:.3f}, "
              f"abs={improvement_abs:.4f} < {REQUIRED_ABSOLUTE_IMPROVEMENT:.3f}, keeping shift=0")
        best_shift = 0.0
```

**修正後の計算** (shift=3.5):
- improvement = 5.74% >= 3% ✓
- improvement_abs = 0.0359 >= 0.01 ✓
- → 採用され、phase=2005ms が使われる

**効果**:
- 明確なスコア改善を持つ大きなシフトが採用される
- 2005ms（3.5ビート後）のフェーズが正しく選択される
- 音声冒頭の無音部分がスキップされ、実際のビート開始に同期する

---

## 変更ファイル一覧

| Fix | ファイル | 行 | リスク |
|-----|---------|-----|--------|
| A | `backend/main.py` | 1431-1436 | 低 (スコアが明確な場合のみ適用) |
| B | `backend/main.py` | 1935 | 最低 (1行削除) |
| C | `frontend/components/ResultDisplay.tsx` | 38-40 | 最低 (URL判定のみ) |
| D | `frontend/components/ResultDisplay.tsx` | 25付近と528行 | 最低 (ref同期の追加のみ) |
| E | `backend/main.py` | 1613-1619 | 低 (既存保護ロジックの拡張のみ) |
| F | `backend/main.py` | 2097-2098 | 最低 (計算修正のみ) |
| G | `backend/main.py` | 1613-1621 | 低 (閾値調整のみ) |

## 検証方法

1. バックエンドを再起動 (`uvicorn main:app --reload`)
2. 同じ YouTube URL を再解析
3. ログで以下を確認:
   - `[OctaveVerify] ... score_ratio=1.25 >= 1.2 (score strongly favors half-tempo) -> overriding gate`
   - `[OctaveVerify] Selected: 110.0 BPM (factor=x0.5, ...)`
   - BassTempoCheck ログが **出ない** こと (bpm=110 < 200 でスキップ)
   - 30sチャンクのログに `seg_dur=2.183s`（Fix F 適用確認）
   - `[BarPhaseRefine] Step B: shift=3.5` が採用される（Fix G 適用確認）
4. フロントエンドで再生し、ハイライトと原曲の同期を確認
5. `backend/tests/verify_modes.py` を実行して回帰確認
