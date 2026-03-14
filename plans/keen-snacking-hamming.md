# Fix: URL音源解析でハイライトが小節頭に合わない（Phase 2）

## Context

前回の修正（MP3変換廃止、チャンク位相修正、オフセット0化）適用後もハイライトがずれる。
新ログで確認: M4A直接配信とチャンク位相計算は正常動作しているが、**BarPhaseRefineが誤ったダウンビートを選択**している。

### 問題の分析

BarPhaseRefine Step Bのスコア:
```
shift=0.0: bass=0.2693, snare=0.0790, bb_snare=0.1080, score=0.2837
shift=1.0: bass=0.3328, snare=0.0992, bb_snare=0.0823, score=0.3244
shift=3.0: bass=0.3676, snare=0.1170, bb_snare=0.0832, score=0.3507 ← 選択（誤）
```

shift=3.0が選ばれるが、backbeat snare分析からshift=0が正しいダウンビート:
- shift=0/2の裏拍スネア: **0.108**（高い = 2,4拍にスネア有り = 正しいダウンビート）
- shift=1/3の裏拍スネア: **0.083**（低い = 裏拍にスネアが無い）

**根本原因**: 現在の `composite = bass_avg - 0.5*snare_avg + 0.5*bb_snare_avg` は**絶対バス値**で評価するため、バス+スネアが同時にヒットする位置（shift=3: bass=0.368, snare=0.117）を過大評価する。

## Changes

### 1. BarPhaseRefineのスコアリング式を改善 (backend/main.py)

**ファイル**: [backend/main.py:929-985](backend/main.py#L929-L985)

絶対バス値の代わりに**バス/スネア比率**を使用し、バス+スネア同時ヒット位置の過大評価を防ぐ。

```python
# Before (line 929-930):
ALPHA = 0.5
BETA = 0.5

# After:
BETA = 1.0

# Before (line 985):
composite = bass_avg - ALPHA * snare_avg + BETA * bb_snare_avg

# After:
bass_ratio = bass_avg / (bass_avg + snare_avg + 1e-8)
composite = bass_ratio + BETA * bb_snare_avg
```

**効果の検証** (このURL音源のデータで計算):
| shift | 旧score | 新score | 順位変化 |
|-------|---------|---------|----------|
| 0.0   | 0.2837  | **0.881**   | 4位→**1位** |
| 1.0   | 0.3244  | 0.852   | 2位→3位 |
| 2.0   | 0.2792  | 0.870   | 5位→2位 |
| 3.0   | **0.3507**  | 0.842   | **1位**→4位 |

shift=0.0が正しく1位に。shift=0とshift=2（beat 1とbeat 3）がトップ2になり、バックビートスネアパターンが適切に評価される。

ALPHAパラメータは不要になる（比率に内在）。`snare_active=False`の場合のフォールバックも更新。

### 2. オフセット調整範囲を拡大 (frontend)

**ファイル**: [frontend/components/ResultDisplay.tsx:730-735](frontend/components/ResultDisplay.tsx#L730-L735)

±0.5s → ±2.0sに拡大。181.6 BPMでは半小節=0.661sのため、位相修正しきれないエッジケースに対応。

```typescript
// Before:
Math.max(-0.5, Number((prev - 0.1).toFixed(1)))
Math.min(0.5, Number((prev + 0.1).toFixed(1)))

// After:
Math.max(-2.0, Number((prev - 0.1).toFixed(1)))
Math.min(2.0, Number((prev + 0.1).toFixed(1)))
```

## Critical Files

- [backend/main.py](backend/main.py) - `refine_bar_phase` 関数 (line 837-1006)
- [frontend/components/ResultDisplay.tsx](frontend/components/ResultDisplay.tsx) - オフセットUI (line 725-738)

## Verification

1. バックエンドを起動し、同じYouTube URLで再解析
2. ログで `[BarPhaseRefine] Final: shift=0.0` になることを確認（shift=3.0ではなく）
3. 位相が ~97.5ms（1088.7msではなく）になることを確認
4. 再生時にハイライトが小節頭（キックドラムのタイミング）と一致することを確認
5. オフセットUIで±2.0sまで調整可能であることを確認
