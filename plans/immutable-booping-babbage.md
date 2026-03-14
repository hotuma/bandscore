# コード解析精度改善: Rule B 長期停滞修正

## Context

前回の修正（stagnation引き継ぎ、キー投票、bass_weight調整）は実装済み。
再解析の結果、以下の新たな問題が判明:
- **G#maj7が16連続bars** (bar 63-78): stagnation引き継ぎでチャンク境界をまたいで同一コードが長期継続
- **Fm7が14-16連続bars** (bar 87-100, bar 139-154): チャンク内でも`max_repeat_segments=6`が効かない

**根本原因**: Rule B (行610-638) で、flux が低く confidence gap > 0.10 の場合、コードが無制限に継続できる。

修正対象ファイル: [backend/main.py](backend/main.py) の `detect_chords_matrix()` 関数内 Rule B

---

## 修正: Rule B に Progressive Gap Escalation + Hard Cap を追加

**現状の Rule B (行610-638)**:
```python
if run_length >= max_repeat_segments:
    if delta[i] >= flux_threshold:
        # Case 1: High Flux - heavy penalty
        row[last] -= long_stag_penalty
        best2 = int(np.argmax(row))
        chosen = best2
    else:
        # Case 2: Low Flux - only switch if gap <= 0.10
        if gap <= 0.10:
            chosen = cand2
        else:
            chosen = best  # ← 無制限に継続可能!
```

**修正後の Rule B**:

3段階で停滞を解消:
1. **Progressive Gap Escalation**: `excess = run_length - max_repeat_segments` に応じて gap threshold を `0.10 + 0.03 * excess` に拡大
2. **Hard Cap**: `run_length >= max_repeat_segments * 2` (=12bars) で `long_stag_penalty` を無条件適用
3. High Flux は既存と同じ

```python
        # ---- Rule B: Long stagnation (UX protection) - strongest intervention
        if run_length >= max_repeat_segments:
            excess = run_length - max_repeat_segments

            # Hard cap: at 2x max_repeat_segments, force switch regardless
            hard_cap = max_repeat_segments * 2

            if run_length >= hard_cap:
                # Absolute limit - apply heavy penalty unconditionally
                row[last] = row[last] - long_stag_penalty
                best2 = int(np.argmax(row))
                chosen = best2

            elif delta[i] >= flux_threshold:
                # Case 1: High Flux (unchanged)
                row[last] = row[last] - long_stag_penalty
                best2 = int(np.argmax(row))
                chosen = best2 if best2 != last else best2

            # Case 2: Low Flux - progressive gap escalation
            else:
                if len(topk_idx) >= 2:
                    cand2 = int(topk_idx[1])
                    gap = scores[i, best] - scores[i, cand2]
                    adjusted_gap_threshold = 0.10 + 0.03 * excess
                    if gap <= adjusted_gap_threshold:
                        chosen = cand2
                    else:
                        chosen = best
                else:
                    chosen = best
```

**Gap Threshold の推移**:
| run_length | excess | gap threshold | 効果 |
|------------|--------|---------------|------|
| 6 | 0 | 0.10 | 既存と同じ |
| 7 | 1 | 0.13 | やや緩和 |
| 8 | 2 | 0.16 | |
| 9 | 3 | 0.19 | |
| 10 | 4 | 0.22 | 多くのケースで切り替え発生 |
| 12 | 6 | **Hard Cap** | 無条件でペナルティ適用 |

**変更しない箇所**: Rule C (min hold), Rule A (high flux), 関数シグネチャ、戻り値

---

## 検証方法

1. バックエンドサーバーを起動:
   ```bash
   cd backend && .\.venv\Scripts\activate && uvicorn main:app --reload
   ```

2. 対象曲で再解析し、以下を確認:
   - G#maj7 の最大連続bars が12以下に制限されているか
   - Fm7 の最大連続bars が12以下に制限されているか
   - コード変化が不自然でないか（progressive escalation が段階的に効いているか）
