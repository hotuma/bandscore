# BPM検出・ビート位相の修正

## 修正対象ファイル

- [backend/main.py](backend/main.py)

---

## Issue A: オクターブ補正後のビート位相ずれ（いんちき和尚）✅ 解決済み

**問題**: BPM 186→93にオクターブ補正されると、位相が半拍ずれてハイライトが1拍目の裏に合ってしまう

**実装済みの修正**:
1. `octave_factor = 1.0` 初期化（~1133行目）
2. `refine_bar_phase`に`octave_factor`パラメータ追加（837行目）
3. 位相検出をオクターブ補正前のBPM（186.2）で実行（~1244行目）
4. `octave_factor=0.5`時にStep A/B両方スキップ（852行目）
5. 呼び出し側で`octave_factor`を渡す（~1295行目）

---

## Issue B: 「はいよろこんで」のBPM誤検出（147→227.3） ← 新規

### Context

正しいBPMは147だが、227.3と検出される。BPM候補のF1スコア：
```
BPM 229: P=0.709 R=0.695 F1=0.702 ← F1では1位
BPM 147: P=0.878 R=0.555 F1=0.680 ← F1では2位（しかし正解）
```

147 BPMはPrecisionが非常に高い（0.878）がRecallが低い（0.555）。Recallが低い理由は曲に8分音符等の細分音符が多くビート間にオンセットがあるため（正常）。229 BPMはこれら細分音符も拾うためRecallが高いが、Precisionは低い（ビート位置が不正確）。

### 根本原因

BPM選択にF1スコア（P/Rを均等に重視）を使用しているため、細分音符の多い曲で高速BPMが有利になる。

### 修正内容

**F1（beta=1.0）→ F_beta（beta=0.8）に変更**（1174-1176行目）

Precisionを約55%、Recallを約45%の重みで評価。音楽のBPM検出ではPrecisionがより重要：
- 高P = 予測ビート位置にオンセットが存在 → 正しいBPM
- 低R = ビート間にもオンセット → 細分音符（正常）

```python
# Before (F1):
score = 2 * precision * recall / (precision + recall)

# After (F_beta, beta=0.8):
BEAT_F_BETA = 0.8
beta_sq = BEAT_F_BETA ** 2  # 0.64
score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
```

ログ出力も更新（「F1」→「Fβ」に変更）:
```python
# 1184行目付近
print(f"[DEBUG] BPM candidate: {c} (P={p:.3f} R={r:.3f} Fβ={s:.3f})")
```

### 検算（既存曲への影響なし）

| 曲 | BPM | F1 | F0.8 | 結果 |
|---|---|---|---|---|
| はいよろこんで 147 | 0.680 | **0.715** | ✅ 1位に |
| はいよろこんで 229 | 0.702 | 0.703 | 2位に下がる |
| いんちき和尚 186 | 0.705 | **0.714** | ✅ 変わらず1位 |
| いんちき和尚 229 | 0.695 | 0.699 | 変わらず2位 |

---

## 検証

1. 「はいよろこんで」で解析し、BPMが~147になることを確認
2. 「いんちき和尚」で解析し、BPMが~186→93（オクターブ補正）で位相が~67msになることを確認
3. 「tax」で解析し、従来と同じ結果になることを確認
4. 既存テスト実行: `python tests/verify_modes.py`, `verify_preview_content.py`
