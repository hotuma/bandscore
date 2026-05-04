# Context

## 実装済みの修正

- **Fix 1**: `seg_dur = beat_dur * forced_beats_per_seg`（チャンク2以降の位相アライメント）
- **Fix 2**: adaptive beat tracking で BarPhaseRefine のフェーズを起点に使用
- **Fix 3**: BassTempoCorrection に比率バリデーション + 3/2 倍音チェック
- **Fix 4**: 全チャンクで `beat_duration`（192 BPM 固定）を統一
- **Fix 6**: Fix 5（adaptive BPM リファイン）をリバート → Fix 4 状態に復帰

**現状**: 最初は合うが 36 秒から徐々にずれる（累積 ~0.1s/29 bar）

---

## 残課題: BPM 精度不足

### BPM 検出パイプライン

```
Stage 1: F-beta scan 60-240 BPM (整数刻み) → 212 BPM
Stage 2: AC refine hop=128, ±5% around 212 → 183.2 で却下 → 212 維持
Stage 2.5: OctaveVerify → 212 維持
Stage 2.7: BassTempoCorrection → 128 × 1.5 = 192.0 (整数)
          ここで 192.0 に対する AC 精密化なし ← 問題
```

### 原因

BassTempoCorrection の bass ピーク検索 (`range(60, 201, 2)`) は **step=2 BPM** なので精度は ±1 BPM:
- 実際の bass ピーク = 128.0 BPM (ステップ=2 の格子点)
- 真の bass ピークが 128.33 BPM なら、step=2 では 128 として検出される
- `128.33 × 1.5 = 192.5` → `128.0 × 1.5 = 192.0` との差 = **0.5 BPM**

0.5 BPM の誤差が積み重なると:
- bar duration: 192 BPM → 1.25000s vs 192.5 BPM → 1.24675s
- 差: 0.00325s/bar
- 29 bar (36 秒) 後の累積ずれ: 29 × 0.00325 = **0.094s** → ユーザーが感じる「段々ずれる」の原因

---

# Root Cause

**`backend/main.py` 行 1680-1685**:

`best_bass_bpm` (128) が step=2 の粗いスキャンから取得されており、`best_bass_bpm * 1.5 = 192.0` も整数精度。BassTempoCorrection 後に AC 精密化が行われないため、**サブ BPM 精度の BPM が得られない**。

---

# Fix Plan

## Fix 7: BassTempoCorrection の 3/2 補正前に bass AC を parabolic interpolation で精密化

**ファイル**: `backend/main.py` 行 1680-1685（`if abs(candidate_3_2 - bpm) / bpm < 0.15:` ブロック内）

```python
# 変更前（現状）
candidate_3_2 = best_bass_bpm * 1.5
if abs(candidate_3_2 - bpm) / bpm < 0.15:
    print(f"[BassTempoCorrection] {bpm:.1f} → {candidate_3_2:.1f} BPM "
          f"(bass={best_bass_bpm:.0f} × 3/2, ratio={ratio:.3f} invalid)")
    bpm = candidate_3_2
    octave_factor = 1.0
```

```python
# 変更後（Fix 7）
candidate_3_2 = best_bass_bpm * 1.5
if abs(candidate_3_2 - bpm) / bpm < 0.15:
    # best_bass_bpm は step=2 BPM スキャン (±1 BPM 精度)
    # hop=128 の bass AC + parabolic interpolation で 0.1 BPM 精度に向上させてから 3/2 補正
    try:
        hop_fine = 128
        _y_bass_fine = lowpass_filter(y, sr, cutoff_hz=200)
        _y_bass_fine = highpass_filter(_y_bass_fine, sr, cutoff_hz=20)
        _bass_env_fine = librosa.onset.onset_strength(
            y=_y_bass_fine, sr=sr, hop_length=hop_fine)
        del _y_bass_fine
        _ac_fine = librosa.autocorrelate(_bass_env_fine)
        del _bass_env_fine
        if _ac_fine[0] > 0:
            _ac_fine = _ac_fine / _ac_fine[0]
        # best_bass_bpm の lag を中心に ±3% 範囲で精密検索
        _fine_lag_center = 60.0 * sr / (best_bass_bpm * hop_fine)
        _fine_radius = max(3, int(_fine_lag_center * 0.03))
        _fine_lo = max(1, int(_fine_lag_center) - _fine_radius)
        _fine_hi = min(len(_ac_fine) - 2, int(_fine_lag_center) + _fine_radius)
        refined_bass_bpm = best_bass_bpm  # fallback
        if _fine_hi > _fine_lo:
            _pk = _fine_lo + int(np.argmax(_ac_fine[_fine_lo:_fine_hi + 1]))
            if 0 < _pk < len(_ac_fine) - 1:
                _a = float(_ac_fine[_pk - 1])
                _b = float(_ac_fine[_pk])
                _g = float(_ac_fine[_pk + 1])
                _denom = _a - 2.0 * _b + _g
                _delta = 0.5 * (_a - _g) / _denom if abs(_denom) > 1e-10 else 0.0
                _rl = _pk + _delta
                _rb = 60.0 * sr / (_rl * hop_fine) if _rl > 0 else best_bass_bpm
                if abs(_rb - best_bass_bpm) < 2.0:  # ±2 BPM 以内なら採用
                    refined_bass_bpm = _rb
        del _ac_fine
        refined_candidate = refined_bass_bpm * 1.5
        print(f"[BassTempoCorrection] {bpm:.1f} → {refined_candidate:.2f} BPM "
              f"(bass={best_bass_bpm:.0f} → {refined_bass_bpm:.2f} × 3/2, "
              f"ratio={ratio:.3f} invalid)")
        bpm = refined_candidate
    except Exception as _e:
        print(f"[BassTempoCorrection] {bpm:.1f} → {candidate_3_2:.1f} BPM "
              f"(bass={best_bass_bpm:.0f} × 3/2, ratio={ratio:.3f} invalid, "
              f"refinement failed: {_e})")
        bpm = candidate_3_2
    octave_factor = 1.0
```

**原理**:
- 現在: step=2 スキャン → bass peak at 128 BPM (ステップ格子点) → 128 × 1.5 = 192.0
- 修正後: hop=128 で bass AC → lag 80.76 frames → parabolic interp → 128.33 BPM → 128.33 × 1.5 = **192.5**
- この 0.5 BPM の精度改善により 36 秒後の cumulative drift が ~0.094s から ~0 に改善

**追加コスト**: hop=128 の bass onset_strength は hop=512 の 4 倍のサンプル数だが、初回チャンク限定かつ低域信号のみ。実用的に許容範囲。

---

# Critical Files

- `backend/main.py`: 行 1680-1685（Fix 7 挿入箇所）

---

# Verification

1. バックエンドを再起動: `uvicorn main:app --reload`
2. ZIEQDjrAdwE を再解析
3. ログで確認:
   - `[BassTempoCorrection] 212.0 → 192.XX BPM (bass=128 → 128.YY × 3/2, ...)` が出ること
   - BPM が 192.0 から 192.X（小数点あり）に変わること
   - `[ChunkMerge] Using detected BPM: 192.X, ..., segment_duration: X.XXXs` に変わること
4. フロントエンドで 36 秒以降のずれが改善されたことを目視確認
5. もし改善しない場合: 実際の BPM が 192.0 に近く（refinement が 128.0 ≈ 128.0 を返す）、drift の原因が他にある可能性を検討
