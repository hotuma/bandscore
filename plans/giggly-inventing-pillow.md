# BPM検出精度の改善プラン（v6: 2段階F1スキャン）

## Context
- F1スキャン（1BPM刻み）でオクターブ解決に成功、BPM 184が選ばれた
- Stage 2で`beat_track(start_bpm=184)`を試すも184.57を返す
- **原因**: beat_trackはフレーム量子化（sr=22050, hop=512）により、BPM=60*22050/(512*N)の離散値しか返せない。N=14→184.57, N=15→172.27。180は構造的に返せない
- Stage 2のbeat_track精密化は無効。代わりにF1スキャン自体を2段階にする

## 修正方針: 粗いF1スキャン → 細かいF1スキャン

### Stage 1（既存）: 粗いスキャン
BPM 60-240を**1BPM刻み**でF1スキャン → 正しいオクターブ付近のBPM（例: 184）を特定。**変更なし**。

### Stage 2（変更）: 細かいF1スキャン
粗い結果 ±10BPM の範囲を**0.1BPM刻み**でF1再スキャン → 小数点レベルの精密値を取得。

beat_trackを使わないため、フレーム量子化の制約を受けない。F1スキャンのビートグリッドは`np.arange`で浮動小数点ベースなので、任意のBPM精度で評価可能。

## 対象ファイル
- [backend/main.py](backend/main.py) L845-859（Stage 2 beat_trackブロック）

## 修正コード

現在のL845-859（beat_trackによるStage 2）を以下に置き換え:

```python
            # Stage 2: 粗い結果の周辺を0.1BPM刻みで細かくF1スキャン
            coarse_bpm = best_bpm
            fine_best_bpm = coarse_bpm
            fine_best_score = best_score

            for c_10x in range(int((coarse_bpm - 10) * 10), int((coarse_bpm + 10) * 10) + 1):
                c = c_10x / 10.0
                beat_period = 60.0 * sr / (c * 512)
                grid = np.arange(0, total_frames, beat_period)
                if len(grid) == 0:
                    continue
                hits = 0
                for g in grid:
                    g_int = int(round(g))
                    for t in range(-tolerance, tolerance + 1):
                        if (g_int + t) in onset_set:
                            hits += 1
                            break
                precision = hits / len(grid)
                grid_set = set()
                for g in grid:
                    g_int = int(round(g))
                    for t in range(-tolerance, tolerance + 1):
                        grid_set.add(g_int + t)
                onset_hits = sum(1 for o in onset_frames_detected if int(o) in grid_set)
                recall = onset_hits / max(1, num_onsets)
                if precision + recall > 0:
                    score = 2 * precision * recall / (precision + recall)
                else:
                    score = 0.0
                if score > fine_best_score:
                    fine_best_score = score
                    fine_best_bpm = c

            bpm = fine_best_bpm
            beat_frames = []
            print(f"[DEBUG] BPM fine-tuned: {coarse_bpm:.0f} -> {bpm:.1f} (F1={fine_best_score:.3f})")
```

## 検証手順
1. `--reload`で自動再起動
2. tax.mp3をアップロード
3. ログ確認:
   - Stage 1上位候補にBPM 184が表示される
   - `BPM fine-tuned: 184 -> 1XX.X` — 180に近い値を期待
4. フロントエンドで再生し同期を確認
