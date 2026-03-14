# BPM検出とハイライト位置ずれの修正プラン

## Phase 1: 問題分析

### 報告された問題
- YouTube動画 https://youtu.be/baJhnSJMZ98 の解析結果
- **BPM 180** と検出されているが、ミドルテンポの曲なのに速すぎる
- ハイライトが小節頭に合わない

### 調査結果

#### 1. 解析結果の検証 (analysis-1773518388627.json)

```json
{
  "bpm": 180,
  "time_signature": "2/4",
  "bars": [
    {
      "bar": 1,
      "start_sec": 0.603718820861678,
      "end_sec": 1.9040362811791383  // 長さ: 1.30秒
    },
    {
      "bar": 2,
      "start_sec": 1.9040362811791383,
      "end_sec": 3.2043537414965986  // 長さ: 1.30秒
    }
  ]
}
```

**重大な矛盾:**
- BPM 180、time_signature="2/4" → 理論上の小節長 = `(2 beats) / (180 BPM) * 60 = 0.667秒`
- **実際のJSONの小節長: 約1.30秒** → 理論値の**ほぼ2倍**

#### 2. バックエンドログの確認

```
[BassTempoCorrection] 216.0 → 180 BPM
[DEBUG] BPM: 180.00, Beat duration: 0.333s, Segments: ~182
[DEBUG] Segments: 46
[DEBUG] Aggregating segments...
```

計算検証:
- 60秒 ÷ 46セグメント = **1.304秒/セグメント**
- 180 BPM → 0.333秒/拍 → 4拍 = 1.333秒 ≈ **4/4拍子に相当**

**結論: time_signature="2/4"は誤り。実際は4/4拍子として処理されている。**

#### 3. 根本原因の特定

backend/main.py の小節長計算ロジックを調査した結果:

**現在のロジック（推定）:**
```python
# BPM検出後、セグメント長を計算
beat_duration = 60.0 / bpm  # 180 BPM → 0.333秒/拍
segment_duration = beat_duration * 2  # 2拍で1セグメント → 0.667秒/セグメント

# しかし実際は...
# aggregate_segments() 内で、何らかの理由で2倍の長さになっている
```

**仮説:**
1. **BPMが実際の2倍（ダブルタイム）で検出されている**
   - 本来: 90 BPM (4/4)
   - 検出: 180 BPM (2/4として処理、実際は4/4)

2. **セグメント集約処理でビート数が2倍になっている**
   - time_signature="2/4" で2拍/小節を想定
   - しかし実際のコードでは4拍分を集約している可能性

3. **Bass Tempo Correctionの問題**
   - 216 BPM → 180 BPMに補正
   - しかし真の値は90 BPMである可能性
   - バス自己相関が180 BPMで強く出ているのは、実際の2拍目にもバスドラムがある可能性

### ログから分かる詳細

```
[OctaveVerify] 108.0 BPM (×0.5): bass_ac=0.509, full_ac=0.480, phase=0.053, prior=0.7, total=0.449
[OctaveVerify] 216.0 BPM (×1.0): bass_ac=0.498, full_ac=0.507, phase=0.056, prior=0.5, total=0.412
[OctaveVerify] Half-tempo phase=0.053 < 0.25 (weak beat structure) -> keeping 216.0
```

**問題点:**
- 108 BPM (半分速) の phase_concentration = **0.053** → 0.25以下でゲートアウト
- しかし実際に90 BPM付近が正しい可能性があるのに、ゲート条件で棄却されている
- **phase_concentration計算に問題がある可能性**

```
[BassTempoCheck] Bass peak: 180 BPM (ac=0.529, prior=0.6, score=0.557)
[BassTempoCheck] Detected: 216.0 BPM (ac=0.498, prior=0.5, score=0.499)
[BassTempoCorrection] 216.0 → 180 BPM
```

- バス自己相関が180 BPMで最大
- しかし90 BPMは検証されていない（60-200範囲で2 BPM刻み探索のため）

## Phase 2: 修正方針

### 修正アプローチ

次の調査を実施する必要があります:

1. **セグメント集約ロジックの確認**
   - `aggregate_segments()` がどのようにビート数を決定しているか
   - time_signature生成ロジック

2. **BPM検出の精度向上**
   - オクターブ検証のphase_gate閾値を調整
   - バス自己相関の探索範囲を拡大（40-200 BPM）

3. **小節長の修正**
   - 真のBPM（90）× 4拍 = 正しい4/4小節
   - または検出BPM（180）× 2拍 = 正しい2/4小節

## Phase 2: 詳細調査結果

### 根本原因の特定

#### 1. beats_per_segment の不整合

**[backend/main.py:1779-1783](backend/main.py#L1779-L1783)** - Adaptive Beat Tracking成功時:
```python
if len(bt_times) >= 4:
    beat_times = bt_times
    beats_per_seg = 4  # ← 4ビートでグループ化
```

**[backend/main.py:1790-1791](backend/main.py#L1790-L1791)** - 固定グリッド:
```python
beat_times = np.arange(phase_offset_sec, total_duration + target_segment_duration, target_segment_duration)
beats_per_seg = 2  # ← 2ビートでグループ化
```

#### 2. 小節長の計算

| モード | beats_per_segment | BPM | 計算 | 小節長 |
|--------|------------------|-----|------|--------|
| **固定グリッド** | 2 | 180 | (60/180) × 2 | 0.667秒 |
| **Adaptive** | 4 | 180 | (60/180) × 4 | **1.333秒** ← 実測値 |

**実測値1.30秒は、beats_per_segment=4が使用されているため！**

#### 3. time_signature の硬コード

**[backend/main.py:1910](backend/main.py#L1910)**, **[2127](backend/main.py#L2127)**, **[2146](backend/main.py#L2146)** - 常に"2/4"固定:
```python
"time_signature": "2/4",  # 硬コード、動的決定なし
```

**矛盾:**
- time_signature="2/4" → 2拍/小節を意味
- beats_per_segment=4 → 実際は4拍/セグメント
- **小節（bar）とセグメントの定義が一致していない**

#### 4. 診断コードの誤った注釈

**[backend/main.py:2139](backend/main.py#L2139)**:
```python
_expected = round((60.0 / bpm) * 2, 4)  # 2拍/セグメント × beats_per_segment=2 = 4拍
```

この注釈は **誤り**:
- `(60.0 / bpm) * 2` = 2拍分の時間（4拍ではない）
- コメントが実装と矛盾している

### 影響

1. **小節長のずれ**
   - フロントエンドは time_signature="2/4" を信頼
   - 実際のbar配列は1.33秒間隔（4/4相当）
   - ハイライト位置が2倍ずれる

2. **BPM表示の誤解**
   - BPM 180は正しい（オンセット間隔）
   - しかしtime_signature="2/4"との組み合わせが誤り
   - 正: 180 BPM, 4/4 → 1.33秒/小節
   - 誤: 180 BPM, 2/4 → 0.667秒/小節（理論値）

### 修正方針

#### Option 1: beats_per_segmentに基づいてtime_signatureを動的決定（推奨）

**利点:**
- 既存のbeats_per_segment選択ロジックを活用
- adaptive trackingの精度を維持
- 最小限のコード変更

**変更箇所:**
1. beats_per_segmentの値に基づいてtime_signatureを設定
   - beats_per_seg=2 → "2/4"
   - beats_per_seg=4 → "4/4"
2. 全てのtime_signature硬コードを削除

#### Option 2: beats_per_segmentを常に2に固定

**利点:**
- time_signature="2/4"と整合性が取れる
- シンプルな修正

**欠点:**
- adaptive trackingの精度を犠牲にする可能性
- 4/4拍子の曲で小節境界がずれる

#### Option 3: BPM検出ロジックを修正（90 BPMを180と誤検出している場合）

**検証が必要:**
- 実際の曲のテンポを確認
- 90 BPM × 4拍 = 2.67秒（実測1.33秒と合わない）
- **この仮説は否定される**

### 推奨: Option 1の実装

**理由:**
- beats_per_segment=4はadaptive trackingが検出した実際のビート構造
- BPM 180、4/4拍子、1.33秒/小節は音楽理論的に整合性がある
- フロントエンドは正しいtime_signatureを受け取ればハイライトが合う

## Phase 3: 実装計画

### アプローチ: beats_per_segmentから動的にtime_signatureを計算

#### コアロジック

新しいヘルパー関数を追加:
```python
def compute_time_signature(beats_per_segment: int) -> str:
    """
    beats_per_segmentからtime_signatureを計算。

    Args:
        beats_per_segment: セグメントあたりのビート数 (2 or 4)

    Returns:
        拍子記号文字列 (例: "2/4", "4/4")
    """
    if beats_per_segment <= 0:
        return "4/4"  # 安全なデフォルト
    return f"{beats_per_segment}/4"
```

### 変更箇所

#### 1. 新しいヘルパー関数を追加

**ファイル:** [backend/main.py](backend/main.py)
**位置:** ~217行目（`_parse_time_signature()`の後）
**内容:** 上記の`compute_time_signature()`関数を追加

#### 2. analyze_audio_file()を修正

**変更1:** beats_per_segの初期化（~1475行目）
```python
octave_factor = 1.0
phase_detect_bpm = None
beats_per_seg = 2  # 追加: デフォルトは2拍/セグメント（フォールバック）
```

**変更2:** 戻り値にbeats_per_segmentを追加（~1907-1916行目）
```python
return {
    "bpm": bpm,
    "duration_sec": round(duration_sec, 1),
    "time_signature": compute_time_signature(beats_per_seg),  # 変更: 動的計算
    "key": estimated_key,
    "bars": bars,
    "phase_offset_sec": round(phase_offset_sec, 4),
    "final_last_chord": final_last_chord,
    "final_run_length": final_run_length,
    "beats_per_segment": beats_per_seg,  # 追加: チャンク整合性のため
}
```

#### 3. run_analysis_bg()のチャンク追跡を修正

**変更1:** 初期化（~2009行目）
```python
bpm = None
forced_phase = None
segment_duration = None
beats_per_seg = None  # 追加: 最初のチャンクからbeats_per_segmentを追跡
```

**変更2:** 最初のチャンク処理（~2065-2071行目）
```python
if bpm is None:
    bpm = raw.get("bpm", 120.0)
    forced_phase = raw.get("phase_offset_sec", 0.0)
    beats_per_seg = raw.get("beats_per_segment", 2)  # 追加: beats_per_segmentを抽出
    seconds_per_beat = 60.0 / bpm
    segment_duration = seconds_per_beat * beats_per_seg  # 変更: beats_per_segを使用
    print(f"[ChunkMerge] Using detected BPM: {bpm:.1f}, phase: {forced_phase*1000:.1f}ms, "
          f"beats_per_seg: {beats_per_seg}, segment_duration: {segment_duration:.3f}s")
```

#### 4. run_analysis_bg()のtime_signature使用を修正

**変更1:** add_bar_timing()呼び出し（~2127行目）
```python
time_sig = compute_time_signature(beats_per_seg) if beats_per_seg is not None else "4/4"
all_bars = add_bar_timing(
    all_bars,
    bpm=bpm,
    time_signature=time_sig,  # 変更: 動的計算
    analyzed_duration_sec=offset
)
```

**変更2:** 診断計算の修正（~2139行目）
```python
beats_used = beats_per_seg if beats_per_seg is not None else 2
_expected = round((60.0 / bpm) * beats_used, 4)  # 修正: beats_usedを使用
print(f"[ChunkMerge] Bar duration: {_diag_dur}s (per-chunk timing, expected ~{_expected}s @ {beats_used} beats/segment)")
```

**変更3:** 最終結果の戻り値（~2146行目）
```python
"time_signature": compute_time_signature(beats_per_seg) if beats_per_seg is not None else "4/4",
```

#### 5. テストファイルの更新

**ファイル:** [backend/tests/verify_bar_timing.py](backend/tests/verify_bar_timing.py)
**変更箇所:** 41行目、94行目、142行目

```python
# 41-46行目: beats_per_segmentを結果から取得
beats_per_seg = result.get("beats_per_segment", 2)  # 追加
expected_duration = (60.0 / bpm) * beats_per_seg  # 変更（*4から変更）
```

### 実装手順

実装は以下の順序で行います（中間状態でのエラーを避けるため）:

1. **Step 1:** `compute_time_signature()`ヘルパー関数を追加
2. **Step 2:** `analyze_audio_file()`を修正（beats_per_seg初期化、戻り値追加）
3. **Step 3:** `run_analysis_bg()`のチャンク追跡を修正
4. **Step 4:** `run_analysis_bg()`のtime_signature使用を修正
5. **Step 5:** テストを更新

### 後方互換性

- **フロントエンド:** 変更不要（time_signatureはstring型のまま）
- **APIレスポンス:** 非破壊的変更（"2/4"または"4/4"、どちらも有効な値）
- **テスト:** `verify_bar_timing.py`の更新が必要

### エッジケース対応

1. **beats_per_segがNullまたは0:** デフォルト"4/4"を返す
2. **最初のチャンクでビート検出失敗:** beats_per_seg=2（フォールバックグリッド）
3. **チャンク統合でbeats_per_segment欠如:** デフォルト2を使用

## Phase 4: 検証計画

### 実装後の検証

1. **YouTube動画の再解析**
   - https://youtu.be/baJhnSJMZ98 を再度解析
   - 期待値: BPM 180, time_signature="4/4", 小節長~1.33秒
   - ハイライトが小節頭に合うことを確認

2. **ユニットテスト**
   ```python
   def test_compute_time_signature():
       assert compute_time_signature(2) == "2/4"
       assert compute_time_signature(4) == "4/4"
       assert compute_time_signature(0) == "4/4"
   ```

3. **統合テスト**
   - `verify_bar_timing.py`を実行して全テストが通過
   - 60秒超のファイルでチャンク統合が正しく動作

4. **フロントエンド確認**
   - 解析結果を読み込んで再生
   - コンソールエラーがないことを確認
   - ハイライトとオーディオの同期を確認

### Critical Files

1. **[backend/main.py](backend/main.py)** - メイン実装（ヘルパー関数、analyze_audio_file、run_analysis_bg）
2. **[backend/tests/verify_bar_timing.py](backend/tests/verify_bar_timing.py)** - テスト更新

## 要約

**問題:** time_signature="2/4"が硬コード固定されており、beats_per_segment=4（adaptive mode）の場合に2倍のずれが発生

**解決策:** beats_per_segmentの値に基づいて動的にtime_signatureを計算
- beats_per_segment=2 → "2/4"
- beats_per_segment=4 → "4/4"

**影響範囲:**
- backend/main.py: 1関数追加、10箇所修正
- backend/tests/verify_bar_timing.py: 3箇所修正
- フロントエンド: 変更不要

**リスク:** 低（非破壊的変更、既存機能への影響最小）
