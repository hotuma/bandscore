# タイミング同期検証結果の分析と次ステップ計画

## Context

Phase 1とPhase 2の修正を適用後、ユーザーがtax.mp3を再分析して得られたフロントエンドログの評価を行う。

### 修正内容の要約
- **Phase 1**: F1スコアBPM検出、位相検出、統一グリッド、診断タグ
- **Phase 2**: ACリファイン結果(183.5 BPM)の常時採用による累積ドリフト対策

## ログ分析結果

### ✅ 成功した改善点

1. **BPM検出の精度向上**
   - API応答: `bpm: 183.5` ✓
   - Phase 2修正(ACリファイン採用)が正しく適用されている

2. **バー間隔の修正**
   - 修正前: 1.115秒(フレーム量子化の誤り)
   - 修正後: 1.308秒 ✓
   - 理論値: 60/(183.5/4) = 1.3079秒 → ほぼ一致

3. **位相オフセットの適用**
   - `firstBarStart: 0.221` ✓
   - ビート位相検出が機能している

4. **短期同期精度**
   - Bar 1-40 (0-52秒): lag < 0.11秒 ✓
   - ほぼ完璧な同期を維持

### ⚠️ 検出された異常

**Bar 50での大きなラグ**:
```
Bar 49: currentTime=64.323s, barStart=64.308s → lag +0.015s ✓
Bar 50: currentTime=66.403s, barStart=65.616s → lag +0.787s ⚠️
```

**問題の可能性**:
1. **ブラウザの表示更新タイミング**: requestAnimationFrameの更新間隔(~16ms)により、たまたまログが飛んだ
2. **データの不整合**: APIレスポンスのバーデータに問題がある
3. **累積誤差**: まだ完全には解消されていない

## 検証が必要な事項

### 1. APIレスポンスデータの整合性確認
バー50付近のデータを確認:
```javascript
// フロントエンドで console.log(result.bars.slice(48, 52))
```

期待される出力:
```javascript
[
  { start_sec: 63.0, end_sec: 64.308, chord: "...", tab: "..." },  // Bar 48
  { start_sec: 64.308, end_sec: 65.616, chord: "...", tab: "..." }, // Bar 49
  { start_sec: 65.616, end_sec: 66.924, chord: "...", tab: "..." }, // Bar 50
  { start_sec: 66.924, end_sec: 68.232, chord: "...", tab: "..." }  // Bar 51
]
```

### 2. 長時間同期の検証
- ログは66秒(50バー)で途切れている
- 全121バー(158秒)の検証が必要
- 2分以上の連続再生でずれが拡大するか確認

### 3. バックエンドログの確認
以下のログメッセージが出力されているか確認:
```
[DEBUG] AC refined=183.5 (delta<1), using refined 183.5  ← 期待されるメッセージ
[ChunkMerge] Unifying bar timing: 121 bars, seg=1.3079s
```

## 次のステップ

### Option A: さらなるデータ収集(推奨)

1. **完全なAPIレスポンスダンプ**
   - [ResultDisplay.tsx:181](frontend/components/ResultDisplay.tsx#L181)付近に追加:
     ```typescript
     console.log('[Bars] Full data:', result.bars.slice(48, 52));
     ```

2. **全バー範囲の同期検証**
   - 2分30秒(158秒)まで再生
   - 30秒ごとにハイライトタイミングを記録

3. **バックエンドログの有効化**
   - サーバーを再起動してログをファイルに保存:
     ```bash
     uvicorn main:app --reload 2>&1 | tee backend/server.log
     ```

### Option B: ユーザーヒアリング(簡易)

ユーザーに直接確認:
1. 「全曲(2分30秒)を通して聞いたときのずれ具合は?」
2. 「最初は合うが、途中からずれる感覚はまだありますか?」
3. 「手動オフセットスライダーで調整が必要ですか?」

### Option C: さらなる精度向上(必要な場合のみ)

もしBar 50の異常が実際の同期ずれなら:

1. **テンポ変動の検出**
   - 曲全体でBPMが一定でない可能性
   - 区間ごとのBPM推定を追加

2. **ブラウザオーディオ遅延補正**
   - AudioContext latency (`audioContext.outputLatency`) を考慮
   - デフォルトで+20ms程度のバッファ遅延がある

3. **デコード差分の計測**
   - librosa読み込み vs. ブラウザデコードの時間差
   - 自動キャリブレーションの実装

## 推奨アクション

**即座に実行すべきこと**:
1. ユーザーにBar 50以降(66秒~158秒)の体感的な同期状況をヒアリング
2. バックエンドログの確認(ACリファイン採用メッセージの有無)

**ユーザー報告次第で実行**:
- 「まだずれる」→ Option A(データ収集)とOption C(精度向上)を検討
- 「ほぼ合っている」→ 完了。手動オフセット機能で微調整可能であることを案内

## 暫定評価(更新)

**Phase 1+2の修正効果**:
- ✅ BPM検出: 184 → 183.5 (AC精密化成功)
- ✅ バー間隔: 1.115s → 1.308s (理論値と一致)
- ✅ 短期同期(50秒): ほぼ完璧(lag < 0.11s)
- ⚠️ 長期同期(2分+): **周期的なずれパターンを確認**

### ユーザーフィードバック

**同期状況**: 「途中からずれてくるが、また合うようになり、またズレの繰り返し」

**重要な発見**: これは累積ドリフトではなく、**周期的な同期ずれ**を示唆している。

## 周期的ずれの原因候補

### 原因A: チャンク境界での不連続(最有力)

**仮説**:
- `CHUNK_SEC=30s` でチャンクを分割
- チャンク境界(0s, 30s, 60s, 90s, 120s)で位相ずれが発生
- 統一グリッド処理に問題がある可能性

**検証方法**:
- バックエンドログから `[ChunkMerge]` メッセージを確認
- ずれが発生するタイミングが30秒の倍数付近か確認

**予想されるログ**:
```
[ChunkMerge] Unifying bar timing: 121 bars, seg=1.3079s
[ChunkMerge] Adjusting chunk boundaries...
```

### 原因B: バーデータの長さの不均一性

**仮説**:
- 一部のバーの `end_sec - start_sec` が1.308sでない
- 統一グリッド処理が一部のバーにのみ適用されている

**検証方法**:
- APIレスポンスの全121バーの長さを計算:
  ```javascript
  const durations = result.bars.map((b, i) =>
    ({ bar: i, duration: (b.end_sec - b.start_sec).toFixed(3) })
  );
  console.table(durations);
  ```

**期待される結果**:
- 全バーが1.308秒(±0.001s)で均一
- もし不均一なら、そのバーのインデックスと位置を特定

### 原因C: ブラウザオーディオ再生の問題

**仮説**:
- HTMLAudioElementの再生速度が微妙に不安定
- AudioContextの時刻とオーディオ再生の時刻がずれる

**検証方法**:
- `audioElement.playbackRate` を確認(1.0であるべき)
- `AudioContext.state` を確認(runningであるべき)

### 原因D: 曲の実際のテンポ変動

**仮説**:
- tax.mp3が人間演奏で、テンポが揺らいでいる
- 固定BPM 183.5では捉えきれない

**検証方法**:
- 別の楽曲(機械的なBPMの曲)でテスト
- tax.mp3の各セクションで個別にBPM推定

## 🔍 根本原因の特定(CONFIRMED)

### バックエンドログ分析

**重要な発見: 各チャンクの位相オフセットが不一致**

```
Chunk 1 (0-60s):   Beat phase offset: 220.6ms (precision=0.6175)
Chunk 2 (60-90s):  Beat phase offset: 264.7ms (precision=0.9231) ← +44.1ms!
Chunk 3 (90-120s): Beat phase offset: 313.5ms (precision=0.8462) ← +92.9ms!
Chunk 4 (120-150s): Beat phase offset: 225.2ms (precision=0.8804) ← +4.6ms (ほぼ一致)
Chunk 5 (150-156s): Beat phase offset: 32.5ms (precision=0.2000) ← -188.1ms!
```

**統一グリッド処理**:
```
[ChunkMerge] Unifying bar timing: 121 bars, seg=1.3079s, phase=0.2206s
```

### 問題の構造

1. **統一グリッドの動作**:
   - バーの `start_sec`/`end_sec` をChunk 1の位相(220.6ms)で統一 ✓
   - これにより **バー間隔は均一** (1.3079s)

2. **コード検出の動作**:
   - 各チャンクは **独立した位相** でクロマ特徴を集約
   - Chunk 2は264.7ms位相でコード変化を検出
   - しかし、グリッドは220.6ms位相 → **44msのタイミングずれ**

3. **ユーザー体験との対応**:
   - 0-60s: 位相220.6ms → グリッド220.6ms → ✓ 「合っている」
   - 60-90s: 位相264.7ms → グリッド220.6ms → ✗ 「ずれてきた」(+44ms)
   - 90-120s: 位相313.5ms → グリッド220.6ms → ✗ 「さらにずれた」(+93ms)
   - 120-150s: 位相225.2ms → グリッド220.6ms → ✓ 「また合う」(+5ms)
   - 150-156s: 位相32.5ms → グリッド220.6ms → ✗ 「大きくずれる」(-188ms)

### なぜこの問題が起きるのか

**現在の実装**:
```python
# analyze_audio_file() の各チャンク処理
for chunk in chunks:
    if chunk_idx == 0:
        bpm, phase = detect_bpm_and_phase(...)  # BPMと位相を検出
    else:
        bpm = forced_bpm  # BPMは強制 ✓
        phase = detect_phase(...)  # 位相は再検出 ← 問題!

    # この位相でクロマ集約とコード検出を実行
    bars = detect_chords_with_segments(..., phase)
```

**統一グリッド処理** ([backend/main.py:1253-1259](backend/main.py#L1253-L1259)):
```python
# チャンク結合後: タイミングを統一
first_start = all_bars[0]["start_sec"]  # Chunk 1の位相(220.6ms)
for i, bar in enumerate(all_bars):
    bar["start_sec"] = first_start + i * seg_duration  # タイミング統一
    bar["end_sec"] = first_start + (i + 1) * seg_duration
```

**問題**: タイミングは統一されるが、**コード検出時の位相は各チャンクで異なる**ため、コード変化のタイミングがグリッドとずれる。

## 💡 解決策

### Phase 3: 位相統一(Unified Phase)

**方針**: 最初のチャンクで検出した位相を、全チャンクで強制使用する。

### 実装案

**修正箇所**: [backend/main.py:746](backend/main.py#L746)付近の `analyze_audio_file()` 関数

```python
# 現在の実装
forced_bpm = None

# 修正後
forced_bpm = None
forced_phase = None  # ← NEW

for chunk_idx, (offset_sec, dur_sec) in enumerate(chunks):
    if chunk_idx == 0:
        # 最初のチャンクでBPMと位相を検出
        bpm, phase = detect_bpm_and_phase(y, sr)
        forced_bpm = bpm
        forced_phase = phase  # ← NEW: 位相も保存
    else:
        # 後続チャンクではBPMと位相を強制
        bpm = forced_bpm
        phase = forced_phase  # ← NEW: 位相を強制(再検出しない)

    # 強制位相でコード検出
    result = detect_chords_with_segments(y, sr, bpm, phase)
```

### 修正が必要な関数

1. **`detect_chords_with_segments()`** ([backend/main.py:400](backend/main.py#L400)付近):
   - 現在: BPMのみを引数で受け取る
   - 修正後: `forced_phase` パラメータを追加
   - 位相が指定されている場合は、再検出をスキップ

2. **`analyze_audio_file()`** チャンクループ:
   - `forced_phase` 変数を追加
   - 最初のチャンクで位相を保存
   - 後続チャンクに位相を渡す

### 期待される効果

**修正前**:
```
Chunk 1: コード検出(位相220ms) → グリッド(220ms) → ✓ 合う
Chunk 2: コード検出(位相265ms) → グリッド(220ms) → ✗ 45msずれ
Chunk 3: コード検出(位相314ms) → グリッド(220ms) → ✗ 94msずれ
```

**修正後**:
```
Chunk 1: コード検出(位相220ms) → グリッド(220ms) → ✓ 合う
Chunk 2: コード検出(位相220ms) → グリッド(220ms) → ✓ 合う
Chunk 3: コード検出(位相220ms) → グリッド(220ms) → ✓ 合う
```

### 副次効果

- チャンク境界での不自然なコード変化を抑制
- 全体の音楽的一貫性が向上
- 位相検出の精度低下(Chunk 5: precision=0.2)の影響を排除

## 実装計画

### Critical Files
- [backend/main.py](backend/main.py) (修正対象)

### Implementation Steps

1. **`detect_chords_with_segments()` 関数の拡張**
   - `forced_phase` パラメータを追加(Optional[float], default=None)
   - 指定されている場合は、ビート位相検出をスキップして使用
   - デバッグログに「Using forced phase」メッセージを追加

2. **`analyze_audio_file()` のチャンクループを修正**
   - `forced_phase = None` 変数を追加
   - 最初のチャンク(chunk_idx == 0)で位相を保存
   - 後続チャンクで `forced_phase` を渡す
   - デバッグログに「Forcing phase from chunk 0」メッセージを追加

3. **診断ログの強化**
   - 各チャンクで「detected phase」と「used phase」を区別してログ出力
   - 統一グリッドログに位相情報を追加

### Verification

**成功判定基準**:
1. バックエンドログで全チャンクが同じ位相(220.6ms)を使用していることを確認
2. フロントエンドで2分30秒再生し、周期的ずれが消失することを確認
3. ハイライトlagが全バーで < 0.15秒を維持することを確認

**テスト手順**:
1. コード修正後、サーバー再起動
2. tax.mp3を再分析
3. バックエンドログを確認:
   ```
   [DEBUG] Beat phase offset: 220.6ms (chunk 0, detected)
   [DEBUG] Beat phase offset: 220.6ms (chunk 1, forced)
   [DEBUG] Beat phase offset: 220.6ms (chunk 2, forced)
   [ChunkMerge] Unifying bar timing: 121 bars, seg=1.3079s, phase=0.2206s
   ```
4. フロントエンドで全曲再生し、同期を確認

**総合判定(FINAL)**:
- ✅ 根本原因特定: チャンクごとの位相不一致
- ✅ 解決策明確: 最初のチャンクの位相を全チャンクで強制使用
- ✅ 実装箇所特定: `analyze_audio_file()` と `detect_chords_with_segments()`
- 📋 Phase 3修正の実装準備完了
