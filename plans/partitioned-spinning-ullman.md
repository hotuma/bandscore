# BPMハーフテンポ補正 + 分析時間制限の修正プラン

## Context

バックエンド手動再起動後、`[ChunkMerge]` ログが表示され、チャンク処理のBPM統合は動作確認済み。

しかし以下の問題が残っている：

1. **BPMがハーフテンポ（92.3 → 実際は~180）**: librosa `beat_track` の既知問題（半分のBPMを検出）
2. **途中で再生速度が変わりドリフトする**: BPMが実際の半分のため、セグメント境界が実際のビート位置とズレる
3. **2:36の音源に対して2分でコード生成が終わる**: `MAX_ANALYSIS_SEC=120` がEARLY_ACCESSモードのハードリミット

## 修正内容

### 修正1: BPM一貫性の確保 + オクターブ補正

**問題**: 各チャンクが独立にBPM検出するため、チャンク間でBPMがバラバラ（92.3 / 184.6 / 107.7）。barの長さが不統一で速度が変わる。

**対策A**: `analyze_audio_file()`に`forced_bpm`パラメータを追加。指定時はBPM検出をスキップ。

```python
def analyze_audio_file(file_path, progress_callback=None, offset_sec=0.0,
                       duration_limit_sec=None, forced_bpm=None):
```

BPM検出部分:
```python
if forced_bpm is not None:
    bpm = forced_bpm
    beat_frames = []
    print(f"[DEBUG] Using forced BPM: {bpm:.1f}")
else:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames', start_bpm=120)
    bpm = float(tempo)
    # オクターブ補正: ハーフテンポ検出の場合は倍にする
    if bpm < 100:
        bpm *= 2
        print(f"[DEBUG] Detected BPM: {tempo:.1f} → octave corrected to {bpm:.1f}")
    else:
        print(f"[DEBUG] Detected BPM: {bpm:.1f}")
```

**対策B**: `run_analysis_bg()`で最初のチャンクのBPMを後続チャンクに強制適用。

```python
raw = analyze_audio_file(
    file_path, progress_callback=chunk_cb,
    offset_sec=offset, duration_limit_sec=dur,
    forced_bpm=bpm  # 最初のチャンク以降はbpmが設定済み
)
```

### 修正2: EARLY_ACCESSモードの分析時間上限を引き上げ

**対象**: `backend/main.py` — `run_analysis_bg()` 関数内、L968-976

現在、EARLY_ACCESSとFULLモードが同じ`MAX_ANALYSIS_SEC=120`を使用しているが、CLAUDE.mdの仕様ではFULLは600秒。EARLY_ACCESSも120秒では一般的な楽曲（3-5分）をカバーできない。

**現在のコード (L968-976)**:
```python
if mode == AnalyzeMode.PREVIEW:
    MAX_ANALYSIS_SEC = 60.0
    print("[INFO] Mode: PREVIEW -> Forced duration 60.0s")
else:
    MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "120"))
```

**修正後**:
```python
if mode == AnalyzeMode.PREVIEW:
    MAX_ANALYSIS_SEC = 60.0
    print("[INFO] Mode: PREVIEW -> Forced duration 60.0s")
elif mode == AnalyzeMode.EARLY_ACCESS:
    MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "300"))
    print(f"[INFO] Mode: EARLY_ACCESS -> Max duration {MAX_ANALYSIS_SEC}s")
else:  # FULL
    MAX_ANALYSIS_SEC = float(os.getenv("MAX_ANALYSIS_SEC", "600"))
    print(f"[INFO] Mode: FULL -> Max duration {MAX_ANALYSIS_SEC}s")
```

**効果**:
- PREVIEW: 60秒（変更なし）
- EARLY_ACCESS: 300秒（5分、ほとんどの楽曲をカバー）
- FULL: 600秒（10分、CLAUDE.md仕様通り）
- 環境変数でオーバーライド可能（クラウドのメモリ制約対応）

## 修正対象ファイル

- `backend/main.py`
  - `analyze_audio_file()` L782-784: BPMオクターブ補正
  - `run_analysis_bg()` L968-976: モード別時間制限

## 検証手順

1. バックエンドを再起動
2. `tax.mp3`（2:36の音源）をアップロード
3. 確認項目:
   - ログに `Detected BPM: 92.3 → octave corrected to 184.6` が表示される
   - フロントエンドのBPM表示が~184になっている
   - バーが音源全体（2:36=156秒）をカバーしている
   - コード再生が一定速度で小節頭に合っている
   - ドリフト（途中で速度が変わる）が解消されている
