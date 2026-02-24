# バー同期問題の根本原因と修正

## Context

再生中の小節ハイライトが音楽と合わない。ログでバー間隔 **1.115s** (BPM 184 なら **1.304s** のはず)。ユーザーが何度も再分析・サーバー再起動しても症状が継続。

## 真の根本原因: サーバーが古いコードで動作中

**git diff 解析の結果**: 修正コード（F1スコアBPM検出、位相検出、統一グリッド、`_build` タグなど）は**すべて未コミット**。サーバーは `HEAD` (古いコミット済みコード) で実行されている。

### 古コード (HEAD) vs 新コード (作業ツリー)

| 機能 | 古コード (実行中) | 新コード (未適用) | ユーザーログ |
|------|------------------|------------------|------------|
| **BPM検出** | `librosa.beat.beat_track` | F1スコア + AC | BPM 184 ✓ |
| **ビートタイム生成** | **フレーム境界ベース** | 連続時間ベース | - |
| **バー間隔** | **1.115s** (量子化誤差) | 1.304s | **1.115s** ← 一致 |
| **位相オフセット** | なし (0.000) | あり (0.265s) | **0.000** ← 一致 |
| **統一グリッド** | なし | あり | - |
| **`_build` タグ** | なし | `"unified-grid-v2"` | **表示されず** ← 一致 |

### なぜ古コードでバー間隔が 1.115s になるか

古コードのセグメント生成はフレーム境界に量子化:

```python
# 古コード (backend/main.py HEAD版)
target_segment_duration = (60/184) * 2 = 0.6522s  # 理論値

frames_per_segment = int(0.6522 * 22050 / 4096)
                   = int(3.51)
                   = 3 frames  # ← int() で切り捨て!

実際のセグメント間隔 = 3 * 4096 / 22050 = 0.557s
バー間隔 = 2 × 0.557s = 1.114s ≈ 1.115s  ✓
```

新コードは連続時間ベース → 正確に 1.304s。

## なぜ `uvicorn --reload` で反映されないか

Windowsの `--reload` はファイル監視が不安定:
1. ファイル保存を検知しないことがある
2. または検知してもモジュールを完全にリロードしない
3. 特に大きなファイルや頻繁な変更で問題が発生しやすい

**結果**: サーバープロセスは古いモジュールオブジェクトをキャッシュし、ディスク上の新コードを読み込まない。

## 修正手順

### ステップ1: サーバープロセスを完全停止

```bash
# バックエンド停止
cd c:/Users/USER/.gemini/antigravity/guitar-tab/backend

# Ctrl+C で停止
# もし反応しない場合、タスクマネージャーで python.exe/uvicorn プロセスを強制終了

# 停止確認
netstat -ano | findstr :8000
# → 何も表示されなければOK
```

### ステップ2: 変更をコミット (推奨)

```bash
cd c:/Users/USER/.gemini/antigravity/guitar-tab

# 変更内容を確認
git diff backend/main.py | head -100

# コミット (作業ツリーの変更をHEADに反映)
git add backend/main.py frontend/components/ResultDisplay.tsx
git commit -m "fix: BPM検出精度向上、位相検出、統一グリッド、診断タグを追加"

# または、コミットせずに続行してもよい (次のステップで新コードが読み込まれる)
```

### ステップ3: サーバーを新規起動

```bash
cd c:/Users/USER/.gemini/antigravity/guitar-tab/backend

# 完全に新しいプロセスで起動
uvicorn main:app --reload
```

**起動時ログで確認**:
- モジュールのインポート時刻が最新か
- エラーがないか

### ステップ4: 検証

1. フロントエンド起動: `cd frontend && npm run dev`
2. tax.mp3 を**新規アップロード・分析**
3. **バックエンドコンソール**で確認:
   ```
   [DEBUG] BPM candidate: 184 (P=... R=... F1=...)
   [DEBUG] AC refined=... keeping coarse 184
   [DEBUG] Beat phase offset: 265.0ms (precision=...)
   [ChunkMerge] Using detected BPM: 184.0, segment_duration: 0.652s
   [ChunkMerge] Unifying bar timing: 120 bars, seg=1.3043s, phase=0.2650s
   [ChunkMerge] Final bar duration: 1.3043s (expected 1.3043s)
   ```
4. **フロントエンドコンソール** (ブラウザ DevTools) で確認:
   ```javascript
   [Bars] Timing dump: {
     totalBars: 120,
     barDuration: '1.304',  // ← 1.115 ではなく 1.304!
     firstBarStart: '0.265', // ← 0.000 ではない!
     bpm: 184,
     _build: 'unified-grid-v2'  // ← 表示される!
   }
   ```
5. 再生して、小節ハイライトが音楽と60秒以上同期することを確認

## フォールバック: それでも反映されない場合

### オプションA: Pythonキャッシュをクリア

```bash
cd c:/Users/USER/.gemini/antigravity/guitar-tab/backend

# __pycache__ を削除
rm -rf __pycache__
rm -f *.pyc

# 再起動
uvicorn main:app --reload
```

### オプションB: --reload を使わない

```bash
# サーバー起動時に毎回Ctrl+Cで停止してから再起動
uvicorn main:app
# (変更後は手動でCtrl+C → 再起動)
```

### オプションC: 仮想環境を再作成

```bash
cd c:/Users/USER/.gemini/antigravity/guitar-tab/backend

deactivate
rm -rf .venv
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt

uvicorn main:app --reload
```

## 変更ファイル

### 主要変更: [backend/main.py](backend/main.py)

| 行範囲 | 変更内容 | 効果 |
|--------|---------|-----|
| L743 | `forced_bpm` パラメータ追加 | チャンク間でBPM統一 |
| L780-929 | F1スコアBPM検出 + AC + 位相検出 | BPM精度・位相精度向上 |
| L955-960 | 連続時間ベースのビートタイム生成 | フレーム量子化誤差を排除 |
| L1043 | `phase_offset_sec` をレスポンスに追加 | デバッグ可能に |
| L1148-1151 | `FIRST_CHUNK_SEC = 60.0` | 初回チャンクでBPM精度確保 |
| L1234-1259 | 統一グリッド (チャンク結合後) | チャンク境界の不連続を修正 |
| L1263-1265 | 診断ログ | バー間隔を確認可能に |
| L1277 | `_build: "unified-grid-v2"` | コードバージョン確認可能に |

### 副次変更: [frontend/components/ResultDisplay.tsx](frontend/components/ResultDisplay.tsx)

| 行 | 変更内容 |
|----|---------|
| L181-188 | `[Bars] Timing dump` に `_build` 追加 |
| L386-392, L486-492 | AudioContext ドリフト補正 |

## 成功の証拠

修正が成功すると:
- ✅ `barDuration: '1.304'` (1.115 ではない)
- ✅ `firstBarStart` が 0 ではない (位相オフセット適用)
- ✅ `_build: 'unified-grid-v2'` が表示される
- ✅ 小節ハイライトが音楽と60秒以上一致
- ✅ バックエンドログに `[ChunkMerge] Unifying bar timing:` が出力

## 失敗した場合

もし上記手順でも `barDuration: '1.115'` のままなら:
1. バックエンドのプロセスIDを確認: `ps aux | grep uvicorn` (Unix) / タスクマネージャー (Windows)
2. 本当に新しいプロセスか確認 (起動時刻)
3. `backend/main.py` の L1277 付近に `print("CODE VERSION: unified-grid-v2")` を追加してモジュールインポート時に出力
4. ブラウザのハードリロード (Ctrl+Shift+R) でフロントエンドキャッシュをクリア
