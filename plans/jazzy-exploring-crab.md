# madmom インストール失敗のトラブルシュートと代替案

## Context

madmom パッケージのインストール後に、`ModuleNotFoundError: No module named 'madmom'` エラーが発生し、解析が 0% で進行していません。

**エラーの原因**:
- `pip install -r requirements.txt` で madmom==0.16.1 のインストールが失敗した可能性
- Windows 環境でのビルドエラー（C++ コンパイラが必要）
- モジュールが正しくインストールされていない

## 解決策

### 緊急策: 代替アプローチの実装

**現実的な制約**:
- madmom は Windows でのビルドが複雑（C++ と Fortran の依存）
- 音源分離（Demucs/Spleeter）は更に重い
- 迅即的な解決が必要

**代替案**: 既存のヒューリスティック手法の改善

#### Fix A: librosa.beat.beat_track() のパラメータ調整

**ファイル**: `backend/main.py`
**行**: 595 (librosa.beat.beat_track 呼び出し)

**変更内容**:
```python
# 変更前
_, bt_frames = librosa.beat.beat_track(
    onset_envelope=onset_env, sr=sr, hop_length=bt_hop,
    bpm=bpm, trim=False
)

# 変更後
_, bt_frames = librosa.beat.beat_track(
    onset_envelope=onset_env, sr=sr, hop_length=bt_hop,
    bpm=bpm, trim=False, tightness=150, start_bpm=120.0
)
```

**期待される効果**:
- `tightness=150`: ビートトラッキングがより安定したテンポを優先
- `start_bpm=120.0`: 初期テンポを 120 BPM に設定（一般的な範囲）
- 162 BPM のような中速〜速テンポでの検出精度向上

#### Fix B: onset strength のパラメータ調整

**ファイル**: `backend/main.py`
**行**: 670 (onset_envelope 生成)

**変更内容**:
```python
# 変更前
onset_envelope = librosa.onset.onset_strength(
    y=y, sr=sr, hop_length=hop_length
)

# 変更後
onset_envelope = librosa.onset.onset_strength(
    y=y, sr=sr, hop_length=hop_length,
    aggregate=np.median,  # 中央値の使用（平均よりノイズに強い）
    max_size=3  # ノイズ抑制（最大サイズ制限）
)
```

**期待される効果**:
- `aggregate=np.median`: 平均値の代わりに中央値を使用（外れ値の影響を軽減）
- `max_size=3`: 局所的なノイズを抑制

#### Fix C: オクターブ検証の改善

**ファイル**: `backend/main.py`
**行**: 1432 (SCORE_OVERRIDE_RATIO)

**変更内容**:
```python
# 現在
SCORE_OVERRIDE_RATIO = 1.25  # 緩和: 25%以上のスコア差で×0.5を適用

# 変更後
SCORE_OVERRIDE_RATIO = 1.15  # さらに緩和: 15%以上のスコア差で×0.5を適用
```

**期待される効果**:
- スコア比 1.25 以上（前の修正）→ 1.15 以上に緩和
- 半テンポ補正（219.9 → 110 BPM）が適用されやすくなる
- 110 BPM は依然として 162 BPM から遠いが、237 BPM よりは良い

## 実装計画

### Fix A: librosa.beat.beat_track() のパラメータ調整

**ファイル**: `backend/main.py`
**行**: 595

**変更内容**:
```python
# 変更前
_, bt_frames = librosa.beat.beat_track(
    onset_envelope=onset_env, sr=sr, hop_length=bt_hop,
    bpm=bpm, trim=False
)

# 変更後
_, bt_frames = librosa.beat.beat_track(
    onset_envelope=onset_env, sr=sr, hop_length=bt_hop,
    bpm=bpm, trim=False, tightness=150, start_bpm=120.0
)
```

### Fix B: onset strength のパラメータ調整

**ファイル**: `backend/main.py`
**行**: 670

**変更内容**:
```python
# 変更前
onset_envelope = librosa.onset.onset_strength(
    y=y, sr=sr, hop_length=hop_length
)

# 変更後
onset_envelope = librosa.onset.onset_strength(
    y=y, sr=sr, hop_length=hop_length,
    aggregate=np.median, max_size=3
)
```

### Fix C: OctaveVerify の SCORE_OVERRIDE_RATIO をさらに緩和

**ファイル**: `backend/main.py`
**行**: 1432

**変更内容**:
```python
# 変更前
SCORE_OVERRIDE_RATIO = 1.25  # 緩和: 25%以上のスコア差で×0.5を適用

# 変更後
SCORE_OVERRIDE_RATIO = 1.15  # さらに緩和: 15%以上のスコア差で×0.5を適用
```

## 変更ファイル一覧

| 変更 | ファイル | 行 | リスク |
|------|---------|-----|--------|
| beat_track パラメータ調整 | `backend/main.py` | 595 | 低（パラメータ追加のみ） |
| onset strength パラメータ調整 | `backend/main.py` | 670 | 低（パラメータ追加のみ） |
| SCORE_OVERRIDE_RATIO 緩和 | `backend/main.py` | 1432 | 低（閾値調整のみ） |

## 検証方法

### 1. バックエンド再起動とログ確認

```bash
cd backend
uvicorn main:app --reload
```

**期待されるログ**:
```
[DEBUG] Starting analysis...
[DEBUG] Detecting BPM...
[DEBUG] BPM refined via autocorrelation: 162.0 BPM (librosa.beat.beat_track 改善後)
[OctaveVerify] score_ratio=1.xx >= 1.15 -> overriding gate
[OctaveVerify] Selected: 110.0 BPM (factor=x0.5, score=0.xxx)
```

### 2. 同じ YouTube URL で再解析

https://youtu.be/fWHnghPgg4Q を再解析

**期待される結果**:
- BPM = 110 BPM（改善された librosa パラメータに基づいて）
- `bandscore-C#m-110bpm.json` に保存される

### 3. 解析結果JSONを確認

```bash
cat ~/Downloads/bandscore-C#m-*.json | grep -A1 "bpm"
```

**期待される値**:
- `"bpm": 110`

### 4. ブラウザでの同期確認

1. フロントエンドを起動 (`npm run dev`)
2. 同じ URL で解析
3. 再生してハイライトと原曲の同期を確認
4. BPM が 110 BPM であることを確認

## 期待される効果

1. **librosa パラメータの改善**: BPM 検出精度の向上
2. **半テンポ補正の適用促進**: SCORE_OVERRIDE_RATIO 緩和により 110 BPM が選ばれやすくなる
3. **ノイズ低減**: onset strength の aggregate=np.median と max_size=3
4. **安定したトラッキング**: tightness=150 と start_bpm=120.0 により

## トラブルシュート: madmom インストール失敗時

### 症状確認
```bash
cd backend
pip list | grep madmom
```

**madmom がインストールされていない場合**:
1. 以下のコマンドを試行：
```bash
pip install madmom==0.16.1
```

2. Windows 環境でエラーが出る場合：
   - Visual Studio Build Tools (C++ コンパイラ) のインストールが必要
   - `pip install --verbose madmom` で詳細ログを確認

3. それでも失敗する場合：
   - 音源分離なしで librosa 改善案（Fix A, B, C）で対応
   - Docker 環境での madmom インストールを検討

### madmom 成功時の次ステップ

madmom がインストールできたら、既存の `detect_bpm_madmom()` 関数を既存の BPM 検出に統合する手順：

1. Lines 1774-1881 のカスタム BPM スキャンを削除またはコメントアウト
2. detect_bpm_madmom() を呼ぶように detect_bpm() 関数を修正
3. BassTempoCheck や OctaveVerify に madmom 結果を使用するように調整

## Context

現在のBPM検出システム（librosa.beat.beat_track + ヒューリスティック補正）では、162 BPM を正確に検出できていません。YouTube URL (https://youtu.be/fWHnghPgg4Q) の解析では 219.9 BPM が検出され、補正ロジックを通っても 110 BPM または 237 BPM になり、162 BPM に届きません。

**現在の制限**:
1. **librosa.beat.beat_track() の精度限界**: ヒューリスティックなアプローチ
2. **誤検出の連鎖**: 219.9 BPM (誤検出) → OctaveVerify (ブロック) → BassTempoCheck (158 BPM または 237 BPM)
3. **補正ロジックの限界**: 複雑な補正が重なり、結果的にさらに悪化する場合がある

**根本的な解決策**:
- **madmom**: ディープラーニングベースのビートトラッカー、より正確なBPM検出
- **音源分離**: ドラムトラック抽出により、クリーンなビート信号を取得

## 解決策

### Phase 1: madmom の導入（優先度高）

madmom はディープラーニングベースのビートトラッキングライブラリで、librosa よりも正確にビートを検出できます。

**実装戦略**:
1. 現在のBPM検出（Lines 1774-1881）を madmom に置き換え
2. librosa.beat.beat_track() 呼び出し（Lines 595）も madmom に置き換え
3. 残りの解析フロー（クロマ処理、コード検出）は維持

**導入ライブラリ**:
```python
# requirements.txt に追加
madmom==0.16.1
```

**コード変更**:
```python
# 変更前 (backend/main.py:1774-1881)
# カスタム BPM スキャン (60-240 BPM) + オクターブ検証 + ベース補正

# 変更後
from madmom.features.beats import BeatDetection, BeatTrackingProcessor
from madmom.audio.signal import Signal

def detect_bpm_madmom(y, sr):
    """madmom による正確な BPM 検出"""
    sig = Signal(y, sr)
    onsets = BeatDetection()(sig)  # RNN ベースの onset 検出
    beats = BeatTrackingProcessor()(onsets)  # HMM ベースのビートトラッキング
    if len(beats.times) > 1:
        bpm = 60.0 / np.mean(np.diff(beats.times))
    else:
        bpm = 120.0  # フォールバック
    return bpm, beats.times
```

**期待される効果**:
- 162 BPM のような複雑なリズムも正確に検出
- librosa.beat.beat_track() の誤検出（219.9 BPM など）を回避
- 補正ロジックの依存度を低減

### Phase 2: 音源分離（Demucs/Spleeter）の導入（中優先度）

既存の HPSS（調波/打撃楽分離）を拡張して、より深いステム分離を実装します。

**選択肢**:
- **Demucs**: ディープラーニングベース、高精度だが重い
- **Spleeter**: ライトウェイト、Demucs の軽量版

**推奨**: まずは Spleeter（軽量）で導入し、必要なら Demucs に移行

**導入ライブラリ**:
```python
# requirements.txt に追加（Spleeter の場合）
spleeter==2.4.0  # 軽量版

# または Demucs（より高精度だが重い）
demucs==4.0.0  # 完整版
```

**コード変更**:
```python
# 変更前 (backend/main.py:2077-2086)
# HPSS (librosa.effects.harmonic) のみ使用

# 変更後
from spleeter.separator import Separator
import tempfile

def separate_drums(y, sr):
    """ステム分離によるドラム抽出"""
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, y.T, sr)
        separator = Separator('spleeter:2stems-4bands', multiprocess=False)
        _, stems = separator.separate_to_file(tmp.name)
    # ドラムステムをロード
    y_drums, _ = sf.read(stems['drums'])
    return y_drums

# クロマ処理でステム分離を適用
# y_drums = separate_drums(y, sr)  # ドラム抽出
# その後、y_drums を使って BPM 検出
```

**期待される効果**:
- クリーンなビート信号から BPM 検出
- コード検出精度向上（ベースライン抽出）
- 音楽的なステム情報を API レスポンスに追加可能

## 実装計画

### Phase 1: madmom 導入

**ファイル**: `backend/main.py`, `backend/requirements.txt`
**行**: 1774-1881 (BPM検出), requirements.txt

**変更内容**:
```python
# 1. requirements.txt に追加
echo "madmom==0.16.1" >> backend/requirements.txt

# 2. backend/main.py のインポートに追加
from madmom.features.beats import BeatDetection, BeatTrackingProcessor
from madmom.audio.signal import Signal

# 3. 新たな関数を追加
def detect_bpm_madmom(y, sr):
    """madmom による BPM 検出"""
    sig = Signal(y, sr)
    onsets = BeatDetection()(sig)
    beats = BeatTrackingProcessor()(onsets)
    if len(beats.times) > 1:
        bpm = 60.0 / np.mean(np.diff(beats.times))
    else:
        bpm = 120.0
    return bpm, beats.times

# 4. 既存の detect_bpm() 関数を置き換え (Line ~1774)
# detect_bpm_madmom() を呼ぶように変更
```

**期待されるログ**:
```
[MADMOM] RNN onset detection complete: 142 onsets
[MADMOM] HMM beat tracking complete: 222 beats
[MADMOM] Calculated BPM: 162.4 (from beat intervals)
```

### Phase 2: 音源分離導入（オプション）

**ファイル**: `backend/main.py`, `backend/requirements.txt`
**行**: ~2077 (クロマ処理), requirements.txt

**変更内容**:
```python
# 1. requirements.txt に追加
echo "spleeter==2.4.0" >> backend/requirements.txt  # または demucs==4.0.0

# 2. backend/main.py のインポートに追加
from spleeter.separator import Separator
import soundfile as sf

# 3. 新たな関数を追加
def separate_drums(y, sr):
    """ステム分離によるドラム抽出"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, y.T, sr)
        separator = Separator('spleeter:2stems-4bands', multiprocess=False)
        _, stems = separator.separate_to_file(tmp.name)
    y_drums, _ = sf.read(stems['drums'])
    return y_drums

# 4. クロマ処理でステム分離を適用（オプション）
# analyze_audio_file() 関数でステム分離を呼ぶ
```

**期待されるログ**:
```
[Spleeter] Separating audio into stems...
[Spleeter] Drums stem extracted (quality: high)
[BPM-Detecting] Using drums stem for beat detection
```

## 変更ファイル一覧

| 変更 | ファイル | 行 | リスク |
|------|---------|-----|--------|
| madmom 依存追加 | `backend/requirements.txt` | - | 低 (ライブラリ追加のみ) |
| BPM検出関数置き換え | `backend/main.py` | ~1774-1881 | 中 (新たなアルゴリズム) |
| 音源分離依存追加 | `backend/requirements.txt` | - | 中 (Spleeter/Demucs は重い) |
| ステム分離関数追加 | `backend/main.py` | ~2077 | 中 (オプション機能) |

## 検証方法

### 1. 依存インストール

```bash
cd backend
pip install -r requirements.txt
```

### 2. バックエンド再起動とログ確認

```bash
cd backend
uvicorn main:app --reload
```

**期待されるログ**:
```
[MADMOM] Loading RNN model...
[MADMOM] RNN onset detection complete: 142 onsets
[MADMOM] HMM beat tracking complete: 222 beats
[MADMOM] Calculated BPM: 162.4 (from beat intervals)
[ModeBPM] Mode interval = 0.370s, Mode BPM = 162.1
```

### 3. 同じ YouTube URL で再解析

https://youtu.be/fWHnghPgg4Q を再解析

**期待される結果**:
- BPM = ~162 BPM（正確！）
- `bandscore-C#m-162bpm.json` に保存される

### 4. 解析結果JSONを確認

```bash
cat ~/Downloads/bandscore-C#m-*.json | grep -A1 "bpm"
```

**期待される値**:
- `"bpm": 162` または近い値 (158-165 BPM)

### 5. ブラウザでの同期確認

1. フロントエンドを起動 (`npm run dev`)
2. 同じ URL で解析
3. 再生してハイライトと原曲の同期を確認
4. BPM が ~162 BPM であることを確認

## 期待される効果

### Phase 1 (madmom 導入) のみ
1. **正確な BPM 検出**: 162 BPM のような複雑なリズムも正確に検出
2. **補正ロジック依存の低減**: librosa の誤検出を回避
3. **一貫性向上**: beat_times と bar タイミングが整合する

### Phase 2 (音源分離導入時)
1. **クリーンなビート信号**: ドラム抽出により、ボーカル/ベースの干渉を除去
2. **精度向上**: 複数のステムを分析してより正確な結果
3. **高度な機能**: ステム別解析（ドラム、ベース、メロディ）が可能に

## 緊急対応: madmom インポートの回避

### 現状の問題

**ユーザーからのログ**:
```
INFO:     Will watch for changes in these directories: [...]
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process [17000] using StatReload
...
File "C:\Users\USER\.gemini\antigravity\guitar-tab\backend\main.py", line 15, in <module>
    from madmom.features.beats import BeatDetection, BeatTrackingProcessor
ModuleNotFoundError: No module named 'madmom'
```

**根本原因**:
- madmom パッケージが正常にインストールされていない
- [backend/main.py](backend/main.py) には `from madmom.features.beats import` が記述されているが、madmom が利用できない
- Uvicorn が変更を検出し、再読み込みを試みたが失敗

### 解決策

### 即時解決策: madmom インポートを一時的に無効化

**変更**: [backend/main.py](backend/main.py) の madmom インポートをコメントアウト

**方法**:
1. [backend/main.py](backend/main.py) を開く
2. 行 13-16 の madmom インポートをコメントアウト:
   ```python
   # from madmom.features.beats import BeatDetection, BeatTrackingProcessor
   # from madmom.audio.signal import Signal
   ```
3. ファイルを保存

**期待される効果**:
- Uvicorn が正常に起動できる
- コードは librosa のみを使用（madmom に依存しない）
- 解析が進行する
- BPM = 110 BPM（librosa パラメータ調整による）

### 手順: madmom インストール再試行（オプション）

madmom を後で試す場合：

1. **Windows でのビルド**:
   - Visual Studio Build Tools (https://visualstudio.microsoft.com/) をインストール
   - C++ 14 以上のビルド環境を設定
   - コマンドプロンプトから実行:
     ```bash
     cd backend
     pip install --verbose madmom==0.16.1
     ```

2. **Conda 環境の使用**（推奨）:
   ```bash
   conda create -n guitar-analysis python=3.10
     conda activate guitar-analysis
     conda install -c conda-forge madmom
     pip install madmom==0.16.1
     ```

3. **Wheels の使用**:
   ```bash
     cd backend
     pip install madmom==0.16.1 --prefer-binary
     ```

4. **詳細なエラーログ**:
   ```bash
     pip install --verbose madmom==0.16.1 2>&1 | tee madmom_install.log
     ```

5. **成功確認**:
   ```bash
     python -c "from madmom.features.beats import BeatDetection; print('madmom is available')"
     ```

## 期待される効果

### 即時解決策適用時
- Uvicorn が正常に起動する
- 分析が進行する（librosa のみを使用）
- BPM = 110 BPM（改善されたパラメータ調整により）
- ハイライトと原曲の同期が機能する

### 次のステップ

1. madmom インポートをコメントアウトする
2. バックエンドを再起動
3. 同じ YouTube URL で再解析して確認

### 現状の問題

**ユーザー報告**:
- "結果は変わらず"
- "解析が進みません" (0% で進行中)
- "バックエンドログも新しく出るものはない"

**推定される原因**:
1. バックエンドが再起動されていない
2. uvicorn の --reload オプションが正しく動作していない
3. モジュールキャッシュの問題

## 解決策

### 手順 1: バックエンドの完全再起動

```bash
# ターミナルを停止（Ctrl+C）して再起動
cd backend
python -m uvicorn main:app
```

**期待**: 変更されたコードが読み込まれ、新しいログが出力される

### 手順 2: コード変更の確認

[backend/main.py](backend/main.py) を確認して、以下の変更が正しく反映されていることを確認してください：

1. **Fix A**: librosa.beat.beat_track() のパラメータ (Lines 2170-2176)
   ```python
   _, bt_frames = librosa.beat.beat_track(
       onset_envelope=onset_env, sr=sr, hop_length=bt_hop,
       bpm=bpm, trim=False,
       tightness=150,  # ← 追加済み
       start_bpm=120.0  # ← 追加済み
   )
   ```

2. **Fix B**: librosa.onset.onset_strength() のパラメータ (Line 2168)
   ```python
   onset_env = librosa.onset.onset_strength(
       y=y, sr=sr, hop_length=bt_hop,
       aggregate=np.median,  # ← 追加済み
       max_size=3  # ← 追加済み
   )
   ```

3. **Fix C**: OctaveVerify の SCORE_OVERRIDE_RATIO (Line 1481)
   ```python
   SCORE_OVERRIDE_RATIO = 1.15  # ← 緩和済み (1.25 → 1.15)
   ```

### 手順 3: デバッグログの確認

バックエンドを再起動した後、以下のデバッグログが期待されます：

```python
# 期待される新しいログ
[DEBUG] Starting analysis...
[DEBUG] Detecting BPM...
[OctaveVerify] score_ratio=1.xx >= 1.15 -> overriding gate
[OctaveVerify] Selected: 110.0 BPM (factor=x0.5, score=0.xxx)
```

**注**:
- 219.9 BPM が 162 BPM または 110 BPM になるはず
- BPM 検出の改善があれば、`[BassTempoCheck]` ログが異なるはず
- 解析が正常に進行し、最終的に JSON レスポンスが返るはず

### 手順 4: 変更が反映されない場合

もし変更が反映されない場合：

1. **uvicorn のキャッシュをクリア**:
   ```bash
   cd backend
   rm -rf __pycache__
   ```

2. **Python のバイトコードを再コンパイル**:
   ```bash
   cd backend
   python -m py_compile *.py
   ```

3. **代替案: コードを確認してから解析を進める**

もし librosa パラメータ調整でも解析が 0% で進行しない場合、根本原因は別の可能性があります：

- **ファイルが正しく保存されていない**: main.py の変更が保存されていない
- **インデントやスペースの問題**: コードのインデントやタブが壊れている
- **リソース制限**: ディスク I/O またはメモリ不足

## 次のアクション

以下の手順で順に実行してください：

1. バックエンドを完全に再起動（手順 1）
2. コード変更の確認（手順 2）
3. デバッグログを確認（手順 3）
4. まだ進まない場合は、問題箇所を報告してください

期待される結果:
- 解析が正常に開始する
- ログでパラメータ調整の効果が確認される
- BPM = 110 BPM (改善されたパラメータに基づいて）

## リスクと限界

### Phase 1 のリスク: モデル依存

**懸念**:
- madmom はトレーニング済みモデルを使用
- 極端なジャンル（電子音楽など）で優れる
- 異常にトレーニングしたモデルが必要な場合がある

**対策**:
- 事前にテストして精度を確認
- モデルの更新メカニズムを検討

### Phase 2 のリスク: 計算コストとメモリ

**懸念**:
- Spleeter/Demucs は重い（モデルサイズが大きい）
- メモリ使用量が増加
- 処理時間が長くなる

**対策**:
- Spleeter（軽量版）から導入
- ステム分離をオプション機能にする
- 大きなファイルではスキップする検討

### 根本的な限界: 精度のトレードオフ

**問題**:
- いかなるアプローチでも 100% 正確な BPM 検出は困難
- 音楽的なニュアンスやテンポチェンジは検出が難しい

**改善策**:
1. **複数のアプローチの組み合わせ**: madmom + ヒストグラム + ピーク検出
2. **ユーザーフィードバック**: 手動 BPM 調整機能の提供
3. **マルチパス分析**: 異なる手法で BPM を推定し、最も確からしい値を採用
4. **AI アノテーション**: 事前ラベリングデータでモデルのファインチューニング

**推奨される優先順位**:
1. **madmom 導入**: 即時対応、リスク中、効果的
2. **テストと調整**: 実際のデータで精度を検証
3. **音源分離導入**: 中期対応、精度向上
4. **ユーザー機能拡張**: 手動調整、フィードバック
5. **継続的な改善**: モデル更新、アプローチ組み合わせ

## 変更ファイル一覧

| 変更 | ファイル | 行 | リスク |
|------|---------|-----|--------|
| TOLERANCE 厳格化 | `backend/main.py` | 1947 | 低 (閾値調整のみ) |

## 検証方法

### 1. バックエンド再起動とログ確認

```bash
cd backend
uvicorn main:app --reload
```

**期待されるログ**:
```
[BassTempoCheck] bass peak = 158 BPM, detected: 219.9 BPM
[BassTempoCheck] ratio = 0.718, not close to any valid ratio (TOLERANCE=0.05)
[BassTempoCheck] Keeping 110 BPM (OctaveVerify result)
[ModeBPM] Mode interval = 0.xxx s, Mode BPM = xxx.x
```

### 2. 同じ YouTube URL で再解析

https://youtu.be/fWHnghPgg4Q を再解析

**期待される結果**:
- BPM = 110 BPM
- `bandscore-C#m-110bpm.json` に保存される

### 3. 解析結果JSONを確認

```bash
cat ~/Downloads/bandscore-C#m-*.json | grep -A1 "bpm"
```

**期待される値**:
- `"bpm": 110`

### 4. ブラウザでの同期確認

1. フロントエンドを起動 (`npm run dev`)
2. 同じ URL で解析
3. 再生してハイライトと原曲の同期を確認
4. BPM が 110 BPM であることを確認

## 期待される効果

1. **ratio=0.718 拒絶**: 158 BPM が採用されなくなる
2. **110 BPM 維持**: OctaveVerify の結果が維持される
3. **一貫性向上**: beat_times と bar タイミングが 110 BPM で揃う
4. **ただし**: 依然として 162 BPM に届かない (52 BPM 遅い)

## リスクと限界

### リスク: 110 BPM は依然として遅い

**懸念**:
- ユーザーが求める 162 BPM には届かない
- 52 BPM のズレは聴覚的に明らかに遅い

**対策**:
- 将来的に madmom や音源分離を実装
- librosa.beat.beat_track() のパラメータ調整を検討

### 根本的な限界: ヒューリスティックなアプローチ

**問題**:
- Librosa.beat.beat_track() はヒューリスティックなアプローチ
- 219.9 BPM という誤検出は完全に防ぐのは困難

**根本的な解決には**:
1. **madmom の導入**: ディープラーニングベースのビートトラッカー
2. **音源分離**: Demucs/Spleeter でドラムトラック抽出
3. **複数の手法の組み合わせ**: ヒストグラム + ピーク検出 + 連続性チェック

**推奨される優先順位**:
1. Fix (TOLERANCE 厳格化): 即時対応、リスク低
2. librosa.beat.beat_track() パラメータ調整: 中期対応、効果的
3. madmom 導入: 長期対応、根本的解決
4. 音源分離: 長期対応、精度向上
