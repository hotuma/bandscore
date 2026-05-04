# コード検出精度向上 + テスト修正: HMM + CQTクロマ

## 緊急修正 Phase 2: クロマコントラスト正規化（ペダルトーン除去）

### 現状（実装済み・未解決）

以下はすでに実装済み:
- `temperature=4.0`, `bass_weight=0.50`, `self-transition=0.50`
- `break_long_stagnation_runs(max_consecutive=8)` を safety net として復活

しかしログに依然 `unique chords: 1`, `Cannot break stagnation - no alternative chord available` が出続けている。

### 根本原因（特定済み）

CQT は低音域の周波数分解能が高いため、楽曲の **C# ベースペダルトーン**（通奏低音）を過剰に強調する。
その結果、全セグメントで C# ピッチクラスが dominant となり、C#maj7 の emission が他を圧倒。

- `break_long_stagnation_runs` は「他のコードが存在しない」場合には機能しない
- `bass_weight=0.50` に増やしたことでペダルトーンの影響をさらに増幅してしまった

### 修正内容（2箇所）

#### 1. `apply_chroma_contrast()` を追加（`compute_chroma_cqt` の直後）

```python
def apply_chroma_contrast(chroma: np.ndarray, filter_size: int = 50) -> np.ndarray:
    """
    ローリング最小値を減算してペダルトーン（通奏低音）を除去する。
    filter_size=50: 50フレーム × 2048/22050 ≈ 4.6秒のウィンドウ
    各ピッチクラスで持続する背景エネルギーを除いて、コード変化を際立たせる。
    """
    from scipy.ndimage import minimum_filter1d
    chroma_bg = minimum_filter1d(chroma, size=filter_size, axis=1, mode='reflect')
    chroma = np.maximum(0.0, chroma - chroma_bg)
    chroma_sum = np.sum(chroma, axis=0, keepdims=True)
    return np.where(chroma_sum > 1e-8, chroma / chroma_sum, chroma)
```

挿入位置: `compute_chroma_cqt()` (行333) の直後（行334付近）

#### 2. `analyze_audio_file()` の2箇所を変更

**A. クロマ抽出直後に contrast 適用**（行2004-2010付近）:
```python
chroma = compute_chroma_cqt(y, sr, hop_length=hop_length)
chroma = apply_chroma_contrast(chroma, filter_size=50)   # ← 追加

bass_chroma = compute_bass_chroma(y, sr, hop_length=hop_length)
bass_chroma = apply_chroma_contrast(bass_chroma, filter_size=50)  # ← 追加
```

**B. `detect_chords_hmm` の `bass_weight` を戻す**（行2105付近）:
```python
# 変更前
bass_weight=0.50,   # 増やしたがペダルトーンを増幅してしまった

# 変更後
bass_weight=0.25,   # contrast 正規化後は適切な重みに戻す
```

### 修正後の期待動作

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| `apply_chroma_contrast` | なし | CQT chroma・bass chroma 両方に適用 |
| `bass_weight` | 0.50 | 0.25（正規化後は過剰に強調不要） |

ペダルトーン除去により C# の持続的優位が解消され、Fm vs C#maj7 などコード変化が emission に反映される。

### 検証

```bash
cd backend
uvicorn main:app --reload
# ブラウザで https://youtu.be/ZIEQDjrAdwE を解析し:
# - unique chords が 4+ になること
# - [STAGNATION] WARNING ログが出ないこと
```

---

## Context

現在のコード検出はSTFT-based chromaによるテンプレートマッチング＋ルールベースのStagnation Prevention（detect_chords_matrix → smooth_chord_sequence_stagnation_aware → break_long_stagnation_runs）で実装されている。この方式はアドホックなパラメータ依存が高く、Chordifyレベルの精度に届かない。

**主な問題点:**
1. STFT chroma は低音弦（E2=82Hz）付近の周波数分解能が低い
2. Stagnation Prevention は局所的なルールのみで大域的なコード列最適化ができない
3. J-Pop/Jロックに多い vi→IV→I→V 系進行の検出が不安定

**改善方針:** CQT chromaで特徴量を改善し、HMM（Viterbiアルゴリズム）でコード列を大域最適化する。商用ツール（Chordify初期）が使用した手法。

---

## 修正ファイル

`backend/main.py` のみ（新規依存なし: librosa 0.10.1/numpy/scipy）

---

## 実装ステップ

### Step 1: compute_chroma_cqt() を追加（行313直後）

```python
def compute_chroma_cqt(y: np.ndarray, sr: int, hop_length: int = 2048) -> np.ndarray:
    """CQT-based chroma: ギター低音弦の周波数分解能が STFT より優れる"""
    y_harmonic = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic, sr=sr, hop_length=hop_length,
        bins_per_octave=36, norm=None,
    )
    del y_harmonic
    chroma_log = np.log1p(10.0 * chroma)
    return chroma_log / (np.sum(chroma_log, axis=0, keepdims=True) + 1e-8)
```

### Step 2: build_transition_matrix() を追加（行493直前）

音楽理論に基づく (72, 72) コード遷移確率行列を構築:
- 自己遷移: 0.65（同コードが続く高確率）
- 完全5度上（属和音 C→G 等）: +0.08
- 完全5度下（下属和音 C→F 等）: +0.06
- 相対長調/短調（Am↔C 等）: +0.08
- その他全て: 0.002（ベース確率）
- 行ごとにL1正規化、log確率で返す

コードラベルの形式は実際の `CHORD_LABELS` 形式（"A", "Am", "Am7", "Amaj7", "Asus4", "A7"）に準拠。
minorの判定: `suffix == "m" or suffix == "m7"` で判断（"maj7"はmajor）。

### Step 3: viterbi_decode() を追加（build_transition_matrix の直後）

```python
def viterbi_decode(emission_log_probs, log_trans, log_init):
    """ログ確率空間でのViterbiデコード。計算量: O(T × n²) = O(200 × 72²) ≈ 100万演算"""
    T, n = emission_log_probs.shape
    dp = np.full((T, n), -np.inf)
    bp = np.zeros((T, n), dtype=np.int32)
    dp[0] = log_init + emission_log_probs[0]
    for t in range(1, T):
        trans_scores = dp[t-1, :, np.newaxis] + log_trans  # (n, n) broadcast
        bp[t] = np.argmax(trans_scores, axis=0)
        dp[t] = trans_scores[bp[t], np.arange(n)] + emission_log_probs[t]
    path = np.zeros(T, dtype=np.int32)
    path[T-1] = np.argmax(dp[T-1])
    for t in range(T-2, -1, -1):
        path[t] = bp[t+1, path[t+1]]
    return path
```

### Step 4: キャッシュ変数をモジュールレベルに追加（行407、build_chord_templates呼び出し直後）

```python
_HMM_LOG_TRANS = build_transition_matrix(CHORD_LABELS)
_n_chords = len(CHORD_LABELS)
_HMM_LOG_INIT = np.full(_n_chords, np.log(1.0 / _n_chords), dtype=np.float64)
```

### Step 5: detect_chords_hmm() を追加（detect_chords_matrix の直後、行695以降）

`detect_chords_matrix()` と同じシグネチャ・戻り値型 `(list[str], str, int)` を維持（後方互換）。

処理フロー:
1. コサイン類似度スコア（現行と同じ: main_scores + bass_scores）
2. diatonicペナルティ加算（penalty_maskがTrueの箇所を -penalty_value）
3. `softmax(scores × temperature=8.0)` → emission log確率
4. Viterbiデコード（`_HMM_LOG_TRANS`, `_HMM_LOG_INIT`を使用）
5. コード名リストに変換、最終コードとrun_lengthを計算して返す

`forced_last_chord` / `forced_run_length` は後方互換のため引数に受け取るが内部では使用しない。

### Step 6: analyze_audio_file() の呼び出し変更

**行1801-1808 付近（chroma抽出）:**
```python
# 変更前
chroma = compute_chroma_log(y, sr, hop_length=hop_length)
# 変更後
chroma = compute_chroma_cqt(y, sr, hop_length=hop_length)
```
`bass_chroma` は `compute_bass_chroma()` のまま維持（低域STFTで根音検出に特化）。

**行1899-1940 付近（コード検出・後処理）:**
```python
# 変更前
raw_chords, ... = detect_chords_matrix(main_matrix, bass_matrix, ...)
smoothed_chords = smooth_chord_sequence_stagnation_aware(raw_chords, ...)
smoothed_chords = break_long_stagnation_runs(smoothed_chords, ...)

# 変更後
smoothed_chords, final_last_chord, final_run_length = detect_chords_hmm(
    main_matrix, bass_matrix,
    penalty_mask=penalty_mask, penalty_value=0.20,
    main_weight=0.6, bass_weight=0.35,
    temperature=8.0,
    forced_last_chord=forced_last_chord,
    forced_run_length=forced_run_length,
)
# HMM後処理は不要。ログ出力のみ残す。
```

---

## 再利用する既存関数

- `cosine_similarity_matrix()` (行494) — emissionスコア計算に流用
- `chord_root_index()` — build_transition_matrix内で流用
- `compute_bass_chroma()` (行315) — そのまま維持
- `get_diatonic_chords_for_key()` — diatonicペナルティマスク生成（呼び出し側は変更なし）
- `CHORD_LABELS`, `TEMPLATE_MATRIX` — グローバルキャッシュをそのまま参照

---

## 削除・不要になる後処理

HMMのViterbiが大域最適化を担うため以下は削除（関数自体は残してよい）:
- `smooth_chord_sequence_stagnation_aware()` の呼び出し
- `break_long_stagnation_runs()` の呼び出し
- `calc_max_run()` ヘルパー（デバッグ用に簡略版は残す）

---

---

## 追加タスク: verify_modes.py のテスト修正

### 問題
`backend/tests/verify_modes.py` 64行目が古い仕様を前提にしている：
```python
assert result.get("bars") is None  # 失敗する（現在は bars を返す）
```

### 原因
`main.py` 2420行目でコメント付きで意図的に変更済み：
```python
"bars": all_bars,  # Return bars even in Preview (limited by duration cap)
```
`verify_preview_content.py` はすでに新仕様（bars を返す）に合わせてあるが、`verify_modes.py` だけ更新されていない。

### 修正内容（1行変更）

**ファイル:** `backend/tests/verify_modes.py` 64行目

```python
# 変更前（古い仕様）
assert result.get("bars") is None

# 変更後（現在の仕様）
assert isinstance(result.get("bars"), list)
```

これにより：
- PREVIEW モードでも bars（コード一覧）が返される現在の仕様を正しく検証
- `is_preview=True` フラグで PREVIEW であることはフロントエンドが判断できる

---

## 検証方法

```bash
cd backend

# 1. 基本動作確認（音声ファイル不要）
python -c "
from main import build_transition_matrix, viterbi_decode, CHORD_LABELS, _HMM_LOG_TRANS, _HMM_LOG_INIT
import numpy as np
trans = np.exp(_HMM_LOG_TRANS)
print('Row sums:', trans.sum(axis=1).min(), trans.sum(axis=1).max())  # 全て ~1.0
print('Self-trans mean:', np.exp(np.diag(_HMM_LOG_TRANS)).mean())  # ~0.65
T, n = 20, 72
path = viterbi_decode(np.random.randn(T, n), _HMM_LOG_TRANS, _HMM_LOG_INIT)
print('Path shape:', path.shape, 'OK')
"

# 2. CQT vs STFT の特徴量比較（音声ファイルが必要）
python -c "
import librosa, numpy as np
from main import compute_chroma_log, compute_chroma_cqt, highpass_filter
y, sr = librosa.load('ガソリン0812.m4a', sr=22050, mono=True, duration=30)
y = highpass_filter(y, sr)
c1 = compute_chroma_log(y, sr)
c2 = compute_chroma_cqt(y, sr)
print('STFT shape:', c1.shape, 'CQT shape:', c2.shape)  # 同じ (12, T)
"

# 3. 既存テストで回帰確認
python tests/verify_modes.py
python tests/verify_preview_content.py
```

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| CQTがSTFTより重い（メモリ/速度） | bins_per_octave=24に下げる（24→12折り畳みで高速化） |
| HMM自己遷移が高くコードが動かない | SELF_PROB=0.65→0.55 または temperature=8.0→10.0 に調整 |
| チャンク境界でコードが不連続 | forced_last_chordを log_init のone-hot確率として渡す（Phase2の改善） |
| CLAUDE.md「CQTはクラウドで不安定」 | librosa 0.10.1では改善済み。問題発生時はcompute_chroma_logにフォールバック |
