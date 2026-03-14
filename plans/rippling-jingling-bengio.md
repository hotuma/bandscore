# コード検出精度向上 - 第2弾（完了済み）+ Stagnation問題の発見

## Context

前回の精度向上計画（HPSS導入、bass chroma独立化、テンプレート60種拡張、diatonic penalty強化、重み調整）は既に実装済み。

**第2弾施策（✅ 実装完了）:**
- m7テンプレート追加（60→72テンプレート）
- ダイアトニックコード品質マッピング修正
- スムージング強化（多パス + 2小節アウトライヤー）
- デフォルトパラメータ更新

**新たに発見された問題:**
YouTubeURL解析（https://youtu.be/Pwht_zL3_go）で、Fm7が異常に連続して検出される現象が発生。調査の結果、**チャンク境界でstagnation状態がリセット**される設計上の欠陥を発見。

## 変更概要

すべて [backend/main.py](backend/main.py) + [backend/tests/verify_chord_accuracy.py](backend/tests/verify_chord_accuracy.py) のみ。フロントエンド変更は不要。

---

### 1. Minor 7th (m7) テンプレート追加 (60→72テンプレート)

**理由**: Am7, Dm7, Em7 はポップス/ジャズで最頻出のコード型。現在m7テンプレートが無いため、Am7 → Am or A7 に誤分類される。

**変更箇所**:

- **CHORD_TO_TAB** (line ~179後): 12個のm7タブ譜を追加
  ```
  "Cm7", "C#m7", "Dm7", "D#m7", "Em7", "Fm7",
  "F#m7", "Gm7", "G#m7", "Am7", "A#m7", "Bm7"
  ```

- **build_chord_templates()** (line ~352後): m7ベーステンプレート追加
  ```
  base_m7: root(1.0) + min3(0.4) + 5th(0.6) + min7(0.35)
  ```
  ループ内に `templates[f"{name}m7"]` を追加

---

### 2. ダイアトニックコード品質マッピングの修正

**理由**: 現在の `get_diatonic_chords_for_key()` は全スケール度数に対して7th/maj7/sus4を無差別に追加している。音楽理論的に誤りであり、非ダイアトニックコードがペナルティを回避してしまう。

**例 (Cメジャーキー)**:
| 度数 | 現在(誤) | 修正後(正) |
|------|----------|-----------|
| ii (D) | Dm, D7, Dmaj7, Dsus4 | Dm, Dm7, Dsus4 |
| V (G) | G, G7, Gmaj7, Gsus4 | G, G7, Gsus4 |
| vi (A) | Am, A7, Amaj7, Asus4 | Am, Am7 |

**変更箇所**: `get_diatonic_chords_for_key()` (line 407-434) を完全書き換え
- メジャー: I=maj7, ii=m7, iii=m7, IV=maj7, V=7, vi=m7, vii=m7
- マイナー: i=m7, ii=m7, III=maj7, iv=m7, v=m7, VI=maj7, VII=7
- sus4はI, ii, IV, Vのみ

---

### 3. スムージング強化 (多パス + 2小節アウトライヤー除去)

**理由**: 現在のsmooth_chord_sequence()は単一小節アウトライヤー(A-B-A→A-A-A)のみ対応。2小節アウトライヤー(A-B-B-A)やカスケード補正を見逃す。

**変更箇所**: `smooth_chord_sequence()` (line 728-739) を拡張
- 単一小節アウトライヤー: A-B-A → A-A-A (既存)
- 2小節アウトライヤー: A-B-B-A → A-A-A-A (新規)
- 最大2パス（収束したら早期終了）

---

### 4. 関数デフォルトパラメータ更新

**理由**: `detect_chords_matrix()` のシグネチャのデフォルト値が古い(0.15/0.7/0.3)。呼び出し側は0.20/0.6/0.4を渡しているが、デフォルトで呼ばれた場合に不整合が生じる。

**変更箇所**: line 474-476
```python
penalty_value: float = 0.15 → 0.20
main_weight: float = 0.7 → 0.6
bass_weight: float = 0.3 → 0.4
```

---

## 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| [backend/main.py](backend/main.py) | CHORD_TO_TAB, build_chord_templates, get_diatonic_chords_for_key, smooth_chord_sequence, detect_chords_matrix defaults |
| [backend/tests/verify_chord_accuracy.py](backend/tests/verify_chord_accuracy.py) | テンプレート数60→72、ダイアトニックアサーション修正 |

## 検証結果（第2弾施策）

✅ すべてのテストが成功:
- テンプレート数: 72 (12音 × 6タイプ)
- ダイアトニックマッピング: 音楽理論的に正しい7th品質を適用
- 実音声解析: Fm7が正常に検出されることを確認

---

## 新規発見: チャンク境界Stagnation問題

### 問題の症状

YouTube URL解析（https://youtu.be/Pwht_zL3_go, 278秒）で以下を観測:
- **Fm7が異常に連続**: Chunk 4-8でほぼFm7のみ検出（本来はコード進行があるはず）
- **ユニークコード数が少ない**: 各チャンクで2-5種類のみ
- **最終チャンクのキー変化が無視**: Chunk 9でG#m検出されるが、全体にFmペナルティマスクが適用される

### 根本原因（調査結果）

#### 1. **Stagnation状態がチャンク境界でリセット** ⭐最重要
- [main.py:1815-1822](main.py#L1815-L1822): チャンク処理は各々独立して`analyze_audio_file()`を呼び出し
- [main.py:502-654](main.py#L502-L654): `detect_chords_matrix()`は`run_length`を関数内部で初期化（line 571: `run_length = 1`）
- `max_repeat_segments=6`の制限は各チャンク内でのみ有効
- **結果**: Fm7が60秒チャンクで30小節連続しても、次チャンクで`run_length`がリセットされ、制限が無効化

#### 2. **キー検出が最初のチャンクのみ使用**
- [main.py:1877-1878](main.py#L1877-L1878): `key = key_votes[0]` でChunk 1のキーのみ採用
- Chunk 9でG#m検出されても無視
- **結果**: Fm7は`i=m7`（トニック）でペナルティ0、他コードは-0.20ペナルティ

#### 3. **bass_weight増加が根音を過剰強調**
- [main.py:1631-1632](main.py#L1631-L1632): `bass_weight=0.4`（旧0.3から増加）
- ベース音が一貫してF音を含む場合、Fm7スコアが不当に上昇
- `final_scores = main_scores * 0.6 + bass_scores * 0.4` (line 547)

#### 4. **スムージングがStagnationを増幅**
- [main.py:760-798](main.py#L760-L798): 2パススムージングで孤立コードを除去
- 例: `Fm7, C#maj7, Fm7, Fm7...` → `Fm7, Fm7, Fm7, Fm7...`
- 本来のコード変化を「ノイズ」として除去してしまう可能性

### 推奨修正案

| 優先度 | 修正内容 | 影響範囲 |
|--------|---------|---------|
| **高** | チャンク間stagnation状態の引き継ぎ | [main.py:1815-1822](main.py#L1815-L1822) チャンク処理ループ<br>[main.py:502](main.py#L502) detect_chords_matrix()に`forced_last_chord`と`forced_run_length`パラメータ追加 |
| **高** | キー検出の投票方式化 | [main.py:1877-1878](main.py#L1877-L1878) 最頻値or中央値を使用、キー変化を検出してログ出力 |
| **中** | bass_weight調整 | [main.py:1632](main.py#L1632) 0.4→0.35 or 0.3に戻す（ベース過剰強調を緩和） |
| **低** | 動的ペナルティ | stagnation検出時に`penalty_value`を0.20→0.30に増加 |

---

---

## 実装計画（ユーザー承認済み）

### 変更概要

3つの修正を実施:
1. **チャンク間stagnation状態引き継ぎ** - `forced_last_chord`と`forced_run_length`パラメータを追加
2. **キー投票方式** - 最初のチャンクのみではなく、全チャンクの多数決でキーを決定
3. **bass_weight調整** - 0.4 → 0.35 に下げてベース過剰強調を緩和

---

### 変更1: detect_chords_matrix() の拡張

#### [main.py:508](main.py#L508) - bass_weightデフォルト値変更
```python
# 変更前
bass_weight: float = 0.4,

# 変更後
bass_weight: float = 0.35,
```

#### [main.py:516-517](main.py#L516-L517) - 新規パラメータ追加
```python
# 追加（line 516の後）
    forced_last_chord: Optional[str] = None,
    forced_run_length: Optional[int] = None,
) -> tuple[list[str], str, int]:  # 戻り値の型も変更
```

**効果**: 前チャンクの最終コードとrun_lengthを引き継いでstagnation判定を継続可能に

#### [main.py:566-571](main.py#L566-L571) - 初期化ロジック変更
```python
# 変更前
out_idx = np.zeros(num_segments, dtype=np.int32)
last = int(np.argmax(scores[0]))
out_idx[0] = last
run_length = 1

# 変更後
out_idx = np.zeros(num_segments, dtype=np.int32)

# 強制状態があればそれを使用（チャンク境界の継続性）
if forced_last_chord is not None and forced_run_length is not None:
    try:
        last = chord_labels.index(forced_last_chord)
        run_length = forced_run_length
        print(f"[StagnationContinuity] Forcing initial state: last={forced_last_chord}, run_length={run_length}")
    except ValueError:
        print(f"[WARN] forced_last_chord '{forced_last_chord}' not in chord_labels, using argmax")
        last = int(np.argmax(scores[0]))
        run_length = 1
else:
    last = int(np.argmax(scores[0]))
    run_length = 1

out_idx[0] = last
```

#### [main.py:654](main.py#L654) - 戻り値をタプルに変更
```python
# 変更前
return [chord_labels[j] for j in out_idx]

# 変更後
final_last_chord = chord_labels[last]
final_run_length = run_length
return [chord_labels[j] for j in out_idx], final_last_chord, final_run_length
```

---

### 変更2: analyze_audio_file() の拡張

#### [main.py:1227](main.py#L1227) - 関数シグネチャ拡張
```python
# 変更前
def analyze_audio_file(file_path: str, progress_callback=None, offset_sec: float = 0.0,
                       duration_limit_sec: float | None = None, forced_bpm: float | None = None,
                       forced_phase: float | None = None) -> dict:

# 変更後
def analyze_audio_file(file_path: str, progress_callback=None, offset_sec: float = 0.0,
                       duration_limit_sec: float | None = None, forced_bpm: float | None = None,
                       forced_phase: float | None = None,
                       forced_last_chord: Optional[str] = None,
                       forced_run_length: Optional[int] = None) -> dict:
```

#### [main.py:1626-1633](main.py#L1626-L1633) - detect_chords_matrix呼び出し変更
```python
# 変更前
raw_chords = detect_chords_matrix(
    main_matrix,
    bass_matrix,
    penalty_mask=penalty_mask,
    penalty_value=0.20,
    main_weight=0.6,
    bass_weight=0.4
)

# 変更後
raw_chords, last_chord, run_length = detect_chords_matrix(
    main_matrix,
    bass_matrix,
    penalty_mask=penalty_mask,
    penalty_value=0.20,
    main_weight=0.6,
    bass_weight=0.35,  # 0.4から変更
    forced_last_chord=forced_last_chord,
    forced_run_length=forced_run_length,
)
```

#### [main.py:1680-1687](main.py#L1680-L1687) - 戻り値にstagnation状態追加
```python
# 変更前
return {
    "bpm": bpm,
    "duration_sec": round(duration_sec, 1),
    "time_signature": "2/4",
    "key": estimated_key,
    "bars": bars,
    "phase_offset_sec": round(phase_offset_sec, 4),
}

# 変更後
return {
    "bpm": bpm,
    "duration_sec": round(duration_sec, 1),
    "time_signature": "2/4",
    "key": estimated_key,
    "bars": bars,
    "phase_offset_sec": round(phase_offset_sec, 4),
    "last_chord": last_chord,
    "run_length": run_length,
}
```

---

### 変更3: チャンク処理ループとキー投票

#### [main.py:1700付近](main.py#L1700) - vote_key()ヘルパー関数を追加
```python
def vote_key(key_votes: list[str]) -> str:
    """
    全チャンクの多数決でキーを選択。
    30%以上のチャンクが異なるキーを検出した場合は警告を出力。
    """
    if not key_votes:
        return "Unknown"

    if len(key_votes) == 1:
        return key_votes[0]

    from collections import Counter
    counts = Counter(key_votes)
    most_common = counts.most_common()
    winner = most_common[0][0]
    winner_count = most_common[0][1]

    # 大きな不一致がある場合は警告
    if len(most_common) > 1:
        runner_up_count = most_common[1][1]
        if runner_up_count >= len(key_votes) * 0.3:
            print(f"[KeyVoting] WARNING: Key disagreement - {winner}: {winner_count}, {most_common[1][0]}: {runner_up_count}")
            print(f"[KeyVoting] All votes: {key_votes}")

    print(f"[KeyVoting] Selected key: {winner} (votes: {counts})")
    return winner
```

#### [main.py:1778付近](main.py#L1778) - stagnation状態変数の初期化
```python
# 追加
forced_last_chord = None
forced_run_length = None
```

#### [main.py:1815-1822](main.py#L1815-L1822) - チャンク分析呼び出しに状態を渡す
```python
# 変更前
raw = analyze_audio_file(
    file_path,
    progress_callback=chunk_cb,
    offset_sec=offset,
    duration_limit_sec=dur,
    forced_bpm=bpm,
    forced_phase=forced_phase
)

# 変更後
raw = analyze_audio_file(
    file_path,
    progress_callback=chunk_cb,
    offset_sec=offset,
    duration_limit_sec=dur,
    forced_bpm=bpm,
    forced_phase=forced_phase,
    forced_last_chord=forced_last_chord,
    forced_run_length=forced_run_length,
)
```

#### [main.py:1836付近](main.py#L1836) - 各チャンク後にstagnation状態を抽出
```python
# 追加（BPM初期化の後）
# 次チャンク用のstagnation状態を抽出
forced_last_chord = raw.get("last_chord")
forced_run_length = raw.get("run_length")
if forced_last_chord and forced_run_length:
    print(f"[ChunkMerge] Stagnation state: last_chord={forced_last_chord}, run_length={forced_run_length}")
```

#### [main.py:1877-1878](main.py#L1877-L1878) - キー選択を投票方式に変更
```python
# 変更前
key = key_votes[0] if key_votes else "Unknown"

# 変更後
key = vote_key(key_votes)
```

---

## 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| [backend/main.py](backend/main.py) | detect_chords_matrix, analyze_audio_file, チャンク処理ループ, vote_key追加 |

## エッジケース対応

1. **後方互換性**: 新規パラメータはすべてOptional（デフォルトNone）なので既存コードは影響なし
2. **無効なforced_last_chord**: try-exceptでargmaxにフォールバック、警告出力
3. **空のkey_votes**: vote_key()が"Unknown"を返す
4. **キー不一致**: 30%以上のチャンクが異なるキーの場合、警告を出力
5. **最初のチャンク**: forced_last_chord=Noneで標準のargmax初期化

## 検証方法

```bash
# 1. 既存テストが通ることを確認
cd backend
python tests/verify_chord_accuracy.py

# 2. 問題のYouTube URLで再解析
# フロントエンドから https://youtu.be/Pwht_zL3_go を解析
# 期待結果: Fm7の連続が減少、より多様なコード検出

# 3. ログで確認すべき項目
# - [StagnationContinuity] が表示される（チャンク2以降）
# - [ChunkMerge] Stagnation state が表示される
# - [KeyVoting] Selected key が表示される
# - 各チャンクのunique chords数が増加しているか
```

## 期待される効果

1. **Fm7過剰検出の解消**: チャンク境界を越えてstagnation制限が機能
2. **正確なキー検出**: 全チャンクの多数決でノイズに強いキー選択
3. **バランス改善**: bass_weight=0.35でベース過剰強調を緩和しつつ、根音検出の恩恵は維持
