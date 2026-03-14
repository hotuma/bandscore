# コード検出精度向上プラン

## Context（背景）

YouTube URL `https://youtu.be/Pwht_zL3_go` の解析結果（`analysis-1773408178704.json`）において、以下の過剰なコード停滞が検出されました：

- **Bars 1-7**: Fm7が7バー連続
- **Bars 8-21**: C#maj7が14バー連続
- **Bars 70-78**: G#maj7が9バー連続

現在のバックエンド実装（[backend/main.py](backend/main.py)）は、既にStagnation Prevention Algorithm（停滞防止アルゴリズム）を実装していますが、パラメータが緩すぎるため、実際の楽曲では不自然な長時間の同一コード継続が発生しています。

### 根本原因

1. **max_repeat_segments = 6**が緩すぎる（6バー～6秒の停滞を許可）
2. **Hard cap at 12 bars**（max_repeat_segments × 2）により、12バーまで継続可能
3. **long_stag_penalty = 0.60**が弱すぎてコード変更を強制できない
4. **Progressive gap escalation**が低flux時に適用され、ペナルティが回避される
5. **Smoothing pass**は1-2バーの外れ値のみ除去し、長期停滞に対処しない

## 実装アプローチ：ハイブリッド戦略

段階的にパラメータ調整とアルゴリズム改善を組み合わせ、最大停滞を**4-6バー以内**に抑えます。

---

## Phase 1: パラメータ調整（最も安全、即効性あり）

### 変更1: max_repeat_segments の削減

**ファイル**: [backend/main.py:512](backend/main.py#L512)

**変更前**:
```python
max_repeat_segments: int = 6,
```

**変更後**:
```python
max_repeat_segments: int = 4,  # 6 → 4に削減
```

**理由**:
- 4バー = 8ビート（2/4拍子）= 約4秒の同一コード
- 音楽的に妥当（ポップソングは通常2-4バーごとにコード変化）
- Hard capが12バー → 8バーに自動的に改善

---

### 変更2: long_stag_penalty の強化

**ファイル**: [backend/main.py:515](backend/main.py#L515)

**変更前**:
```python
long_stag_penalty: float = 0.60,
```

**変更後**:
```python
long_stag_penalty: float = 0.85,  # 0.60 → 0.85に増強
```

**理由**:
- 現在の0.60では強いコードスコアを覆せない
- 0.85は不安定化せずにコード変更を強制できる十分な強度

---

### 変更3: Progressive Escalation のより積極的な設定

**ファイル**: [backend/main.py:630-640](backend/main.py#L630-L640)

**変更前**:
```python
adjusted_gap_threshold = 0.10 + 0.03 * excess
```

**変更後**:
```python
adjusted_gap_threshold = 0.08 + 0.05 * excess
```

**効果比較**:

| run_length | excess | 現在の閾値 | 新しい閾値 | 効果 |
|------------|--------|-----------|-----------|------|
| 4 (新閾値) | 0 | 0.10 | 0.08 | より早期に介入 |
| 5 | 1 | 0.13 | 0.13 | 同等 |
| 6 | 2 | 0.16 | 0.18 | やや積極的 |
| 7 | 3 | 0.19 | 0.23 | 大幅に変化しやすい |
| 8 (新hard cap) | 4 | 0.22 | **Hard Cap** | 強制変更 |

**理由**:
- 低いスタート値（0.08 vs 0.10）で初期スイッチを容易化
- 急勾配（0.05 vs 0.03）でスイッチ圧力を加速
- より早く決定点に到達

---

## Phase 2: アルゴリズム改善（より効果的、中リスク）

### 改善1: Stagnation-Aware Smoothing（停滞を考慮したスムージング）

**ファイル**: [backend/main.py](backend/main.py)（line 824の後に新関数を追加）

**問題**: 現在のスムージングは全ての外れ値を同等に扱い、短いコード変化を停滞に統合してしまう可能性がある。

**解決策**: スムージングロジックに停滞検出を追加

**新関数**:
```python
def smooth_chord_sequence_stagnation_aware(chords: list[str], passes: int = 2, max_run: int = 6) -> list[str]:
    """
    コードシーケンスをスムージングしつつ、長期停滞を防止する。

    ルール:
    1. 1バー外れ値: A-B-A -> A-A-A（既存）
    2. 2バー外れ値: A-B-B-A -> A-A-A-A（既存）
    3. 停滞防止: スムージングがmax_runバーを超える連続を作る場合、外れ値を保持

    Args:
        chords: 入力コードシーケンス
        passes: スムージングパス数
        max_run: 許可される最大連続バー数
    """
    if len(chords) < 3:
        return chords[:]

    smoothed = chords[:]

    for _ in range(passes):
        changed = False
        result = smoothed[:]

        # 1バー外れ値: A-B-A -> A-A-A
        for i in range(1, len(result) - 1):
            prev_c = result[i - 1]
            curr_c = result[i]
            next_c = result[i + 1]

            if prev_c == next_c and curr_c != prev_c:
                # このスムージングが過剰停滞を生むか確認
                run_before = 1
                j = i - 1
                while j > 0 and result[j-1] == prev_c:
                    run_before += 1
                    j -= 1

                run_after = 1
                j = i + 1
                while j < len(result) - 1 and result[j+1] == next_c:
                    run_after += 1
                    j += 1

                potential_run = run_before + 1 + run_after

                # 過剰停滞を作らない場合のみスムージング
                if potential_run <= max_run:
                    result[i] = prev_c
                    changed = True

        # 2バー外れ値処理（同様のロジック）
        for i in range(1, len(result) - 2):
            if (result[i - 1] == result[i + 2] and
                result[i] == result[i + 1] and
                result[i] != result[i - 1]):

                prev_c = result[i - 1]
                run_before = 1
                j = i - 1
                while j > 0 and result[j-1] == prev_c:
                    run_before += 1
                    j -= 1

                run_after = 1
                j = i + 2
                while j < len(result) - 1 and result[j+1] == prev_c:
                    run_after += 1
                    j += 1

                potential_run = run_before + 2 + run_after

                if potential_run <= max_run:
                    result[i] = result[i - 1]
                    result[i + 1] = result[i - 1]
                    changed = True

        smoothed = result
        if not changed:
            break

    return smoothed
```

**呼び出し場所の更新** ([backend/main.py:1667](backend/main.py#L1667)):
```python
# 現在
smoothed_chords = smooth_chord_sequence(raw_chords)

# 変更後
smoothed_chords = smooth_chord_sequence_stagnation_aware(raw_chords, passes=2, max_run=6)
```

---

### 改善2: 長期停滞ブレイク用のセカンダリパス

**ファイル**: [backend/main.py](backend/main.py)（line 1668の後に追加）

**新関数**:
```python
def break_long_stagnation_runs(chords: list[str], max_consecutive: int = 6) -> list[str]:
    """
    残存する長期停滞を分割する後処理パス。

    コードがmax_consecutiveバーを超えて継続する場合、
    周辺コンテキストから第2候補コードで分割を試みる。

    検出とスムージングの両方が停滞防止に失敗した場合のセーフティネット。
    """
    if len(chords) <= max_consecutive:
        return chords[:]

    result = chords[:]
    i = 0

    while i < len(result):
        # 連続実行をカウント
        j = i
        while j < len(result) and result[j] == result[i]:
            j += 1

        run_length = j - i

        if run_length > max_consecutive:
            # 長期実行を発見 - ブレイクを挿入
            # 戦略: max_consecutiveバーごとに1バーのバリエーションを挿入
            # 前後の異なるコードがあれば使用

            alt_chord = None
            if i > 0 and result[i-1] != result[i]:
                alt_chord = result[i-1]
            elif j < len(result) and result[j] != result[i]:
                alt_chord = result[j]

            if alt_chord:
                # 定期的な間隔でブレイクを挿入
                insert_positions = list(range(i + max_consecutive, j, max_consecutive + 1))
                # インデックスシフトを避けるため逆順で処理
                for pos in reversed(insert_positions):
                    result[pos] = alt_chord

        i = j

    return result

# スムージング後に呼び出し（line 1667の後に挿入）
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=6)
```

---

## Phase 3: 環境変数による設定（オプション、調整を容易化）

**ファイル**: [backend/main.py](backend/main.py)（トップレベル）

```python
import os

# ファイルトップまたは関数デフォルト値で定義
DEFAULT_MAX_REPEAT = int(os.getenv("STAGNATION_MAX_REPEAT", "4"))
DEFAULT_LONG_PENALTY = float(os.getenv("STAGNATION_LONG_PENALTY", "0.85"))

# detect_chords_matrix関数シグネチャで使用（line 512-516）
def detect_chords_matrix(
    # ... 他のパラメータ ...
    max_repeat_segments: int = DEFAULT_MAX_REPEAT,
    long_stag_penalty: float = DEFAULT_LONG_PENALTY,
    # ... 他のパラメータ ...
):
```

**使用例**:
```bash
# より積極的な設定でテスト
export STAGNATION_MAX_REPEAT=3
export STAGNATION_LONG_PENALTY=0.90
uvicorn main:app --reload

# 検証後の本番環境
export STAGNATION_MAX_REPEAT=4
export STAGNATION_LONG_PENALTY=0.85
```

---

## テスト戦略

### Test 1: 停滞上限のユニットテスト

**新規ファイル**: `backend/tests/test_stagnation_limits.py`

- Hard capが8バー以内を保証するテスト
- Progressive escalationの効果テスト
- 合成データを使用した単体テスト

### Test 2: 実音源によるリグレッションテスト

**新規ファイル**: `backend/tests/test_real_audio_stagnation.py`

- `ガソリン0812.m4a`（問題のあったファイル）を使用
- 最大停滞が8バー以内であることを確認
- コードバリエーションが妥当であることを確認

### Test 3: 既存テストの更新

**既存ファイル**: `backend/tests/verify_chord_accuracy.py`

- 既存の`test_audio_analysis()`に停滞チェックを追加:

```python
# Line 122の後に追加
max_run = 1
current_run = 1
for i in range(1, len(chords)):
    if chords[i] == chords[i-1]:
        current_run += 1
        max_run = max(max_run, current_run)
    else:
        current_run = 1

print(f"  Max consecutive bars: {max_run}")
assert max_run <= 8, f"Excessive stagnation detected: {max_run} bars"
```

---

## 推奨実装順序（更新版）

### ユーザーフィードバック反映
初期テスト結果から、保守的アプローチ（Iteration 1）では効果が不十分であることが判明。ユーザーの要望により、より積極的なアプローチを採用：**Iteration 2（積極的パラメータ）+ Phase 2（アルゴリズム改善）を同時実装**。

### 実装ステップ（一括適用）

#### Step 1: 積極的パラメータ調整（Iteration 2相当）
1. **max_repeat_segments = 4**（6 → 4に削減）
   - [backend/main.py:512](backend/main.py#L512)
2. **long_stag_penalty = 0.85**（0.60 → 0.85に強化）
   - [backend/main.py:515](backend/main.py#L515)
3. **Progressive escalation: `0.08 + 0.05 * excess`**（0.10 + 0.03 → 0.08 + 0.05）
   - [backend/main.py:639](backend/main.py#L639)

#### Step 2: アルゴリズム改善（Phase 2）
1. **Stagnation-aware smoothing関数を追加**
   - [backend/main.py](backend/main.py)（line 824の後に新関数）
   - `smooth_chord_sequence_stagnation_aware()` を実装
   - 呼び出し元を更新（[backend/main.py:1667](backend/main.py#L1667)）

2. **長期停滞ブレイク関数を追加**
   - [backend/main.py](backend/main.py)（line 1667の後に追加）
   - `break_long_stagnation_runs()` を実装
   - スムージング後に自動適用

#### Step 3: テスト実装
1. **ユニットテストファイル作成**
   - `backend/tests/test_stagnation_limits.py`（新規）
   - Hard cap検証テスト
   - Progressive escalation効果テスト

2. **既存テスト更新**
   - `backend/tests/verify_chord_accuracy.py`
   - 停滞アサーションを追加（line 122の後）

#### Step 4: 検証
1. バックエンド再起動（**重要**）
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
2. 同じ音源を再解析
3. 最大停滞が ≤ 8バー（目標: ≤ 6バー）であることを確認

---

## 検証結果（2026-03-13）

### 改善効果 ✅

**BEFORE**（analysis-1773408178704.json）:
- 最大停滞: **22バー**

**AFTER**（analysis-1773410840356.json）:
- 最大停滞: **10バー**（Fm7, bars 83-92）
- **改善率: 54%削減**

### 残存問題 ⚠️

**目標未達成**: 目標は ≤ 8バー（許容）/ ≤ 6バー（理想）

**6バー以上の停滞（10箇所検出）**:
1. **Fm7: 10バー**（bars 83-92）← 最悪
2. **Fm7: 9バー**（bars 32-40）
3. **G#maj7: 8バー**（bars 67-74）
4. G#maj7: 6バー（bars 24-29）
5. Fm7: 6バー（bars 43-48、55-60、75-80）
6. C#maj7: 6バー（bars 93-98、144-149）
7. G#maj7: 6バー（bars 123-128）

### 原因分析

`break_long_stagnation_runs(max_consecutive=6)` が期待通りに動作していない可能性：

1. **`alt_chord`が見つからないケース**:
   - 前後に異なるコードがない長い停滞の場合、`alt_chord = None`となり分割されない
   - 実装コード（line 941-945）では前後のコードのみ検索

2. **検証不足**:
   - 関数が実際に呼び出されているか
   - どの停滞が検出されたか
   - 代替コードが正しく選ばれているか

### 追加修正プラン

#### Phase 2.5: デバッグと強化（推奨）

**ステップ1: デバッグログ追加**

`break_long_stagnation_runs()`関数にログを追加（line 936の後）:

```python
if run_length > max_consecutive:
    print(f"[STAGNATION] Found long run: {result[i]} × {run_length} bars (position {i}-{j-1})")

    alt_chord = None
    if i > 0 and result[i-1] != result[i]:
        alt_chord = result[i-1]
    elif j < len(result) and result[j] != result[i]:
        alt_chord = result[j]

    print(f"[STAGNATION] Alternative chord: {alt_chord}")

    if alt_chord:
        insert_positions = list(range(i + max_consecutive, j, max_consecutive + 1))
        print(f"[STAGNATION] Inserting {alt_chord} at positions: {insert_positions}")
        for pos in reversed(insert_positions):
            result[pos] = alt_chord
    else:
        print(f"[STAGNATION] WARNING: No alternative chord found, skipping break")
```

**ステップ2: 代替コード生成の強化**

`alt_chord`が見つからない場合の対処（line 941-945を置き換え）:

```python
# まず前後の異なるコードを探す
alt_chord = None
if i > 0 and result[i-1] != result[i]:
    alt_chord = result[i-1]
elif j < len(result) and result[j] != result[i]:
    alt_chord = result[j]

# それでも見つからない場合、最も頻度の高い異なるコードを使用
if not alt_chord:
    from collections import Counter
    chord_counts = Counter(result)
    # 現在のコード以外で最も多いコードを選択
    for chord, _ in chord_counts.most_common():
        if chord != result[i]:
            alt_chord = chord
            break

if alt_chord:
    # ... 既存の挿入ロジック ...
```

**ステップ3: より積極的なブレイク**

`max_consecutive`を4に削減、または複数回実行:

```python
# Line 1803を変更
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=4)
```

または2回実行:

```python
# Line 1803を変更
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=6)
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=4)  # 2nd pass
```

### 推奨実装順序

1. **ステップ1のみ実装**: デバッグログを追加してバックエンド再起動 → 実際の動作を確認
2. **ステップ2または3を追加**: ログから原因が明確になった後、適切な対策を選択

---

## Phase 2.5: 代替コード生成の強化（ユーザー選択）

ユーザーの選択により、**Option 3: 代替コード生成を強化**を実装します。

### 実装詳細

**ファイル**: [backend/main.py](backend/main.py)

**変更箇所**: `break_long_stagnation_runs()`関数（lines 941-952）

#### 変更前（現在の実装）:

```python
alt_chord = None
if i > 0 and result[i-1] != result[i]:
    alt_chord = result[i-1]
elif j < len(result) and result[j] != result[i]:
    alt_chord = result[j]

if alt_chord:
    # Insert breaks at regular intervals
    insert_positions = list(range(i + max_consecutive, j, max_consecutive + 1))
    # Work backwards to avoid index shifting
    for pos in reversed(insert_positions):
        result[pos] = alt_chord
```

**問題点**: `alt_chord`が見つからない場合、分割が行われない

#### 変更後（強化版）:

```python
alt_chord = None

# Strategy 1: Use adjacent different chord
if i > 0 and result[i-1] != result[i]:
    alt_chord = result[i-1]
elif j < len(result) and result[j] != result[i]:
    alt_chord = result[j]

# Strategy 2: If no adjacent chord, use most frequent different chord
if not alt_chord:
    from collections import Counter
    chord_counts = Counter(result)
    # Find most common chord that's different from current
    for chord, count in chord_counts.most_common():
        if chord != result[i]:
            alt_chord = chord
            print(f"[STAGNATION] Using fallback chord: {alt_chord} (frequency: {count})")
            break

# Strategy 3: If still not found (entire sequence is same chord),
# use a diatonic substitute based on music theory
if not alt_chord:
    # This should rarely happen, but provides ultimate fallback
    # For now, we'll skip breaking if no alternative exists
    print(f"[STAGNATION] WARNING: Cannot break stagnation - no alternative chord available")

if alt_chord:
    # Insert breaks at regular intervals
    insert_positions = list(range(i + max_consecutive, j, max_consecutive + 1))
    print(f"[STAGNATION] Breaking {result[i]} run (length {run_length}) with {alt_chord} at positions: {insert_positions}")
    # Work backwards to avoid index shifting
    for pos in reversed(insert_positions):
        result[pos] = alt_chord
```

**追加機能**:
1. **Strategy 1（既存）**: 隣接する異なるコードを使用
2. **Strategy 2（新規）**: 頻度の高い異なるコードを使用（Counterで統計分析）
3. **Strategy 3（将来拡張）**: ダイアトニック理論に基づく代替コード生成（現時点では警告のみ）
4. **デバッグログ**: 実際の動作を追跡可能

### さらなる積極化（オプション）

より確実に8バー以内に抑えるため、`max_consecutive`を段階的に適用：

**変更箇所**: [backend/main.py:1803](backend/main.py#L1803)

```python
# 変更前
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=6)

# 変更後（2段階適用）
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=6)
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=4)
```

**効果**:
- 1回目: 6バーを超える停滞を分割（10バー → 6+4バー）
- 2回目: 4バーを超える停滞をさらに分割（6バー → 4+2バー、4バー → 4バー）
- 最終的な最大停滞: ≤ 4バー（理想的）

**リスク**:
- 過度なコード変化（フリッカー）の可能性
- `min_hold_segments=2`により、最低2バーは保持されるため、ある程度緩和される

### 検証計画

実装後、以下を確認：

1. **最大停滞**: ≤ 6バー（2段階適用の場合は ≤ 4バー）
2. **デバッグログ**: どの停滞が分割されたか
3. **フリッカーチェック**: 不自然な頻繁なコード変化がないか
4. **リグレッション**: 既存のテストがパスするか

### 実装ステップ

1. `break_long_stagnation_runs()`関数を強化版に置き換え（lines 941-952）
2. （オプション）2段階適用を追加（line 1803の後に1行追加）
3. バックエンド再起動
4. 同じ音源（`https://youtu.be/Pwht_zL3_go`）を再解析
5. 結果を比較：
   - 最大停滞: 10バー → ?バー
   - 6バー以上の停滞箇所: 10箇所 → ?箇所

### 期待される最終結果

| 指標 | 現在 | 目標（1段階） | 目標（2段階） |
|------|------|--------------|--------------|
| 最大停滞 | 10バー | ≤ 6バー | ≤ 4バー |
| 6バー以上の停滞箇所 | 10箇所 | 0箇所 | 0箇所 |
| フリッカー | なし | なし | 軽微な可能性 |

**推奨**: まず1段階（Strategy 2追加のみ）で試し、結果が不十分な場合のみ2段階適用を追加

---

## 実装サマリー（Phase 2.5）

### 変更ファイル

**[backend/main.py](backend/main.py)** - 1箇所のみ修正

### 変更内容

#### 1. `break_long_stagnation_runs()` 関数の強化（必須）

**場所**: Lines 941-952

**変更**: `alt_chord`が見つからない場合の代替ロジックを追加

```python
# 変更前: 隣接コードのみチェック
alt_chord = None
if i > 0 and result[i-1] != result[i]:
    alt_chord = result[i-1]
elif j < len(result) and result[j] != result[i]:
    alt_chord = result[j]

# 変更後: 隣接コード → 頻度分析 → フォールバック
alt_chord = None

# Strategy 1: Use adjacent different chord
if i > 0 and result[i-1] != result[i]:
    alt_chord = result[i-1]
elif j < len(result) and result[j] != result[i]:
    alt_chord = result[j]

# Strategy 2: If no adjacent chord, use most frequent different chord
if not alt_chord:
    from collections import Counter
    chord_counts = Counter(result)
    for chord, count in chord_counts.most_common():
        if chord != result[i]:
            alt_chord = chord
            print(f"[STAGNATION] Using fallback chord: {alt_chord} (frequency: {count})")
            break

# Strategy 3: Ultimate fallback (rare)
if not alt_chord:
    print(f"[STAGNATION] WARNING: Cannot break stagnation - no alternative chord available")

if alt_chord:
    insert_positions = list(range(i + max_consecutive, j, max_consecutive + 1))
    print(f"[STAGNATION] Breaking {result[i]} run (length {run_length}) with {alt_chord} at positions: {insert_positions}")
    for pos in reversed(insert_positions):
        result[pos] = alt_chord
```

**効果**: 前後に異なるコードがない長い停滞でも、楽曲内で最も頻度の高い別のコードで分割可能

#### 2. 2段階適用（オプション、結果に応じて追加）

**場所**: Line 1803の後に1行追加

```python
# 既存（line 1803）
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=6)

# 追加（line 1804）
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=4)
```

**効果**: より積極的に停滞を抑制（最大 ≤ 4バー）

**推奨アプローチ**:
1. まず1段階（Strategy 2のみ）で実装・検証
2. 結果が ≤ 6バーを達成できない場合のみ、2段階目を追加

### 検証方法

```bash
cd backend
uvicorn main:app --reload
```

同じ音源（`https://youtu.be/Pwht_zL3_go`）を再解析し、以下を確認：

1. **最大停滞**: 現在10バー → 目標 ≤ 6バー
2. **デバッグログ**: `[STAGNATION]`タグで分割動作を追跡
3. **フリッカー**: 不自然なコード変化がないか確認

### 成功基準

- ✅ 最大停滞 ≤ 6バー（許容）
- ✅ 最大停滞 ≤ 4バー（理想）
- ✅ 6バー以上の停滞箇所が0になる
- ✅ フリッカー（過度なコード変化）なし
- ✅ 既存テストがパス

---

## Phase 2.5検証結果とデバッグ（2026-03-15）

### 検証結果

**Status**: ❌ **Phase 2.5が効果なし**

| 指標 | Phase 2.5前 | Phase 2.5後 | 変化 |
|------|------------|------------|------|
| 最大停滞 | 10バー | 10バー | **変化なし** |
| 6バー以上停滞箇所 | 10箇所 | 10箇所 | **変化なし** |

**ユーザー報告**:
- ✅ バックエンド再起動済み
- ❌ `[STAGNATION]`デバッグログなし
- 新しい問題: 音楽的に異なる部分が同じコードと検出されている

### 原因診断

**実装確認**:
1. ✅ `break_long_stagnation_runs()`関数は正しく実装済み（lines 908-974）
2. ✅ 関数は正しく呼び出されている（line 1821）
3. ✅ シンタックスエラーなし
4. ❌ デバッグログが出力されていない

**原因の可能性**:
- ログ出力の問題（バッファリング、リダイレクト）
- 条件に到達していない（中間処理で停滞が解消されている？）
- 関数が途中で失敗している

### デバッグプラン

#### ステップ1: ファイルベースのデバッグログ追加

`print()`が見えない場合に備え、ファイル出力を追加：

**[backend/main.py:908](backend/main.py#L908)** - `break_long_stagnation_runs()`関数の開始直後:

```python
def break_long_stagnation_runs(chords: list[str], max_consecutive: int = 6) -> list[str]:
    """..."""
    import sys

    # デバッグログをファイルに出力
    with open('debug_stagnation.log', 'a', encoding='utf-8') as f:
        f.write(f"\n=== break_long_stagnation_runs called ===\n")
        f.write(f"Input: {len(chords)} chords, max_consecutive={max_consecutive}\n")
    sys.stdout.flush()

    print(f"[STAGNATION-DEBUG] Function called with {len(chords)} chords")
```

**line 936** - 長い停滞検出時:

```python
if run_length > max_consecutive:
    with open('debug_stagnation.log', 'a', encoding='utf-8') as f:
        f.write(f"LONG RUN: {result[i]} × {run_length} bars at position {i}-{j-1}\n")
```

#### ステップ2: 中間結果の追跡

**[backend/main.py:1815-1825](backend/main.py#L1815-L1825)** - 各処理段階の最大停滞を記録:

```python
def calc_max_run(chords):
    max_run = 1
    current_run = 1
    for i in range(1, len(chords)):
        if chords[i] == chords[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run

# Line 1815
print(f"[DEBUG] Raw chords max stagnation: {calc_max_run(raw_chords)} bars")

# Line 1818
smoothed_chords = smooth_chord_sequence_stagnation_aware(raw_chords, passes=2, max_run=6)
print(f"[DEBUG] After stagnation-aware smoothing: {calc_max_run(smoothed_chords)} bars")

# Line 1821
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=6)
print(f"[DEBUG] After break_long_stagnation_runs: {calc_max_run(smoothed_chords)} bars")
```

#### ステップ3: 2段階適用（より積極的）

デバッグ後も改善が見られない場合、より積極的なアプローチ：

**[backend/main.py:1821](backend/main.py#L1821)** - 2回実行:

```python
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=6)
smoothed_chords = break_long_stagnation_runs(smoothed_chords, max_consecutive=4)
```

### 新しい問題: 音楽的セクション区別

**問題**: 音楽的に異なる部分（Aメロ vs サビ）が同じコード（例: Fm7）と検出される

**原因**: 現在のアルゴリズムは純粋にクロマ特徴のみでコード検出し、セクション変化を考慮していない

**解決アプローチ（将来実装）**:
1. **エネルギー/ダイナミクス情報**: RMS energy, spectral centroidでセクション境界を検出
2. **セグメンテーション**: librosa.segment.agglomerative()でセクション分割
3. **テンポラルコントラスト**: 音色変化が大きい場所でコード変化を優先

**推奨**: まず停滞問題を完全解決してから、セクション区別に取り組む

### 次のアクション

1. **最優先**: デバッグログを追加してPhase 2.5が動作していない原因を特定
2. **次**: 原因が判明したら修正、または2段階適用
3. **その後**: 停滞解決後、音楽的セクション区別の改善

---

### 廃止された段階的アプローチ
~~Iteration 1（保守的）~~ → スキップ（効果不十分）
Iteration 2 + Phase 2 → **一括実装（推奨）**
~~Iteration 3~~ → 上記に統合
Iteration 4（本番強化） → 検証成功後に実施

---

## 期待される結果

### 定量的目標
- **最大停滞**: ≤ 6バー（目標）、≤ 8バー（許容）
- **コードバリエーション**: 楽曲あたり ≥ 25%のユニークコード
- **リグレッションなし**: 既存テストスイートがパス
- **フリッカーなし**: 最低2バーでコード変更（min_hold_segmentsで保持）

### 定性的目標
- より自然なコード進行
- 実際のハーモニック変化をより正確に追跡
- 「コードがスタックする」というユーザー苦情の削減
- 安定性の維持（過度なフリッカーなし）

---

## ロールバックと安全性

### ロールバックプラン

問題が発生した場合（フリッカー、不正確なコード）：

**クイックロールバック**（環境変数）:
```bash
export STAGNATION_MAX_REPEAT=6
export STAGNATION_LONG_PENALTY=0.60
```

**コードロールバック**（git）:
```bash
git revert <commit-hash>
```

**中間パラメータ調整**:
4バーが積極的すぎる場合：
- `max_repeat_segments = 5`を試す
- `long_stag_penalty = 0.75`（中間値）を試す

### 既知のリスク

| リスク | 可能性 | 影響 | 緩和策 |
|--------|-------|------|--------|
| コード間のフリッカー増加 | 中 | 高 | min_hold_segments=2で防止 |
| 持続セクションでの誤変更 | 中 | 中 | Progressive escalationで遅延 |
| 既存コードの破壊的変更 | 低 | 高 | 全パラメータオプショナル |
| パフォーマンス低下 | 低 | 低 | 新関数はO(n)、最小オーバーヘッド |

---

## 重要ファイル一覧

実装に必要な主要ファイル：

1. **[backend/main.py](backend/main.py)** - コアロジック修正:
   - `detect_chords_matrix()` パラメータ（lines 510-516）
   - Progressive escalation formula（lines 630-640）
   - Smoothing functions（lines 786-824）
   - 変更の90%がこのファイル

2. **[backend/tests/verify_chord_accuracy.py](backend/tests/verify_chord_accuracy.py)** - 既存テスト更新:
   - 停滞アサーションを追加
   - 既存機能の破壊なしを検証

3. **backend/tests/test_stagnation_limits.py** - 新規ユニットテスト:
   - 合成データでの停滞防止テスト
   - 実音源テスト前の分離テスト

4. **[analysis-1773408178704.json](analysis-1773408178704.json)** - 参照ファイル:
   - 現在の問題動作を示す
   - Before/After比較用

5. **ガソリン0812.m4a** - オリジナル音源:
   - エンドツーエンド検証用
   - 実際の問題を解決できるか確認

---

## Verification（検証方法）

実装完了後、以下の手順で検証します：

1. **ユニットテスト実行**:
   ```bash
   cd backend
   python tests/test_stagnation_limits.py
   ```

2. **既存テストの実行**:
   ```bash
   python tests/verify_chord_accuracy.py
   ```

3. **実音源での再解析**:
   ```bash
   # バックエンド起動
   uvicorn main:app --reload

   # フロントエンドから同じYouTube URLを再解析
   # または curl コマンドで直接テスト
   ```

4. **結果の比較**:
   - 最大停滞バー数を確認（目標: ≤ 8バー、理想: ≤ 6バー）
   - コード進行の自然さを確認
   - フリッカー（過度な変化）がないか確認

5. **他の楽曲でのテスト**:
   - 異なるジャンル、BPM、キーの楽曲で検証
   - 副作用がないことを確認
