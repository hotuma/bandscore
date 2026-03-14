# コード検出精度検証ツールの実装計画

## Context（背景）

YouTubeから解析した楽曲（https://youtu.be/Pwht_zL3_go）の解析結果（analysis-1773516890583.json）について、コード検出の精度を検証するツールを実装します。

### 解析結果の概要
- **キー**: Fm（F minor）
- **BPM**: 134
- **総バー数**: 159
- **検出コード**: Fm7, C#maj7, A#m7, G#maj7, Fm, A#m, Csus4, Cm7, F#7, Bmaj7（10種類）

### ユーザー要件
- **正解データ**: なし（手動確認済みの正解コード進行は存在しない）
- **検証方法**: 定量メトリクス計算、音声との同期確認、視覚的な分析
- **出力形式**: コンソール出力

### 既存システム
BandScoreプロジェクトには既存の検証スクリプト（`backend/tests/verify_chord_accuracy.py`）が存在し、以下の3つのテストを実施しています：
1. **TEST 1**: Template Expansion（72個のコードテンプレート生成確認）
2. **TEST 2**: Diatonic Chord List Expansion（ダイアトニックコードの検証）
3. **TEST 3**: Full Audio Analysis（実際の音声ファイルの解析）

コード検出アルゴリズムは以下の3層構造：
- **テンプレートマッチング**: 72個のコードテンプレート（12根音 × 6コード型）とのコサイン類似度計算
- **ダイアトニックペナルティ**: キー外のコードにペナルティを適用（0.2-0.3）
- **スタグネーション防止**: 同一コードの連続を最大6バーに制限

---

## Implementation Plan（実装計画）

### Phase 1: Core Infrastructure（基盤機能）

#### 1.1 JSON読み込み機能
**ファイル**: [backend/tests/verify_chord_accuracy.py](backend/tests/verify_chord_accuracy.py)

```python
def load_json_analysis(json_path: str) -> dict:
    """JSON解析結果を読み込んで検証する"""
```

**実装内容**:
- `json.load()`でファイルを読み込み
- 必須キーの存在確認: `bpm`, `key`, `bars`, `duration_sec`
- bars配列の各要素が`chord`, `start_sec`, `end_sec`を持つことを確認
- エラー処理: FileNotFoundError, JSONDecodeError, ValueError

#### 1.2 キー解析機能

```python
def parse_key(key_str: str) -> tuple[str, str]:
    """キー文字列を(root, mode)に分解"""
```

**実装内容**:
- "Fm" → ("F", "m")
- "C" → ("C", "")
- "G#" → ("G#", "")
- シャープ記号（C#, D#, F#, G#, A#）に対応

---

### Phase 2: Metrics Computation（メトリクス計算）

#### 2.1 スタグネーション分析

```python
def compute_stagnation_metrics(chords: list[str]) -> dict:
    """コードのスタグネーション（同一コードの連続）を分析"""
```

**出力メトリクス**:
- `max_consecutive`: 最長連続数
- `avg_consecutive`: 平均連続数
- `stagnation_runs`: 連続区間のリスト `[(chord, count, start_bar)]`
- `exceeds_threshold`: 6バー超過の有無
- `total_runs`: 総連続区間数

**閾値**: 6バー（[main.py:908](backend/main.py#L908)の`break_long_stagnation_runs`のデフォルト値）

#### 2.2 ダイアトニック準拠分析

```python
def compute_diatonic_metrics(chords: list[str], key_root: str, key_mode: str) -> dict:
    """ダイアトニックコードの使用率を計算"""
```

**実装内容**:
- [main.py:429](backend/main.py#L429)の`get_diatonic_chords_for_key()`を使用
- Fmキーのダイアトニックコード: Fm, Fm7, Fsus4, Gm, Gm7, Gsus4, G#, G#maj7, G#sus4, A#m, A#m7, A#sus4, Cm, Cm7, Csus4, D#, D#7, D#sus4
- ダイアトニックバー数 / 総バー数 で準拠率を計算
- 期待値: >80%

**出力メトリクス**:
- `diatonic_rate`: 準拠率（0.0-1.0）
- `diatonic_chords`: 使用されたダイアトニックコード
- `non_diatonic_chords`: 使用された非ダイアトニックコード
- `expected_diatonic`: キーに対する期待されるダイアトニックコード全リスト

#### 2.3 コード多様性分析

```python
def compute_diversity_metrics(chords: list[str]) -> dict:
    """コードの多様性を統計的に分析"""
```

**出力メトリクス**:
- `unique_count`: ユニークコード数
- `distribution`: コードごとの出現回数 `{chord: count}`
- `simpson_index`: シンプソン多様性指数（0.0-1.0、高いほど多様）
- `entropy`: シャノンエントロピー（bits）
- `most_common`: 頻出上位5コード

**計算式**:
- Simpson Index: `D = 1 - Σ(ni/N)²`
- Shannon Entropy: `H = -Σ(pi * log2(pi))`

#### 2.4 タイミング一貫性分析

```python
def compute_timing_metrics(bars: list[dict], bpm: float) -> dict:
    """バータイミングの一貫性を検証"""
```

**実装内容**:
- 期待バー長: `(60/BPM) * 4` = 1.791秒（134 BPM時）
- 各バーの実際の長さ: `end_sec - start_sec`
- 統計値: 平均、標準偏差、最小、最大
- 異常値検出: 期待値から0.1秒以上ずれているバー
- 判定基準: 標準偏差 <0.05秒（[verify_bar_timing.py:55-159](backend/tests/verify_bar_timing.py#L55-L159)参考）

**注意事項**:
- チャンク境界のバー（0.3-0.9秒程度）は異常として検出されるが、これは正常な動作
- bars 35, 52, 69, 86, 103, 120, 137, 154あたりがチャンク境界

**出力メトリクス**:
- `expected_duration`: 期待バー長
- `mean_duration`, `std_duration`, `min_duration`, `max_duration`
- `anomalies`: 異常バーのリスト `[(bar_idx, duration, deviation)]`
- `timing_ok`: 標準偏差が閾値内かどうか

---

### Phase 3: Visualization（視覚化）

#### 3.1 コード進行タイムライン

```python
def visualize_chord_progression(bars: list[dict], width: int = 120) -> str:
    """コード進行のASCIIタイムラインを生成"""
```

**出力形式**:
```
Bar 1-10:   [Fm7 ][Fm7 ][C#  ][Fm7 ][Fm7 ][Fm7 ][Fm7 ][C#  ][C#  ][C#  ]
Bar 11-20:  [C#  ][A#m7][A#m7][C#  ][C#  ][C#  ][C#  ][Fm  ][Fm  ][C#  ]
```

- 1行に10バーを表示
- 各コードを5文字の固定幅セル `[Fm7 ]` で表示

#### 3.2 スタグネーションヒートマップ

```python
def visualize_stagnation_heatmap(chords: list[str], width: int = 80) -> str:
    """スタグネーション強度のASCIIヒートマップを生成"""
```

**出力形式**:
```
Bar 1-80:   ██░░░░░░░░░░░░░░░░░░░░░░░░██████████░░░░░░...
Legend: █ = 7+ bars, ▓ = 4-6 bars, ░ = 1-3 bars
```

- 各バーの連続数に応じてシンボルを変更
- 1-3バー: `░`, 4-6バー: `▓`, 7バー以上: `█`

#### 3.3 ダイアトニックマーキング

```python
def visualize_diatonic_marking(bars: list[dict], diatonic_chords: list[str], width: int = 80) -> str:
    """ダイアトニック/非ダイアトニックコードをマーキング"""
```

**出力形式**:
```
Bar 1-80:   ○○○○○○○○○○○●●○○○○○○○...
Legend: ○ = diatonic (87.4%), ● = non-diatonic (12.6%)
```

- ダイアトニック: `○`, 非ダイアトニック: `●`

---

### Phase 4: Main Test Function（メインテスト関数）

#### 4.1 test_json_analysis() 関数

```python
def test_json_analysis(json_path: str = None):
    """JSON解析結果からコード検出精度を検証"""
```

**処理フロー**:
1. JSON読み込み（パス未指定時は自動検出）
2. メタデータ抽出とキー解析
3. 全メトリクス計算（スタグネーション、ダイアトニック、多様性、タイミング）
4. 視覚化生成
5. 総合レポート出力
6. 品質評価（PASS/WARNING/FAIL）

**出力レポート形式**:
```
============================================================
TEST 4: JSON Analysis Verification
============================================================
File: analysis-1773516890583.json
BPM: 134
Key: Fm (F minor)
Duration: 278.3s
Total Bars: 159

--- Stagnation Analysis ---
Max consecutive: 10 bars
Average consecutive: 3.2 bars
Total runs: 49
Exceeds threshold (>6 bars): YES - 3 occurrences
  Run 1: Fm7 for 10 bars starting at bar 87

--- Diatonic Compliance ---
Expected diatonic chords: [Fm, Fm7, Fsus4, Gm, Gm7, ...]
Diatonic rate: 87.4% (139/159 bars)
Diatonic chords used: Fm7, C#maj7, A#m7, G#maj7, Fm, A#m, Csus4, Cm7
Non-diatonic chords: F#7, Bmaj7

--- Chord Diversity ---
Unique chords: 10
Simpson diversity index: 0.68
Shannon entropy: 2.31 bits
Top 5 chords:
  1. Fm7: 71 bars (44.7%)
  2. C#maj7: 33 bars (20.8%)

--- Timing Consistency ---
Expected bar duration (134 BPM): 1.791s
Mean duration: 1.789s
Std deviation: 0.012s
Anomalies: 8 bars (chunk boundaries)
Timing OK: YES (std < 0.05s)

--- Visualizations ---
[ASCII charts here]

--- Overall Assessment ---
✓ PASS: Diatonic compliance > 80%
✗ WARNING: Max stagnation (10 bars) exceeds threshold (6 bars)
✓ PASS: Chord diversity acceptable (10 unique chords)
✓ PASS: Timing consistency OK

Overall: PASS with WARNINGS
```

#### 4.2 __main__ ブロック更新

```python
if __name__ == "__main__":
    import sys

    # 既存テスト実行
    test_templates()
    test_diatonic_expansion()
    test_audio_analysis()

    # 新規JSONテスト実行
    if len(sys.argv) > 1:
        test_json_analysis(sys.argv[1])
    else:
        # 自動検出
        json_files = [f for f in os.listdir(project_root)
                      if f.startswith('analysis-') and f.endswith('.json')]
        if json_files:
            test_json_analysis(os.path.join(project_root, json_files[0]))
```

---

## Critical Files（重要ファイル）

### 実装対象
- [backend/tests/verify_chord_accuracy.py](backend/tests/verify_chord_accuracy.py) - すべての新機能をここに追加

### 参照ファイル
- [backend/main.py:429-466](backend/main.py#L429-L466) - `get_diatonic_chords_for_key()` 関数
- [backend/main.py:908-937](backend/main.py#L908-L937) - `break_long_stagnation_runs()` とスタグネーション閾値
- [backend/tests/verify_bar_timing.py:55-159](backend/tests/verify_bar_timing.py#L55-L159) - タイミング検証のパターン
- [analysis-1773516890583.json](analysis-1773516890583.json) - 検証対象データ

---

## Quality Thresholds（品質閾値）

| メトリクス | 閾値 | 根拠 |
|----------|------|------|
| スタグネーション | ≤6バー | main.py:908の`max_consecutive=6` |
| ダイアトニック準拠率 | >80% | トーナル音楽の一般的な期待値 |
| タイミング標準偏差 | <0.05秒 | verify_bar_timing.pyの許容範囲 |
| コード多様性 | 5-15種類 | ポップ/ロック楽曲の一般的範囲 |

---

## Verification（検証方法）

### ユニットテスト
各関数を個別にテスト：
- `parse_key()`: "Fm", "C", "G#", "A#m"
- `compute_stagnation_metrics()`: 既知のコードシーケンス
- `compute_diatonic_metrics()`: 既知のダイアトニック/非ダイアトニック混合
- `compute_diversity_metrics()`: 均一vs多様なシーケンス
- `compute_timing_metrics()`: 一貫性vs異常値

### 統合テスト
1. `analysis-1773516890583.json`で実行
2. コンソール出力が期待形式と一致するか確認
3. 全メトリクスが正しく計算されているか検証
4. 視覚化が正しくレンダリングされるか確認

### エッジケース
- 空のJSONファイル
- 1バーだけのJSON
- 全て同じコード（100%スタグネーション）
- タイミングデータ欠損
- 無効なキー表記

---

## Implementation Checklist（実装チェックリスト）

### Phase 1: Core Infrastructure
- [ ] `load_json_analysis()` 実装
- [ ] `parse_key()` 実装
- [ ] ファイルI/Oエラーハンドリング追加

### Phase 2: Metrics
- [ ] `compute_stagnation_metrics()` 実装
- [ ] `compute_diatonic_metrics()` 実装
- [ ] `compute_diversity_metrics()` 実装
- [ ] `compute_timing_metrics()` 実装

### Phase 3: Visualization
- [ ] `visualize_chord_progression()` 実装
- [ ] `visualize_stagnation_heatmap()` 実装
- [ ] `visualize_diatonic_marking()` 実装

### Phase 4: Integration
- [ ] `test_json_analysis()` 実装
- [ ] `__main__` ブロック更新
- [ ] コマンドライン引数サポート追加

### Phase 5: Testing
- [ ] analysis-1773516890583.json でテスト実行
- [ ] コンソール出力フォーマット検証
- [ ] 全メトリクス値検証
- [ ] エッジケーステスト

### Phase 6: Documentation
- [ ] 全関数にdocstring追加
- [ ] 複雑なロジックにインラインコメント追加
- [ ] ファイルヘッダーにテスト説明追加

---

## Execution（実行方法）

```bash
cd backend

# 全テスト実行（自動検出）
python tests/verify_chord_accuracy.py

# 特定JSONファイルを指定
python tests/verify_chord_accuracy.py ../analysis-1773516890583.json
```

---

## Expected Results（期待される結果）

### analysis-1773516890583.json の予測結果

| メトリクス | 予測値 | 判定 |
|----------|--------|------|
| 最大スタグネーション | 10バー | WARNING（6バー超過） |
| ダイアトニック準拠率 | 87.4% | PASS（>80%） |
| ユニークコード数 | 10種類 | PASS（5-15の範囲内） |
| タイミング標準偏差 | 0.012秒 | PASS（<0.05秒） |

**非ダイアトニックコードの説明**:
- F#7, Bmaj7 はバー157-159（曲の終わり）に出現
- 意図的な転調や借用和音と考えられる
- 音楽的には一般的な手法（感情的効果のための半音階的和声）

**総合評価**: PASS with WARNINGS
- 全体的に高品質な解析結果
- スタグネーション超過は検出ロジックの改善余地を示唆
- ダイアトニック準拠率、多様性、タイミングは全て良好

---

## Dependencies（依存関係）

すべて標準ライブラリを使用（新規依存なし）：
- `json` - JSON読み込み
- `sys`, `os` - ファイル操作、コマンドライン引数
- `collections.Counter` - コード頻度カウント
- `math` - log計算（エントロピー）
- `itertools.groupby` - 連続区間検出

main.pyからのインポート：
- `get_diatonic_chords_for_key`

---

## Notes（注意事項）

### なぜverify_chord_accuracy.pyを拡張するのか
- 既存のテスト構造との一貫性維持
- 既存のインポートとセットアップを再利用
- 全コード精度テストの単一エントリポイント
- プロジェクトの規約に従う（verify_modes.py, verify_preview_content.pyパターン）

### なぜASCII視覚化か
- 新規依存なし（matplotlibなど不要）
- あらゆるターミナル環境で動作
- 高速実行
- ユーザー要求のコンソール出力に適合

### Ground Truth なしでの検証
- 正解データが利用不可能
- 内部一貫性と音楽理論的妥当性に焦点
- メトリクスはアルゴリズム動作を検証、絶対的精度ではない
- 既存テスト哲学に一致（verify_bar_timing.pyもタイミング一貫性を検証、正しさではない）

### タイミング異常の扱い
- 短いバー（<0.5秒）はチャンク境界で予想される
- これらはlibrosa解析の再起動によるもの
- レポートはするがテスト失敗にはしない
- 予期しない異常のみを懸念事項とする
