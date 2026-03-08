# Phase 4.2: オクターブ検証アルゴリズムの根本改善

## Context

Phase 4/4.1で実装したオクターブ検証（`verify_tempo_octave`）が、「いんちき和尚のブルース」で正しく動作していない。

**問題**: BPM 186.2（倍速）が検出され、オクターブ検証が93.1 BPM（正解）に補正できていない。

**ログ証拠**:
```
[OctaveVerify] Candidates: [('93.1', '×0.5', '0.242'), ('186.2', '×1.0', '0.299')]
[OctaveVerify] Selected: 186.2 BPM (factor=×1.0, score=0.299)
```

**スコア内訳（逆算）**:
| BPM | downbeat+regularity (×0.8) | prior (×0.2) | Total |
|-----|---------------------------|-------------|-------|
| 93.1 | ≈0.102 | 0.14 | **0.242** |
| 186.2 | ≈0.199 | 0.10 | **0.299** |

**根本原因**: `evaluate_downbeat_alignment`と`evaluate_onset_regularity`がオクターブ（×1 vs ×2）を区別できない構造的な問題。

---

## 修正対象

`backend/main.py` の以下の関数（lines 663-839）:
- `evaluate_downbeat_alignment()` → **削除して新関数に置換**
- `evaluate_onset_regularity()` → **削除して新関数に置換**
- `verify_tempo_octave()` → **スコアリングを書き換え**
- `evaluate_tempo_prior()` → **そのまま維持**

### 再利用する既存関数
- `highpass_filter()` (line 208)
- `lowpass_filter()` (line 213)
- `librosa.autocorrelate()` パターン (line 1041付近)

---

## 新しいスコアリング方式

### 方針: 自己相関（Autocorrelation）ベースの評価

従来のオンセットマッチング方式をやめ、**各候補BPMのラグ位置での自己相関値**を主要シグナルとする。

自己相関が優れている理由:
- ビートグリッドの密度に影響されない（従来の問題を解消）
- 基本周期 vs 倍音周期を自然に区別できる
- 特に低音域では、バスドラムが主拍（真のBPMの周期）に集中するため、基本周期のACが倍速のACより強くなる

### Signal 1: 低音域AC（`evaluate_bass_ac`）— 重み 0.35

```python
def evaluate_bass_ac(y: np.ndarray, sr: int, candidate_bpm: float,
                     hop_length: int = 512) -> float:
    """
    低音域(20-200Hz)のオンセットエンベロープの自己相関値で
    候補BPMの周期性の強さを評価。
    バスドラム/ベースは主拍に集中するため、真のテンポで高スコア。
    """
    try:
        y_bass = lowpass_filter(y, sr, cutoff_hz=200)
        y_bass = highpass_filter(y_bass, sr, cutoff_hz=20)
        bass_env = librosa.onset.onset_strength(y=y_bass, sr=sr,
                                                 hop_length=hop_length)
        del y_bass

        ac = librosa.autocorrelate(bass_env)
        if ac[0] > 0:
            ac = ac / ac[0]

        lag = 60.0 * sr / (candidate_bpm * hop_length)
        lag_int = int(round(lag))

        if 0 < lag_int < len(ac):
            # 放物線補間でサブフレーム精度
            if 0 < lag_int < len(ac) - 1:
                alpha = float(ac[lag_int - 1])
                beta = float(ac[lag_int])
                gamma = float(ac[lag_int + 1])
                return max(0.0, (alpha + beta + gamma) / 3.0)
            return max(0.0, float(ac[lag_int]))
        return 0.0
    except Exception as e:
        print(f"[WARNING] evaluate_bass_ac failed: {e}")
        return 0.0
```

**期待される効果**:
- いんちき和尚(93BPM): バスドラムが0.644s間隔 → lag(93)で強いAC
- いんちき和尚(186BPM): 0.322s間隔にはバスイベントなし → lag(186)で弱いAC
- tax(183.5BPM): バスドラムが0.327s間隔 → lag(183.5)で強いAC

### Signal 2: 全帯域AC（`evaluate_fullband_ac`）— 重み 0.25

```python
def evaluate_fullband_ac(onset_env: np.ndarray, sr: int,
                         candidate_bpm: float,
                         hop_length: int = 512) -> float:
    """
    全帯域オンセットエンベロープの自己相関値で
    候補BPMの全体的な周期性を評価。
    """
    ac = librosa.autocorrelate(onset_env)
    if ac[0] > 0:
        ac = ac / ac[0]

    lag = 60.0 * sr / (candidate_bpm * hop_length)
    lag_int = int(round(lag))

    if 0 < lag_int < len(ac):
        if 0 < lag_int < len(ac) - 1:
            alpha = float(ac[lag_int - 1])
            beta = float(ac[lag_int])
            gamma = float(ac[lag_int + 1])
            return max(0.0, (alpha + beta + gamma) / 3.0)
        return max(0.0, float(ac[lag_int]))
    return 0.0
```

### Signal 3: ビート位相エネルギー集中度（`evaluate_phase_concentration`）— 重み 0.20

```python
def evaluate_phase_concentration(onset_env: np.ndarray, sr: int,
                                  candidate_bpm: float,
                                  hop_length: int = 512,
                                  n_bins: int = 8) -> float:
    """
    オンセットエンベロープを候補BPMの周期で折りたたみ、
    エネルギー分布の「尖度」を評価。
    正しいテンポ → ダウンビートにエネルギー集中（高スコア）
    倍速テンポ → エネルギーが均等分布（低スコア）
    """
    beat_interval = 60.0 / candidate_bpm

    # 各フレームのビート位相を計算
    times = librosa.frames_to_time(np.arange(len(onset_env)),
                                    sr=sr, hop_length=hop_length)
    phases = (times % beat_interval) / beat_interval

    # ビンごとにエネルギーを集約
    bins = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i, phase in enumerate(phases):
        bin_idx = min(int(phase * n_bins), n_bins - 1)
        bins[bin_idx] += onset_env[i]
        counts[bin_idx] += 1

    avg_bins = bins / np.maximum(counts, 1)

    # 変動係数（CV）で尖度を評価
    mean_val = np.mean(avg_bins)
    if mean_val > 0:
        cv = np.std(avg_bins) / mean_val
        return min(cv, 1.0)
    return 0.0
```

### Signal 4: テンポ事前分布（`evaluate_tempo_prior`）— 重み 0.20

既存の実装をそのまま使用（line 663-685）。

### 統合: `verify_tempo_octave` 書き換え

```python
def verify_tempo_octave(y, sr, detected_bpm, onset_env):
    if detected_bpm < 40 or detected_bpm > 240:
        return detected_bpm, 1.0

    candidates = [
        (detected_bpm * 0.5, 0.5),
        (detected_bpm * 1.0, 1.0),
        (detected_bpm * 2.0, 2.0),
    ]
    candidates = [(bpm, f) for bpm, f in candidates if 40 <= bpm <= 240]

    scores = []
    for candidate_bpm, factor in candidates:
        bass_ac = evaluate_bass_ac(y, sr, candidate_bpm)
        full_ac = evaluate_fullband_ac(onset_env, sr, candidate_bpm)
        phase_conc = evaluate_phase_concentration(onset_env, sr, candidate_bpm)
        prior = evaluate_tempo_prior(candidate_bpm)

        total = (bass_ac * 0.35 +
                 full_ac * 0.25 +
                 phase_conc * 0.20 +
                 prior * 0.20)

        scores.append((candidate_bpm, factor, total))
        print(f"[OctaveVerify] {candidate_bpm:.1f} BPM (×{factor}): "
              f"bass_ac={bass_ac:.3f}, full_ac={full_ac:.3f}, "
              f"phase={phase_conc:.3f}, prior={prior:.1f}, total={total:.3f}")

    # 現状維持バイアス: 検出BPMを変更するには明確な証拠が必要
    # ACでは周期Tのシグナルが2T(半分速)でも強いAC値を示すため、
    # ×1.0候補に小さなボーナスを付与して誤補正を防ぐ
    STATUS_QUO_BONUS = 0.03
    for i, (bpm, factor, score) in enumerate(scores):
        if factor == 1.0:
            scores[i] = (bpm, factor, score + STATUS_QUO_BONUS)

    best_bpm, best_factor, best_score = max(scores, key=lambda x: x[2])
    print(f"[OctaveVerify] Selected: {best_bpm:.1f} BPM "
          f"(factor=×{best_factor}, score={best_score:.3f})")

    return best_bpm, best_factor
```

---

## 実装手順

### Step 1: 古い関数を削除
- `evaluate_downbeat_alignment()` (lines 734-786) → 削除
- `evaluate_onset_regularity()` (lines 688-731) → 削除

### Step 2: 新しい関数を追加（line 687付近）
- `evaluate_bass_ac()`
- `evaluate_fullband_ac()`
- `evaluate_phase_concentration()`

### Step 3: `verify_tempo_octave()` を書き換え
- 新しいスコアリング方式に変更
- 詳細な診断ログを追加

### Step 4: ビルドタグ更新
```python
"_build": "build-v5.2-octave-ac"
```

---

## Verification

### テスト1: いんちき和尚のブルース
- **期待**: BPM 186.2 → **93.1** (factor=0.5)
- **確認ポイント**: `bass_ac`スコアが93.1で明確に高い

### テスト2: tax.mp3
- **期待**: BPM 183.5 → **183.5** (factor=1.0、変更なし)
- **確認ポイント**: `bass_ac`スコアが183.5で最も高いか、少なくとも91.75より高い

### テスト3: バックエンドログの確認
```
[OctaveVerify] 93.1 BPM (×0.5): bass_ac=0.xxx, full_ac=0.xxx, phase=0.xxx, prior=0.7, total=0.xxx
[OctaveVerify] 186.2 BPM (×1.0): bass_ac=0.xxx, full_ac=0.xxx, phase=0.xxx, prior=0.5, total=0.xxx
[OctaveVerify] Selected: 93.1 BPM (factor=×0.5, score=0.xxx)
```

### テスト4: フロントエンドでの再生確認
- いんちき和尚: バー持続時間が約2.578sになり、ハイライトが曲と同期
- tax: バー持続時間が1.308sのまま、同期が維持される

---

## リスクと緩和策

## 実装経過

### build-v5.2 (AC方式 + STATUS_QUO_BONUS)
- **結果**: 両曲とも不正解。AC値がlag(T)とlag(2T)でほぼ同一（差<0.03）のため、ACベースの区別が不可能。
- STATUS_QUO_BONUS=0.03がいんちき和尚を0.001差で逆転。taxはpriorの影響で91.8に誤補正。

### build-v5.3 (位相ゲート方式) ← 現在の実装
- **方式**: `evaluate_phase_concentration`の値で半分速候補をゲート
  - `phase_half < 0.25` → 半分速を拒否（リズム構造なし）
  - `phase_half >= 0.25` → スコア比較で決定
- **結果**:
  - tax: phase(91.8)=0.146 < 0.25 → ゲート拒否 → **183.5 BPM ✓**
  - いんちき和尚: phase(93.1)=0.503 ≥ 0.25 → スコア比較 → **93.1 BPM ✓**
  - いんちき和尚の位相: 385.5ms (precision=0.9247)、「小節頭に若干合っていない」

---

## Phase 5: 小節頭位相リファインメント (build-v5.4-bar-phase)

### 問題

BPM検出とオクターブ補正は正しく動作するが、**小節頭がダウンビートに合わない**。

- tax: phase=220.6ms (precision=0.6175) → 小節頭でハイライトしない
- いんちき和尚: phase=385.5ms (precision=0.9247) → 小節頭に若干合っていない

### 根本原因

Stage 3の位相検出は**ビートグリッド**（0.327s間隔@183.5BPM）の最適オフセットを探すが、
そのビートが小節内の**何拍目**（beat 1,2,3,4）かを考慮しない。
キック（beat 1,3）とスネア（beat 2,4）を区別しないため、スネア位置でロックされる可能性がある。

### 解決策: `refine_bar_phase()` 関数

Stage 3の後に「Stage 3.5」を追加。検出されたビート位相から4つの小節開始候補（0,1,2,3ビートシフト）を
テストし、**低音域（バスドラム）のエネルギーが最も強い位置**を小節頭として選択。

### 修正対象ファイル

- `backend/main.py`
  - ~line 831: `refine_bar_phase()` 関数を追加
  - ~line 1125: Stage 3.5の呼び出しを挿入
  - line 1475: ビルドタグ更新

### アルゴリズム

```python
def refine_bar_phase(y, sr, bpm, phase_offset_sec, hop_length=512,
                     beats_per_bar=4, window_frames=2):
    beat_duration = 60.0 / bpm

    # 低音域オンセットエンベロープを計算
    y_bass = lowpass_filter(y, sr, cutoff_hz=200)
    y_bass = highpass_filter(y_bass, sr, cutoff_hz=20)
    bass_env = librosa.onset.onset_strength(y=y_bass, sr=sr, hop_length=hop_length)
    del y_bass

    total_frames = len(bass_env)
    total_duration_sec = total_frames * hop_length / sr

    if np.max(bass_env) < 1e-6:
        return phase_offset_sec, 0  # バスなし → 変更なし

    # 4つの候補をテスト（shift 0,1,2,3ビート）
    scores = []
    for shift in range(beats_per_bar):
        candidate_phase = phase_offset_sec + shift * beat_duration
        bar_starts = np.arange(candidate_phase, total_duration_sec,
                               beat_duration * beats_per_bar)
        bar_frames = (bar_starts * sr / hop_length).astype(int)

        energy = 0.0
        count = 0
        for f in bar_frames:
            lo = max(0, f - window_frames)
            hi = min(total_frames, f + window_frames + 1)
            if lo < hi:
                energy += np.max(bass_env[lo:hi])
                count += 1

        avg = energy / max(count, 1)
        scores.append(avg)

    best_shift = int(np.argmax(scores))

    # 5%未満の改善なら変更しない（ステータスクオ保護）
    if best_shift != 0:
        improvement = (scores[best_shift] - scores[0]) / (scores[0] + 1e-8)
        if improvement < 0.05:
            best_shift = 0

    return phase_offset_sec + best_shift * beat_duration, best_shift
```

### 統合箇所

```python
# Stage 3.5（Stage 3の直後、chroma計算の前）
if forced_phase is None:
    refined_phase, bar_shift = refine_bar_phase(y, sr, bpm, phase_offset_sec)
    if bar_shift != 0:
        print(f"[BarPhaseRefine] shift={bar_shift} beats: "
              f"{phase_offset_sec*1000:.1f}ms -> {refined_phase*1000:.1f}ms")
    phase_offset_sec = refined_phase
# forced_phaseの場合は既にリファインメント済み
```

### 期待される動作

**tax (183.5 BPM)**:
- beat_duration = 0.327s
- 4候補: 220.6ms, 547.5ms, 874.3ms, 1201.2ms
- キックドラムが最も強い位置が選ばれる → 小節頭がダウンビートに一致

**いんちき和尚 (93.1 BPM)**:
- beat_duration = 0.644s
- 4候補: 385.5ms, 1030.0ms, 1674.4ms, 2319.0ms
- 同様にバスエネルギーでリファイン

### メモリ/パフォーマンス

- 一時メモリ: ~5MB（y_bassのコピー）→ 即座にdel
- 追加計算時間: ~100-300ms（60sチャンク）
- 既存の`evaluate_bass_ac()`と同じパターン

### build-v5.4 テスト結果

**tax (183.5 BPM)**:
```
[BarPhaseRefine] shift=0: phase=220.6ms, bars=46, avg_bass=0.6726
[BarPhaseRefine] shift=1: phase=547.6ms, bars=46, avg_bass=0.6571
[BarPhaseRefine] shift=2: phase=874.5ms, bars=46, avg_bass=0.4834
[BarPhaseRefine] shift=3: phase=1201.5ms, bars=45, avg_bass=0.6701
[BarPhaseRefine] Selected shift=0: 220.6ms -> 220.6ms
```

**結果**: shift=0（変更なし）を選択。shift 0,1,3のスコアが非常に近い（差<2.4%）ため区別不能。
**ユーザー報告**: "微妙に合っていない" → 小節頭がダウンビートに合わない問題が未解決。

### 根本原因分析

1. **離散シフトの限界**: 4つの離散的なビートシフトしか試さないため、
   Stage 3で検出されたビート位相（220.6ms）自体が微妙にずれている場合に修正不可能。
2. **Stage 3の位相精度**: precision=0.6175（61.75%のオンセットマッチ）は低く、
   全帯域オンセット（ハイハット、スネア含む）でグリッド探索するため、
   キックドラム（ダウンビート）に最適化されていない。
3. **均一なバスエネルギー**: shift 0,1,3で同程度 → キックが毎拍叩かれているか、
   ビートグリッド自体がずれてバスエネルギーが分散している。

---

## Phase 5.1: バス基準ビート位相微調整 (build-v5.5-bass-phase)

### 方針

現在のStage 3は**全帯域オンセット**でビート位相を検出するが、
ダウンビートの位置はバスドラムで決まる。

**2段階アプローチ**:
1. **Step A**: バスオンセットエンベロープのビート周期での自己相関ピーク位相を計算し、
   ±半拍の範囲でビート位相を微調整（"微妙に"のズレを修正）
2. **Step B**: 微調整後の位相で4シフトテスト（既存の小節頭選択）

### 修正対象

- `backend/main.py`
  - `refine_bar_phase()` → 書き換え（2段階方式）
  - ビルドタグ → `build-v5.5-bass-phase`

### アルゴリズム

```python
def refine_bar_phase(y: np.ndarray, sr: int, bpm: float, phase_offset_sec: float,
                     hop_length: int = 512, beats_per_bar: int = 4,
                     window_frames: int = 2) -> tuple[float, int]:
    """
    2段階でビート位相と小節頭を最適化:
    Step A: バスオンセットで±半拍のビート位相微調整
    Step B: 微調整後の位相で4シフトテスト（小節頭選択）
    """
    beat_duration = 60.0 / bpm
    frame_duration = hop_length / sr

    # 低音域オンセットエンベロープ
    y_bass = lowpass_filter(y, sr, cutoff_hz=200)
    y_bass = highpass_filter(y_bass, sr, cutoff_hz=20)
    bass_env = librosa.onset.onset_strength(y=y_bass, sr=sr, hop_length=hop_length)
    del y_bass

    total_frames = len(bass_env)
    total_duration_sec = total_frames * frame_duration

    if np.max(bass_env) < 1e-6:
        print("[BarPhaseRefine] No bass energy detected, skipping")
        return phase_offset_sec, 0

    # === Step A: バス基準ビート位相微調整 ===
    # ±半拍の範囲でフレーム単位に探索し、ビートグリッド上のバスエネルギーが最大の位相を見つける
    beat_period_frames = beat_duration / frame_duration
    search_range_frames = int(beat_period_frames / 2)
    current_phase_frames = phase_offset_sec / frame_duration

    best_bass_score = -1.0
    best_delta = 0

    for delta in range(-search_range_frames, search_range_frames + 1):
        candidate_phase = current_phase_frames + delta
        if candidate_phase < 0:
            continue

        # ビートグリッドを生成
        grid = np.arange(candidate_phase, total_frames, beat_period_frames)
        grid_int = np.round(grid).astype(int)
        grid_int = grid_int[(grid_int >= 0) & (grid_int < total_frames)]

        if len(grid_int) == 0:
            continue

        # ビート位置でのバスエネルギー合計
        score = float(np.sum(bass_env[grid_int]))

        if score > best_bass_score:
            best_bass_score = score
            best_delta = delta

    refined_beat_phase = (current_phase_frames + best_delta) * frame_duration
    if best_delta != 0:
        print(f"[BarPhaseRefine] Step A: beat phase adjusted by {best_delta} frames "
              f"({best_delta * frame_duration * 1000:.1f}ms): "
              f"{phase_offset_sec*1000:.1f}ms -> {refined_beat_phase*1000:.1f}ms")
    else:
        print(f"[BarPhaseRefine] Step A: beat phase unchanged at {refined_beat_phase*1000:.1f}ms")

    # === Step B: 4シフトテスト（小節頭選択） ===
    scores = []
    for shift in range(beats_per_bar):
        candidate_phase = refined_beat_phase + shift * beat_duration
        bar_starts = np.arange(candidate_phase, total_duration_sec,
                               beat_duration * beats_per_bar)
        bar_frames = (bar_starts * sr / hop_length).astype(int)

        energy = 0.0
        count = 0
        for f in bar_frames:
            lo = max(0, f - window_frames)
            hi = min(total_frames, f + window_frames + 1)
            if lo < hi:
                energy += np.max(bass_env[lo:hi])
                count += 1

        avg = energy / max(count, 1)
        scores.append(avg)
        print(f"[BarPhaseRefine] Step B: shift={shift}: "
              f"phase={candidate_phase*1000:.1f}ms, bars={count}, avg_bass={avg:.4f}")

    best_shift = int(np.argmax(scores))

    # 5%未満の改善なら変更しない
    if best_shift != 0:
        improvement = (scores[best_shift] - scores[0]) / (scores[0] + 1e-8)
        if improvement < 0.05:
            print(f"[BarPhaseRefine] Step B: shift={best_shift} improvement={improvement:.3f} < 0.05, keeping shift=0")
            best_shift = 0

    final_phase = refined_beat_phase + best_shift * beat_duration
    print(f"[BarPhaseRefine] Final: shift={best_shift}, "
          f"{phase_offset_sec*1000:.1f}ms -> {final_phase*1000:.1f}ms")
    return final_phase, best_shift
```

### 期待される効果

**tax (183.5 BPM)**:
- beat_duration = 327ms → search_range = ±163ms (±7 frames)
- Step Aでバスエネルギーが最大のビート位相に微調整
  - 220.6msから数十ms程度シフトして、キックドラムのタイミングに合致
- Step Bで正しい小節頭を選択

### パフォーマンス

- Step A: ~14回のイテレーション × ビート数（~180）= ~2,520 ops → 瞬時
- Step B: 4回のイテレーション（既存と同じ）
- 追加コスト: 無視できるレベル

---

## Phase 5.2: スネア認識型位相最適化 (build-v5.5-snare-phase)

### Context

build-v5.4-bar-phase でバスエネルギーのみの4シフトテストを実装したが、
taxで「ハイライトがワンテンポ遅い + 漸進的なドリフト」が報告された。

**build-v5.4テスト結果（tax 183.5 BPM）**:
```
shift=0: avg_bass=0.6726  ← 選択されたが、ダウンビートではない可能性
shift=1: avg_bass=0.6571
shift=2: avg_bass=0.4834  ← 最低（スネアビート？）
shift=3: avg_bass=0.6701  ← shift=0とほぼ同じ
```

**ユーザー追加確認**: "ハイライトの切り替わりが小節頭ではなく2拍目になっている"
→ 位相(220.6ms)がbeat 2にロックされていることが確定。shift=3(-1拍)への修正が必要。

**根本原因**:
1. バスエネルギーだけではダウンビートとバックビートを区別不能（キックが毎拍鳴る場合）
2. Stage 3の位相検出(precision=0.6175)がサブビートレベルで不正確
3. shift=0がbeat 2（スネアビート）にロック → スネアペナルティで明確に区別可能

### 修正対象ファイル

- `backend/main.py`
  - line 217: `bandpass_filter()` 追加
  - lines 832-889: `refine_bar_phase()` 全面書き換え
  - line ~1543: ビルドタグ → `build-v5.5-snare-phase`

### 変更1: `bandpass_filter()` 追加 (line 217)

```python
def bandpass_filter(y: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """Band-pass filter using Butterworth design."""
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, y)
```

`butter`/`sosfilt` は line 22 でインポート済み。既存の `highpass_filter`/`lowpass_filter` と同パターン。

### 変更2: `refine_bar_phase()` 書き換え (lines 832-889)

**2段階アプローチ:**

#### Step A: バス基準ビート位相微調整

±半拍の範囲でフレーム単位に探索し、ビートグリッド（毎拍）上のバスオンセットエネルギーが最大になる位相を選択。

- 探索範囲: ±`beat_period_frames/2` フレーム（tax: ±7フレーム = ±163ms）
- 各候補で全ビートグリッドのバスエネルギー合計を計算
- 数十msの位相ズレを修正 → "微妙に合っていない" を解消

#### Step B: スネア認識型ダウンビート選択

Step Aで微調整した位相から4つのバーシフト（0,1,2,3拍）をテスト。
**バスエンベロープ（20-200Hz）とスネアエンベロープ（2000-5000Hz）の複合スコア**で評価:

```
score = bass_avg - ALPHA × snare_avg   (ALPHA = 0.5)
```

| ビート位置 | キック | スネア | bass_avg | snare_avg | score |
|-----------|--------|--------|----------|-----------|-------|
| beat 1（ダウンビート） | ○ | × | 高 | 低 | **高** |
| beat 2（バックビート） | ○ | ○ | 高 | 高 | **低** |
| beat 3 | ○ | × | 高 | 低 | **高** |
| beat 4（バックビート） | ○ | ○ | 高 | 高 | **低** |

- スネアペナルティにより beat 1,3 と beat 2,4 を明確に区別
- バスのみの差(2.4%)では不可能だった判別がスネアペナルティで可能に
- スネアバンド(2000-5000Hz)はキック(20-200Hz)と帯域が重ならない
- スネアが検出されない場合(snare_max < bass_max × 5%)はALPHA=0にフォールバック

#### 安全策

- ステータスクオ保護: shift≠0 の改善が5%未満なら変更しない
- forced_phase チャンク(chunk 1+)ではスキップ（既存のガード維持）

### パフォーマンス

| 処理 | 追加コスト |
|------|-----------|
| bandpass_filter (スネアバンド) | ~20ms |
| onset_strength (スネア) | ~20ms |
| Step A 探索 (~15回 × ~180拍) | <1ms |
| Step B (4シフト × ~46バー) | <1ms |
| **合計** | **~40ms** |

### build-v5.5 テスト結果

**tax (183.5 BPM)**:
```
Step A: unchanged at 220.6ms  ← Step Aは変更なし（バスが毎拍均等）
Step B: shift=0: bass=0.2204, snare=0.1738, score=0.1335  ← snare最高=バックビート確認
Step B: shift=1: bass=0.2153, snare=0.1626, score=0.1341
Step B: shift=2: bass=0.1584, snare=0.1547, score=0.0811
Step B: shift=3: bass=0.2196, snare=0.1606, score=0.1393  ← 最高score=beat 1
Step B: shift=3 improvement=0.044 < 0.05, keeping shift=0  ← ★5%閾値でブロック★
```

**問題**: スネアペナルティは機能（shift=3が最高score）だが、改善幅4.4%が5%閾値を下回りブロック。
**ユーザー報告**: "小節の3拍目でハイライト" / "いんちき和尚は2拍目"

---

## Phase 5.3: 閾値修正 (build-v5.5.1)

### Context

build-v5.5のスネア認識型スコアリングは正しくshift=3（beat 1）を最高スコアとしたが、
ステータスクオ保護閾値（5%）が改善幅4.4%をブロックした。

### 修正内容

**`backend/main.py` の `refine_bar_phase()` 内、ステータスクオ閾値を 5% → 2% に変更。**

```python
# 変更前
if improvement < 0.05:

# 変更後
if improvement < 0.02:
```

### 根拠

- shift=3の改善幅: 4.4%（0.1393 vs 0.1335）→ 2%閾値を通過
- スネア認識スコアリングはバスのみより信頼性が高い（複合指標）
- 5%閾値はバスのみ方式（v5.4）用だった。スネア併用時は低閾値で十分
- shift=0(beat 2/3): snare=0.1738（最高） → バックビート確定
- shift=3(beat 1): snare=0.1606（低め） → キックビート → 正解

### 修正対象

- `backend/main.py`: `refine_bar_phase()` 内の閾値 1箇所
- ビルドタグ → `build-v5.5.1`

### Verification

1. taxのログで `shift=3 improvement=0.044 > 0.02` → shift=3が選択されること
2. ハイライトがbeat 1（ダウンビート）に移動すること
3. いんちき和尚でも同様にshift選択が改善されること
