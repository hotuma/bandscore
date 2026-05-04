# プラン: ギタースタイルのストロークパターン実装

## Context

現在は各小節の先頭に1ストローク（全弦を低→高の順で0.012sずつずらして再生）しているだけ。
実際のギター演奏のようにジャカジャカと1小節内で複数ストロークを鳴らしたい。

目標: 1小節内で DDUD (ダウン・ダウン・アップ・ダウン) パターン等を繰り返し、
- ダウン: 全弦（低音→高音）
- アップ: 上3弦のみ（高音→低音）の軽い音

---

## 変更ファイル

1. **[frontend/lib/guitarSound.ts](frontend/lib/guitarSound.ts)** — ストローク関数を追加
2. **[frontend/components/ResultDisplay.tsx](frontend/components/ResultDisplay.tsx)** — スケジューラの呼び出し側を変更

---

## 実装手順

### Step 1: `guitarSound.ts` に型とヘルパーを追加

既存の `PlayChordOptions` の下に追記する。

```typescript
export type StrokeDirection = 'down' | 'up';

export interface StrokeEvent {
    direction: StrokeDirection;
    offsetSec: number;   // 小節先頭からの相対時間
    gain: number;        // アクセント係数
}
```

### Step 2: `playStroke()` 関数を追加

```typescript
export async function playStroke(
    frets: Array<number | string | null | undefined>,
    direction: StrokeDirection,
    options?: PlayChordOptions
): Promise<void>
```

- **ダウン**: `fretsToMidiNotes(frets)` の全音を低→高順 (stagger +0.012s/弦)
- **アップ**: インデックス 3〜5 (G3,B3,E4) のみを高→低順 (stagger +0.010s/弦)、gain を 0.75 倍

内部実装:
```typescript
const guitar = await getGuitar();
if (!guitar || !audioContext) return;
const baseWhen = options?.whenSec ?? audioContext.currentTime;
const duration = options?.durationSec ?? 0.8;
const gain = options?.gain ?? 1.0;

if (direction === 'down') {
    const notes = fretsToMidiNotes(frets);
    notes.forEach((midi, idx) => {
        guitar.play(midi, baseWhen + idx * 0.012, { duration, gain });
    });
} else {
    // up: 上3弦 (idx 3,4,5) のみ、高→低順
    const upFrets = frets.map((f, i) => i >= 3 ? f : null);
    const notes = fretsToMidiNotes(upFrets).reverse();
    notes.forEach((midi, idx) => {
        guitar.play(midi, baseWhen + idx * 0.010, { duration, gain: gain * 0.75 });
    });
}
```

### Step 3: `scheduleStrumPattern()` 関数を追加

```typescript
export async function scheduleStrumPattern(
    frets: Array<number | string | null | undefined>,
    barStartCtxSec: number,
    barDurationSec: number
): Promise<void>
```

バー長さからストロークイベントを生成して全ストロークをスケジュール:

```typescript
const strokes = buildStrokeEvents(barDurationSec);
for (const stroke of strokes) {
    const whenSec = barStartCtxSec + stroke.offsetSec;
    // 次のストロークまでの長さ（最後は残り時間）
    const nextOffset = ...; // strokes[i+1]?.offsetSec ?? barDurationSec
    const durationSec = Math.max(0.15, nextOffset - stroke.offsetSec);
    await playStroke(frets, stroke.direction, { whenSec, durationSec, gain: stroke.gain });
}
```

### Step 4: `buildStrokeEvents()` ヘルパーを追加

バー長さに応じたストロークパターンを返す:

```typescript
function buildStrokeEvents(barDurationSec: number): StrokeEvent[] {
    // 短いバー (< 0.5s): ダウン1回のみ
    if (barDurationSec < 0.5) {
        return [{ direction: 'down', offsetSec: 0, gain: 1.0 }];
    }
    // 標準バー: DDUD を barDurationSec に均等配置
    const pattern: Array<[StrokeDirection, number]> = [
        ['down', 1.0],
        ['down', 0.7],
        ['up',   0.6],
        ['down', 0.85],
    ];
    return pattern.map(([direction, gain], i) => ({
        direction,
        gain,
        offsetSec: (i / pattern.length) * barDurationSec,
    }));
}
```

バー長さが長い場合 (> 3s) はパターンを2倍繰り返す。

### Step 5: `ResultDisplay.tsx` の呼び出し変更

スケジューラ `tick()` 内の以下の箇所（約 550〜580 行）:

**変更前:**
```typescript
const frets = bar?.tab?.frets;
if (frets) {
    const sustainSec = Math.min(1.8, Math.max(0.25, (barEndAudio - barStartAudio) * 0.95));
    playChordFromTabWithSoundFont(frets, {
        durationSec: sustainSec,
        whenSec,
        strumSec: 0.012,
    }).then(...).catch(console.error);
}
```

**変更後:**
```typescript
const frets = bar?.tab?.frets;
if (frets) {
    const barDuration = barEndAudio - barStartAudio;
    scheduleStrumPattern(frets, whenSec, barDuration).catch(console.error);
}
```

import 行に `scheduleStrumPattern` を追加。

---

## 確認方法

1. `npm run dev` でフロントエンド起動
2. 分析済みデータで音声を再生 → 1小節に4ストローク（DDUD）聞こえることを確認
3. アップストロークが軽い音（高弦のみ）になっていることを確認
4. シーク後に正常にストロークパターンが再開することを確認
5. バー長さが短い小節（< 0.5s）では1ストロークのフォールバック動作を確認
