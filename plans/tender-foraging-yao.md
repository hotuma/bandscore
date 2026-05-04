# TAP Align機能の修正計画

## Context

TAP Align機能が正しく機能していません。ユーザーがTAPボタンを押しても、そこが小節頭として認識されず、タイミング補正が正しく動作しません。

## 問題の原因

### 1. 使用するバー配列の不一致
**問題**: `handleTap` 関数で `rawBars`（未補正のバー）を使用していますが、TAP Align機能自体が `mappedBars`（補正済みバー）を生成しています。

```typescript
// ResultDisplayV2.tsx:324 (現在の問題のあるコード)
const nearestIdx = addCheckpoint(audioTimeSec, rawBars);
```

**影響**:
- `rawBars` には `mapped_start_sec` がないため、補正済みのタイミングで小節を特定できない
- ユーザーが見ている波形位置（補正済み）と、チェックポイント計算に使われるタイミング（未補正）が一致しない

### 2. 小節境界判定のロジック不足
**問題**: 現在の `addCheckpoint` は「最も近い小節の `start_sec`」を探していますが、これはTAPした位置が必ずしも小節の先頭であることを意味しません。

```typescript
// useTempoMap.ts:29-38 (現在のロジック)
let nearestIdx = 0;
let minDist = Infinity;
for (let i = 0; i < barsArr.length; i++) {
  const dist = Math.abs(barsArr[i].start_sec - audioTimeSec);
  if (dist < minDist) {
    minDist = dist;
    nearestIdx = i;
  }
}
```

**影響**:
- ユーザーが「ここが小節頭」とタップしたつもりでも、単に最も近い小節の先頭が選ばれる
- 小節の途中をタップしても、前の小節または後の小節の先頭が選ばれる可能性がある

### 3. ユーザー体験の不一致
**期待**: TAPした場所がそのまま小節頭になる
**現状**: TAPした場所に最も近い既存の小節頭が選ばれる

## 実装計画

### ステップ1: チェックポイント追加時の配列修正

**ファイル**: [frontend/components/ResultDisplayV2.tsx](frontend/components/ResultDisplayV2.tsx)

`handleTap` 関数で `rawBars` ではなく、適切なバー配列を使用するように修正：

```typescript
// 修正前（行324）
const nearestIdx = addCheckpoint(audioTimeSec, rawBars);

// 修正後
const nearestIdx = addCheckpoint(audioTimeSec, mappedBars);
```

**理由**:
- TAP Align機能自体が `mappedBars` を生成しているため、チェックポイントの計算にも `mappedBars` を使用すべき
- `mappedBars` には補正済みのタイミング情報が含まれているため、正確な小節特定が可能

### ステップ2: 小節境界判定ロジックの改善

**ファイル**: [frontend/hooks/useTempoMap.ts](frontend/hooks/useTempoMap.ts)

`addCheckpoint` 関数のロジックを改善し、TAPした位置に最も近い小節をより正確に特定：

```typescript
// 改善案: 小節の中間点も考慮して、より直感的な小節選択
const addCheckpoint = useCallback((audioTimeSec: number, barsArr: MappedBar[]) => {
  if (!barsArr || barsArr.length === 0) return;

  // どの小節の中にタップした位置があるか、または最も近い小節を見つける
  let nearestIdx = 0;
  let minDist = Infinity;

  for (let i = 0; i < barsArr.length; i++) {
    const barStart = barsArr[i].mapped_start_sec;
    const barEnd = barsArr[i].mapped_end_sec;

    // 小節の中に含まれている場合（優先）
    if (audioTimeSec >= barStart && audioTimeSec <= barEnd) {
      // 小節の前半ならこの小節、後半なら次の小節を候補に
      const midpoint = barStart + (barEnd - barStart) / 2;
      if (audioTimeSec <= midpoint) {
        nearestIdx = i;
        break; // この小節を採用
      } else if (i < barsArr.length - 1) {
        nearestIdx = i + 1; // 次の小節の先頭として扱う
        break;
      }
    }

    // 小節の外の場合: 最も近い小節を探す
    const distToStart = Math.abs(barStart - audioTimeSec);
    if (distToStart < minDist) {
      minDist = distToStart;
      nearestIdx = i;
    }
  }

  // チェックポイント追加処理（既存のコードと同じ）
  setCheckpoints(prev => {
    const filtered = prev.filter(cp => cp.barIndex !== nearestIdx);
    const next = [...filtered, { barIndex: nearestIdx, audioTimeSec }];
    next.sort((a, b) => a.barIndex - b.barIndex);
    return next;
  });

  return nearestIdx;
}, []);
```

**理由**:
- TAPした位置が小節のどの部分にあるかを考慮し、より直感的な小節選択を実現
- 小節の前半をタップした場合、その小節を選択
- 小節の後半をタップした場合、次の小節の先頭として扱う（「ここから次の小節」という意図を推定）

### ステップ3: 型定義の修正

**ファイル**: [frontend/hooks/useTempoMap.ts](frontend/hooks/useTempoMap.ts)

`addCheckpoint` 関数の引数型を `Bar[]` から `MappedBar[]` に修正：

```typescript
// 修正前
const addCheckpoint = useCallback((audioTimeSec: number, barsArr: Bar[]) => {

// 修正後
const addCheckpoint = useCallback((audioTimeSec: number, barsArr: MappedBar[]) => {
```

## 重要なファイル

- [frontend/components/ResultDisplayV2.tsx](frontend/components/ResultDisplayV2.tsx) - TAP Align UIとハンドラー
- [frontend/hooks/useTempoMap.ts](frontend/hooks/useTempoMap.ts) - テンポマップとチェックポイント管理
- [frontend/components/WaveformDisplay.tsx](frontend/components/WaveformDisplay.tsx) - 波形表示とTAPボタン

## 検証方法

1. **基本動作確認**:
   - 音楽ファイルをアップロードして分析を実行
   - 「Tap Align」ボタンをクリックしてアライメントモードを有効化
   - 音楽を再生し、明らかに小節の頭である位置でTAPボタンをクリック
   - チェックポイントが正しく登録され、波形上の小節線の位置が補正されることを確認

2. **複数チェックポイントの確認**:
   - 曲の冒頭、中盤、終盤など複数の位置でTAPを実行
   - すべてのチェックポイントが正しく登録されること
   - 各セグメント間のタイミングが正しく線形補間されていること

3. **リセット機能の確認**:
   - チェックポイントをいくつか追加した後、「リセット」ボタンをクリック
   - すべてのチェックポイントが削除され、元のタイミングに戻ることを確認

4. **再生同期の確認**:
   - TAP Align後に音楽を再生
   - コードのハイライトと実際の音声が同期していることを確認
   - 小節線が正しい位置に表示されていることを確認
