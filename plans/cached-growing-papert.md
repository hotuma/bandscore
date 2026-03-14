# コード音の音量制御修正 + 再生信頼性改善

## Context

ユーザーの報告:
1. **コード音が音源再生と同時に鳴らないことが多い**（なる時もあるが、ほとんど鳴らない）
2. **音量を300%にしても変化がない**

### 根本原因（調査結果）

**音量が効かない理由:**
- [ResultDisplay.tsx:715](frontend/components/ResultDisplay.tsx#L715) で呼ばれる `setChordVolume()` は [chordAudio.ts](frontend/lib/chordAudio.ts) の masterGain を更新するが、実際のコード再生は [guitarSound.ts](frontend/lib/guitarSound.ts) の **別の AudioContext** で行われている
- つまり `setChordVolume()` は**全く効果がない**（2つのモジュールが独立した AudioContext を持っている）
- soundfont-player の `.play()` に渡す `gain` パラメータ（L153）は機能するが、soundfont-player 内部で GainNode がどう扱われるか不透明

**コードが鳴らない理由:**
- Soundfont の読み込みが **遅延初期化**（初回 `playChordFromTabWithSoundFont` 呼び出し時にロード開始）
- ロード中（数百ms〜数秒）に予定された最初のバーは再生時点で過去になり、`delta < -0.1` でスキップされる（L544）
- ロード失敗時に `guitarPromise` が `null` でキャッシュされ、**以降永久にリトライされない**（L25-31）
- スケジューラの依存配列に `chordVolume` が含まれ（L580）、スライダー操作のたびにスケジューラが再マウントされる

## 修正内容

### 1. [guitarSound.ts](frontend/lib/guitarSound.ts) — マスターGainNode追加 + プリロード + リトライ

**a. マスターGainNodeの追加（音量制御の正しい実装）:**
```typescript
let masterGain: GainNode | null = null;
```
- `getGuitar()` 内で AudioContext 作成時に masterGain を作成し `audioContext.destination` に接続
- `Soundfont.instrument()` の第3引数に `{ destination: masterGain }` を渡して、soundfont出力をmasterGain経由にする
- 新関数 `setGuitarSoundVolume(v: number)` をエクスポート：`masterGain.gain.setTargetAtTime(v, ctx.currentTime, 0.01)` で滑らかに音量変更

**b. プリロード関数の追加:**
```typescript
export function preloadGuitar(): void {
    getGuitar(); // 呼ぶだけでロード開始（結果は内部キャッシュ）
}
```

**c. ロード失敗時のリトライ:**
- `.catch()` 内で `guitarPromise = null` にリセットし、次回呼び出し時にリトライ可能にする

### 2. [ResultDisplay.tsx](frontend/components/ResultDisplay.tsx) — 音量制御の修正 + プリロード

**a. import の変更:**
- `setChordVolume` (from chordAudio.ts) → `setGuitarSoundVolume` (from guitarSound.ts) に変更

**b. コンポーネントマウント時にプリロード:**
- `useEffect` で `preloadGuitar()` を呼び出し、ユーザーが再生ボタンを押す前にsoundfontを事前ロード

**c. 音量スライダーの修正:**
- `max={3}`（前回変更済み）を維持
- `setChordVolume(v)` → `setGuitarSoundVolume(v)` に変更
- `chordVolume` 初期化の `useEffect` (L615-618) も同様に修正

**d. スケジューラの依存配列から `chordVolume` を除去:**
- `chordVolume` を `useRef` でも保持し、`tick()` 内ではrefから読む
- 依存配列: `[autoChord, safeBars, chordVolume]` → `[autoChord, safeBars]`
- これにより、音量スライダー操作でスケジューラが再起動しなくなる

### 3. [chordAudio.ts](frontend/lib/chordAudio.ts) — 変更なし（import削除のみ）

- `setChordVolume` のimportが ResultDisplay.tsx から削除されるため、このファイルへの変更は不要
- （将来的にはこのファイル自体を削除候補とできるが、今回のスコープ外）

## 修正ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| [guitarSound.ts](frontend/lib/guitarSound.ts) | masterGain追加, `setGuitarSoundVolume()` 追加, `preloadGuitar()` 追加, リトライロジック |
| [ResultDisplay.tsx](frontend/components/ResultDisplay.tsx) | import変更, プリロード追加, 音量制御修正, スケジューラ依存配列修正 |

## 検証方法

1. `cd frontend && npm run dev` でフロントエンド起動
2. ページ表示時にブラウザのDevTools Consoleで soundfont ロード開始のログを確認
3. 音源を再生し、コード音が**最初のバーから確実に鳴る**ことを確認
4. Volスライダーを動かし、**リアルタイムで音量が変化する**ことを確認
5. 300%でコード音が明確に大きくなることを確認
6. スライダーを素早く動かしても**コードが途切れない**ことを確認（スケジューラが再起動しないため）
