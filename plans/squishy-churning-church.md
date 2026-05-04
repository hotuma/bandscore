# 修正計画: コード音タイミングズレ・多重鳴り修正

## Context
**バグ1: ハイライトと1テンポ後にコード音が鳴る**
WaveSurfer（実音声出力）と muted audioRef（タイミング計測用）が独立した別 audio 要素として動作している。前回の修正（muted + onPlayPause）で2つの decode pipeline が分離されたため、`anchorRef` が audioRef の時刻に基づく一方、実際の音声は WaveSurfer が別の時計で再生する。この誤差が累積して "1テンポ" のズレとして現れる。

**バグ2: コード音がディレイ状に重なって鳴る**
- `onSeeked` で `scheduledUpToRef` がリセットされ、直前にスケジュール済みのバーが再スケジュールされる
- `scheduleStrumPattern` が `await playStroke()` ループ中にシークが割り込むと残りのストロークが重複してキューされる
- WebAudio API にキュー済みのノードはキャンセル不可能

**修正の方針:**
1. WaveSurfer に `media: audioRef.current` を渡して同一 HTMLAudioElement を共有する → 2つの audio 要素問題を根本から解消
2. `scheduleStrumPattern` にジェネレーショントークンを追加し、シーク時に進行中スケジュールをキャンセル

---

## 修正対象ファイル

### 1. `frontend/lib/guitarSound.ts`

**追加: ジェネレーショントークン (多重スケジュール防止)**

```typescript
let _scheduleToken = 0;

/** シーク時に呼び出す: 進行中の scheduleStrumPattern ループをキャンセル */
export function invalidateScheduled(): void {
  _scheduleToken++;
}

// scheduleStrumPattern を修正: await ループ前にトークンをキャプチャし、各イテレーションでチェック
export async function scheduleStrumPattern(frets, barStartCtxSec, barDurationSec): Promise<void> {
  const token = _scheduleToken;   // ← 追加
  const strokes = buildStrokeEvents(barDurationSec);
  for (let i = 0; i < strokes.length; i++) {
    if (_scheduleToken !== token) return;  // ← 追加: シーク後なら中断
    const stroke = strokes[i];
    // ... 既存ロジック
    await playStroke(frets, stroke.direction, { whenSec, durationSec, gain: stroke.gain });
  }
}
```

---

### 2. `frontend/components/WaveformDisplay.tsx`

**追加: `mediaElement` prop サポート**

WaveSurfer 7 の `media` オプションで既存の HTMLAudioElement を共有できる。

```typescript
interface WaveformDisplayProps {
  // ... 既存
  mediaElement?: HTMLAudioElement | null;  // ← 追加
}

// useEffect の変更:
useEffect(() => {
  if (!containerRef.current) return;
  if (mediaElement === null) return; // null = まだ audio 要素が未準備、スキップ

  const ws = WaveSurfer.create({
    container: containerRef.current,
    // ... 既存オプション
    ...(mediaElement ? { media: mediaElement } : {}),  // ← 追加
  });
  ws.load(audioUrl);  // 波形データのデコードには audioUrl を使用
  // ... 既存イベントリスナー
}, [audioUrl, mediaElement]);  // ← mediaElement を deps に追加
```

**注意:** `mediaElement` が null → effect スキップ（まだ audio 要素未準備）。undefined → 既存の動作（mediaElement prop なし）。actual element → WaveSurfer が共有使用。

---

### 3. `frontend/components/ResultDisplayV2.tsx`

**アーキテクチャ変更: 共有 media element を WaveSurfer に渡す**

```typescript
// 追加
const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);

// callback ref に変更
const audioCallbackRef = useCallback((el: HTMLAudioElement | null) => {
  audioRef.current = el;
  setAudioEl(el);
}, []);

// audio 要素の変更:
// Before: <audio ref={audioRef} src={audioUrl} muted .../>
// After:  <audio ref={audioCallbackRef} src={audioUrl} preload="auto" style={{display:'none'}} />
// (muted 削除)

// onSeeked ハンドラに追加:
import { invalidateScheduled, ... } from '../lib/guitarSound';

const onSeeked = () => {
  invalidateScheduled();  // ← 追加: 進行中スケジュールをキャンセル
  initAudioContext();
  resetAnchor();
  // ... 既存ロジック
};

// handleBarClick の変更:
// Before: setWaveformSeekTo(targetTime);  (auto-play なし)
// After:  audioRef.current.currentTime = targetTime; がそのまま WaveSurfer もシーク
//         auto-play するなら audioRef.current.play(); を追加（同一要素なので WaveSurfer も再生）

// WaveformDisplay の変更:
<WaveformDisplay
  audioUrl={audioUrl}
  bars={finalBars}
  currentBarIndex={currentBarIndexRef.current}
  mediaElement={audioEl}            // ← 追加
  // seekTo={waveformSeekTo}       // ← 削除 (不要)
  // onPlayPause={...}             // ← 削除 (不要: 同一要素なので WaveSurfer play = audioRef play)
  onTap={showAlignMode ? handleTap : undefined}
  // onSeek も削除: 同一要素なので WaveSurfer シーク = audioRef シーク
  //   ただし onSeeked イベントは audioRef に自動で発火するのでスケジューラーリセットは動作する
/>
```

**削除するもの:**
- `waveformSeekTo` state と `setWaveformSeekTo` 全使用箇所
- `onPlayPause` callback
- `onSeek={(t) => { audioRef.currentTime = t; }}` (同一要素なので不要)
- `muted` 属性

**保持するもの:**
- `audioRef` useRef（コード scheduling の timing 計測に引き続き使用）
- audioRef の全イベントリスナー（play, pause, ended, seeked）- 同一要素なので WaveSurfer 操作でも全て発火する

---

## 修正後のデータフロー

```
WaveSurfer play ボタン
  → wavesurfer.play() → audioRef.play() (同一要素)
  → audioRef 'play' event → onPlay handler → resetAnchor(), コードスケジューラー開始

WaveSurfer pause ボタン
  → wavesurfer.pause() → audioRef.pause() (同一要素)
  → audioRef 'pause' event → setIsPlaying(false)

WaveSurfer クリック（シーク）
  → wavesurfer.seekTo(t) → audioRef.currentTime = t (同一要素)
  → audioRef 'seeked' event → invalidateScheduled(), onSeeked handler

ChordCell クリック
  → handleBarClick()
  → playChordFromTabWithSoundFont() (soundfont 単発再生)
  → audioRef.currentTime = targetTime → (同一要素なので WaveSurfer カーソルも移動)
  → audioRef.play() → WaveSurfer も再生開始
  → audioRef 'play' event → onPlay → resetAnchor(), スケジューラー開始
```

---

## 検証手順
1. 音源アップロード → 解析完了後
2. play ボタン → 再生開始、コードが小節頭でハイライトされ、同時にコード音が鳴ることを確認
3. コード音が1回だけ鳴り、重ならないことを確認
4. pause ボタン → 即座に停止
5. コードセルをクリック → soundfont 単発、そこから再生継続
6. `npm run build` でビルド確認

---

# 修正計画: WaveSurfer ↔ audioRef 音声同期バグ修正 [実装済み・一部問題あり]

## Context
**バグ:** コード名をクリックすると音源が再生されるが、停止できない。

**根本原因:**
`ResultDisplayV2` には2つの独立した音源が存在する：
1. **WaveformDisplay 内の WaveSurfer** — 波形表示＋ユーザー向けの play/pause UI を持つ実際の音声出力
2. **ResultDisplayV2 の hidden `<audio>` (audioRef)** — コードスケジューリングのタイミング計測用（別の音声出力）

`handleBarClick`（コードセル クリック時）が `audioRef.current.play()` を呼ぶが、WaveformDisplay の play/pause ボタンは WaveSurfer のみ制御し、audioRef には一切触れない。そのため、audioRef で再生が始まると停止する手段がない。さらに両方が同じ音を出すので二重再生になる。

---

## 修正対象: `frontend/components/ResultDisplayV2.tsx` のみ

### 変更 1: hidden `<audio>` を muted にする（二重音声を防止）

```tsx
// Before
<audio ref={audioRef} src={audioUrl} preload="auto" style={{ display: 'none' }} />

// After
<audio ref={audioRef} src={audioUrl} muted preload="auto" style={{ display: 'none' }} />
```

`muted` にしても `currentTime` は進むため、コードスケジューリングのタイミング計測は正常に機能する。

---

### 変更 2: `waveformSeekTo` state を追加

```typescript
const [waveformSeekTo, setWaveformSeekTo] = useState<number | null>(null);
```

コードクリック時に WaveSurfer の再生位置も同期するために使用。

---

### 変更 3: `handleBarClick` からの自動再生を削除 & WaveSurfer シーク追加

```typescript
// Before (末尾)
audioRef.current.play();  // ← 削除

// After (targetTime計算後に追加)
setWaveformSeekTo(targetTime);  // WaveSurferも同じ位置にシーク
// audioRef.current.play() は削除 — クリックでは自動再生しない
```

クリックでは soundfont の単発再生とシークのみ行い、連続再生はユーザーの play ボタン操作に委ねる。

---

### 変更 4: `WaveformDisplay` に `seekTo` と `onPlayPause` を接続

```tsx
<WaveformDisplay
  audioUrl={audioUrl}
  bars={finalBars}
  currentBarIndex={currentBarIndexRef.current}
  seekTo={waveformSeekTo}          // 追加: handleBarClick からのシーク指示
  onPlayPause={(playing) => {       // 追加: WaveSurfer の再生状態を audioRef に反映
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.play();
    } else {
      audioRef.current.pause();
    }
  }}
  onTap={showAlignMode ? handleTap : undefined}
  onSeek={(t) => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, t);
    }
  }}
/>
```

---

## データフロー（修正後）

```
WaveSurfer play/pause ボタン
  → onPlayPause(playing)
  → audioRef.play() / audioRef.pause()
  → audioRef events (play/pause) → isPlaying state → コードスケジューリング ON/OFF

WaveSurfer シーク（波形クリック）
  → onSeek(t)
  → audioRef.currentTime = t （既存）

ChordCell クリック
  → handleBarClick()
  → playChordFromTabWithSoundFont() （soundfont 単発再生）
  → audioRef.currentTime = targetTime
  → setWaveformSeekTo(targetTime) → WaveSurfer もシーク
  → 自動再生なし（ユーザーが play を押す）
```

---

## 変更しないファイル
- `WaveformDisplay.tsx` — `seekTo`, `onPlayPause` は既に定義済みの props のため変更不要
- `ChordEditor.tsx` — 変更不要
- バックエンド — 変更不要

---

## 検証手順
1. 音源をアップロードして解析完了まで待つ
2. コードセルをクリック → soundfont の単発音が鳴り、波形カーソルが移動する。連続再生は始まらないことを確認
3. WaveformDisplay の play ボタンを押す → 再生開始、コードが自動ハイライトされることを確認
4. pause ボタンを押す → 即座に停止することを確認
5. 再生中にコードセルをクリック → その小節にシークして再生継続（soundfont は鳴る）
6. `npm run build` でビルドが通ることを確認

---

# 修正計画: workspace ページの非同期ジョブポーリング実装 [実装済み]

## Context
**バグ:** ワークスペースページで音源をアップロードしても BPM/KEY/BARS/DURATION がすべて空/NaN/0 になり、コード進行が表示されない。

**根本原因:** `workspace/page.tsx` が `api.ts` の `analyzeAudio()` を使っているが、バックエンドは `POST /analyze` で即座に結果を返さず `{"job_id": "..."}` (HTTP 202) を返す非同期設計になっている。`analyzeAudio()` はこのポーリングパターンを実装していないため、`job_id` オブジェクトをそのまま `AnalysisResult` として扱ってしまい、全フィールドが undefined/NaN になる。

**確認済みの正常な実装:** `/early-access/page.tsx` と `/preview/page.tsx` は各ページ内に `pollJob()` 関数を直接実装しており、正常に動作している。

---

## API フロー（正しい実装）

```
1. POST /analyze (FormData: file + mode)
   → 202 + {"job_id": "abc123"}

2. ポーリング: GET /analyze/status/{job_id} （1.5秒間隔）
   → {"status": "analyzing"/"done", "progress": 0.0-1.0, "error": null, "started_at": ...}

3. status === "done" になったら:
   GET /analyze/result/{job_id}
   → AnalysisResult (bpm, key, bars, duration_sec, ...)
```

---

## 修正対象ファイル

### [MODIFY] `frontend/app/workspace/page.tsx`

`analyzeAudio()` の呼び出しをやめ、`early-access/page.tsx` と同じポーリングパターンを実装する。

**変更前（現状）:**
```typescript
const data = await analyzeAudio(file, 'EARLY_ACCESS', 180000);
setResult(data);
```

**変更後（実装する内容）:**
```typescript
// 1. ジョブ送信
const formData = new FormData();
formData.append('file', file);
formData.append('mode', 'EARLY_ACCESS');
const res = await fetch(`${API_URL}/analyze`, { method: 'POST', body: formData, signal });
if (!res.ok) throw new Error(...);
const { job_id } = await res.json();

// 2. ポーリング
while (true) {
  await sleep(1500);
  const s = await fetch(`${API_URL}/analyze/status/${job_id}`, { signal });
  const { status, progress, started_at } = await s.json();
  setProgress(progress);

  if (status === 'done') {
    const r = await fetch(`${API_URL}/analyze/result/${job_id}`, { signal });
    const result = await r.json();
    setResult(result);
    break;
  }
  if (status === 'error') throw new Error('Analysis failed');
  // タイムアウト: started_at がなく15秒経過 → 投げる
}
```

**追加する UI 要素:**
- `progress` state (0-100)
- 解析中ローディング画面にプログレスバーを追加（`{Math.round(progress * 100)}%`）
- AbortController でキャンセル可能にする（「新しい音源を読み込む」ボタン押下時にcancel）

---

## 実装詳細

### 状態管理の追加
```typescript
const [progress, setProgress] = useState<number>(0);
const abortControllerRef = useRef<AbortController | null>(null);
```

### processFile の再実装（`early-access/page.tsx` L347-L390 を参考）

```typescript
const processFile = useCallback(async (file: File) => {
  // ファイル種別チェック...

  // 前のジョブをキャンセル
  abortControllerRef.current?.abort();
  const controller = new AbortController();
  abortControllerRef.current = controller;
  const { signal } = controller;

  setError(null);
  setResult(null);
  setProgress(0);
  setIsAnalyzing(true);
  setFileName(file.name);

  // Blob URL
  if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
  const blobUrl = URL.createObjectURL(file);
  blobUrlRef.current = blobUrl;
  setAudioUrl(blobUrl);

  const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
  const submittedAt = Date.now();

  try {
    // Step 1: Submit
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', 'EARLY_ACCESS');
    const res = await fetch(`${base}/analyze`, { method: 'POST', body: formData, signal });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Upload failed');
    }
    const { job_id } = await res.json();
    if (!job_id) throw new Error('No job ID returned');

    // Step 2: Poll
    let lastProgress = -1;
    let lastUpdateTime = Date.now();
    while (true) {
      if (signal.aborted) return;
      await new Promise(r => setTimeout(r, 1500));
      if (signal.aborted) return;

      const s = await fetch(`${base}/analyze/status/${job_id}`, { signal });
      if (s.status === 404) throw new Error('ジョブが見つかりません。再度お試しください。');
      const data = await s.json();
      const p = typeof data.progress === 'number' ? data.progress : 0;
      setProgress(p);

      // スタートアップチェック
      if (!data.started_at && Date.now() - submittedAt > 15000) {
        throw new Error('サーバーが応答しません。バックエンドが起動しているか確認してください。');
      }
      // ストールチェック
      if (p > lastProgress) { lastProgress = p; lastUpdateTime = Date.now(); }
      else if (Date.now() - lastUpdateTime > 25000) {
        throw new Error('解析がタイムアウトしました。');
      }

      if (data.status === 'error') throw new Error('解析に失敗しました。');
      if (data.status === 'done') {
        const r = await fetch(`${base}/analyze/result/${job_id}`, { signal });
        if (!r.ok) throw new Error('結果の取得に失敗しました。');
        const resultData = await r.json();
        // audio_url の正規化
        if (resultData.audio_url && !resultData.audio_url.startsWith('http')) {
          resultData.audio_url = `${base}${resultData.audio_url.startsWith('/') ? '' : '/'}${resultData.audio_url}`;
        }
        setResult(resultData);
        setProgress(1.0);
        return;
      }
    }
  } catch (err: any) {
    if (signal.aborted || err.name === 'AbortError') return;
    setError(err instanceof Error ? err.message : 'Analysis failed');
    setAudioUrl(null);
  } finally {
    setIsAnalyzing(false);
  }
}, []);
```

### 解析中 UI の更新（プログレスバー追加）
```tsx
{isAnalyzing && (
  <div ...>
    {/* スピナー */}
    <div>解析中… {Math.round(progress * 100)}%</div>
    {/* プログレスバー */}
    <div style={{ width: '100%', height: 4, background: 'var(--bg-overlay)', borderRadius: 2 }}>
      <div style={{ width: `${progress * 100}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.3s ease' }} />
    </div>
    {fileName && <div>{fileName}</div>}
  </div>
)}
```

### 「新しい音源を読み込む」ボタンのリセット時にAbort
```typescript
onClick={() => {
  abortControllerRef.current?.abort();
  setResult(null); setAudioUrl(null); setFileName(null); setProgress(0);
}}
```

---

## 変更しないファイル
- `api.ts` — 他ページが `analyzeAudio()` を使っているため変更しない
- `ResultDisplayV2.tsx` — データが正しく来れば動作する（変更不要）
- バックエンド — 変更不要

---

## 検証手順

1. バックエンド起動
   ```bash
   cd backend && uvicorn main:app --reload
   ```
2. フロントエンド起動
   ```bash
   cd frontend && npm run dev
   ```
3. `http://localhost:3000/workspace` で MP3 をアップロード
4. プログレスバーが進行し、解析完了後に BPM/KEY/BARS/DURATION が正しく表示されること
5. コード進行グリッドが表示され、タップで再生・編集できること
6. ビルドが通ること: `npm run build`
