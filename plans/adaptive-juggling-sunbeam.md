# Plan: ローカル変更を Vercel にデプロイ

## Context

early-accessページで積み上げてきた改善（YouTube URL解析、audio_url正規化、エラーハンドリング等）がpreviewページに反映されていない。また、previewページにあるモバイル最適化・INIT auto-retryがearly-accessにも不足している。両ページの機能を同期する。

## 変更対象ファイル

- `frontend/app/preview/page.tsx` (主要変更)
- `frontend/app/early-access/page.tsx` (モバイル最適化のみ追加)

## Part 1: preview/page.tsx への変更

### 1-A. 型定義・import追加

```typescript
// AppStatus に 'downloading' を追加
type AppStatus = 'idle' | 'uploading' | 'downloading' | 'analyzing' | 'ready' | 'error';
type InputMode = 'file' | 'url';  // 新規追加

// import に extractVideoId を追加
import { extractVideoId } from "../../lib/youtube";
```

### 1-B. state追加

```typescript
const [inputMode, setInputMode] = useState<InputMode>('file');
const [youtubeUrl, setYoutubeUrl] = useState('');
```

### 1-C. pollJob: audio_url正規化 + progress(100)設定

`done` ブロック（97〜106行目）に追加：
```typescript
if (resultData.audio_url) {
    const normalized = resultData.audio_url.startsWith('http')
        ? resultData.audio_url
        : `${base}${resultData.audio_url}`;
    setAudioUrl(normalized);
}
setProgress(100);  // 明示的に100%設定
```

### 1-D. blob URL only cleanup

```typescript
// 変更前: if (audioUrl) URL.revokeObjectURL(audioUrl);
// 変更後:
if (audioUrl && audioUrl.startsWith('blob:')) URL.revokeObjectURL(audioUrl);
```

### 1-E. handleAnalyzeUrl 関数を追加（handleAnalyzeの直後）

- `extractVideoId()` でURL検証
- エンドポイント: `${base}/analyze/url/preview`（バックエンドで PREVIEW mode 強制）
- `status = 'downloading'` → fetch → `status = 'analyzing'` → `pollJob()`
- エラーコード: `YOUTUBE_RATE_LIMIT` (429), `YOUTUBE_ACCESS_DENIED` (403), `INVALID_URL`
- `signal: controller.signal` を fetch に渡す（中断対応）

### 1-F. JSX変更

1. **Input mode タブ**（HEADERとUPLOAD SECTIONの間に追加）
   - アクティブカラー: `text-yellow-500 border-b-2 border-yellow-500`（previewのアクセントカラー統一）
   - タブ切替時に `setError(null)` もリセット

2. **File upload section を `{inputMode === 'file' && ...}` で条件付き**

3. **YouTube URL input section を追加**（`inputMode === 'url'` のとき表示）
   - フォーカスカラー: `focus:border-yellow-500`
   - ボタン: `bg-neutral-700 hover:bg-neutral-600`（previewトーン統一）

4. **Downloading state UI を追加**（LOADING STATE の直前）

5. **Retry ボタンの切り替え**
   ```tsx
   onClick={inputMode === 'url' ? handleAnalyzeUrl : handleAnalyze}
   ```

## Part 2: early-access/page.tsx への変更

### 2-A. handleAnalyze にモバイル最適化を追加

`handleAnalyze`（301行目）の `setStatus('analyzing')` 後、FormData構築前に追加：

```typescript
// Mobile/iCloud optimization: pre-read to ensure file is fully local before POST
const buffer = await file.arrayBuffer();
const safeFile = new File([buffer], file.name, {
    type: file.type || 'audio/mpeg',
    lastModified: file.lastModified,
});
// formData.append('file', file) → formData.append('file', safeFile) に変更
```

### 2-B. handleAnalyze に INIT auto-retry を追加

既存の直接 `fetch` 呼び出しを3回リトライ（800ms, 1.6s, 3.2sバックオフ）で包む。
AbortError はリトライせず即時 throw する。

## 変更しないもの

- `early-access/page.tsx` の `pollJob` は現状維持（pollの実装差異はスコープ外）
- `ResultDisplay.tsx` は変更不要（ハイライトタイミングは既に両ページ共有）
- バックエンド変更不要（`/analyze/url/preview` エンドポイントは既存）
- `lib/api.ts` 変更不要（`handleAnalyzeUrl` では直接 `fetch` を使う）

## 確認・テスト方法

1. `npm run dev` でフロントエンド起動
2. `/preview` ページで「YouTube URL」タブが表示されることを確認
3. YouTube URL入力 → 「Run Preview Analysis」でダウンロード → 解析 → ResultDisplay表示
4. ファイルアップロードも引き続き動作することを確認
5. `/early-access` ページのファイル解析が引き続き動作することを確認（モバイル最適化が追加されてもUIは同じ）
