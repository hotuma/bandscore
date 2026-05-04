# Plan: Workspace ページへの URL 解析機能追加

## Context

`/workspace` ページはファイルアップロード（ドラッグ&ドロップ）のみ対応しているが、他のページ（/preview, /early-access, /lab）はすでに YouTube URL 解析をサポートしている。ユーザーはワークスペースでも URL から解析できるようにしたい。

バックエンドの `/analyze/url` エンドポイントはファイルアップロードと同様に `{job_id}` を返す（HTTP 202）ため、既存のポーリング機構がそのまま流用できる。

## 変更ファイル

- **`frontend/app/workspace/page.tsx`** のみ変更

## 実装方針

### 1. 新規 state 追加

```typescript
const [inputMode, setInputMode] = useState<'file' | 'url'>('file');
const [urlInput, setUrlInput] = useState('');
const [cookiesFile, setCookiesFile] = useState<File | null>(null);
const cookiesInputRef = useRef<HTMLInputElement>(null);
```

### 2. `processUrl` 関数を追加

`processFile` の構造を踏襲し、以下の差分のみ変更：

- **初期送信**: `FormData` に `url`・`mode`・任意 `cookies` を付加し `/analyze/url` へ POST
- **URL バリデーション**: `lib/youtube.ts` の `extractVideoId` を使用して YouTube URL か確認（無効なら早期エラー）
- **audioUrl**: blob URL ではなく `result.audio_url` をそのまま使用（結果取得後に `setAudioUrl(resultData.audio_url ?? null)` ）
- **fileName 表示**: URL の短縮表示（例: `youtu.be/xxxxxxx`）

```typescript
const processUrl = useCallback(async (url: string) => {
  // YouTube URL バリデーション
  const videoId = extractVideoId(url);
  if (!videoId) {
    setError('有効な YouTube URL を入力してください');
    return;
  }

  // abort/reset 処理（processFile と同じ）
  // ...

  const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
  const formData = new FormData();
  formData.append('url', url);
  formData.append('mode', 'EARLY_ACCESS');
  if (cookiesFile) formData.append('cookies', cookiesFile);

  // Step 1: ジョブ送信 → /analyze/url
  const res = await fetch(`${base}/analyze/url`, { method: 'POST', body: formData, signal });
  // ...

  // Step 2: ポーリング（processFile と同一ロジック）
  // ...

  // 結果取得後
  setResult(resultData);
  setAudioUrl(resultData.audio_url ?? null);  // blob URL ではなく audio_url を使用
}, [cookiesFile]);
```

### 3. UI 変更（ドロップゾーン部分）

ファイル/URL の切り替えタブをドロップゾーン上部に追加：

```
[ ファイル ] [ URL ]
```

- **ファイルタブ**: 既存のドラッグ&ドロップ UI
- **URL タブ**:
  - YouTube URL 入力フィールド
  - 「解析開始」ボタン
  - Advanced: cookies.txt アップロード（折りたたみ）

## 再利用する既存リソース

| リソース | パス | 用途 |
|---|---|---|
| `extractVideoId` | `frontend/lib/youtube.ts:1` | YouTube URL バリデーション |
| `processFile` のポーリングロジック | `frontend/app/workspace/page.tsx:65-103` | URL 解析のポーリングに流用 |
| ドロップゾーンの CSS クラス | `frontend/app/globals.css` | タブ UI のスタイル |

## 検証方法

1. `npm run dev` でフロントエンド起動（ポート 3000）
2. `/workspace` を開く
3. 「URL」タブをクリック → 入力フィールドが表示されることを確認
4. YouTube URL を入力して「解析開始」→ ポーリングが始まり結果が表示されることを確認
5. 不正 URL（例: `https://example.com`）入力時にエラーが表示されることを確認
6. 「ファイル」タブに戻るとドラッグ&ドロップが機能することを確認
