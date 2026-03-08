# /early-access ページにURL解析機能を統合する

## Context

現在 `/early-access` ページはMP3ファイルのアップロードのみに対応している。ユーザーはYouTube URLからも音声解析を行いたい。バックエンドにはすでに `/analyze/url` エンドポイントが存在するが、解析完了後に音声ファイルが削除されるため音声再生ができない問題がある。フロントエンド・バックエンド両方を修正し、URL解析 + 音声再生を実現する。

## 変更対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| [backend/main.py](backend/main.py) | URL解析時の音声ファイル保持 + `audio_url` をレスポンスに追加 |
| [frontend/app/early-access/page.tsx](frontend/app/early-access/page.tsx) | URL入力UI追加 + URL解析フロー実装 |

## Step 1: バックエンド修正 — 音声ファイルの保持と `audio_url` の返却

### 1-1. `run_analysis_bg` に `source` 引数を追加

[main.py:1538](backend/main.py#L1538) の `run_analysis_bg` 関数シグネチャを変更:

```python
def run_analysis_bg(job_id: str, file_path: str, mode: AnalyzeMode = AnalyzeMode.PREVIEW, source: str = "upload"):
```

### 1-2. `finally` ブロックでURL由来のファイルを保持

[main.py:1756-1761](backend/main.py#L1756-L1761) の `finally` ブロックを修正:

```python
finally:
    try:
        if source == "upload" and os.path.exists(file_path):
            os.remove(file_path)
        # source == "url" の場合はファイルを保持（cleanup_temp_dir の6時間TTLで自動削除）
    except Exception:
        pass
```

### 1-3. `final_result` に `audio_url` を追加

[main.py:1723-1735](backend/main.py#L1723-L1735) の `final_result` dict に条件付きで `audio_url` を追加:

```python
final_result = {
    ...既存フィールド...
}
if source == "url" and os.path.exists(file_path):
    final_result["audio_url"] = "/temp/" + os.path.basename(file_path)
```

### 1-4. `_process_analyze_url` からの呼び出しを更新

[main.py:1926](backend/main.py#L1926) を変更:

```python
threading.Thread(target=run_analysis_bg, args=(job_id, file_path, mode, "url")).start()
```

## Step 2: フロントエンド修正 — `/early-access` にURL入力を統合

### 2-1. 入力モード切り替えの状態追加

`page.tsx` に新しい状態変数を追加:

```typescript
type InputMode = 'file' | 'url';
const [inputMode, setInputMode] = useState<InputMode>('file');
const [youtubeUrl, setYoutubeUrl] = useState('');
```

`AppStatus` 型に `'downloading'` を追加:
```typescript
type AppStatus = 'idle' | 'uploading' | 'downloading' | 'analyzing' | 'ready' | 'error';
```

### 2-2. UI: ファイル/URL切り替えタブ

既存のアップロードセクション（line 460-507）の上にタブUIを追加:

```
[  File Upload  |  YouTube URL  ]
```

- `inputMode === 'file'` → 既存のファイルアップロードUIをそのまま表示
- `inputMode === 'url'` → YouTube URL入力欄 + 解析ボタン（ダークテーマで統一）

### 2-3. URL入力UI（ダークテーマ）

`inputMode === 'url'` のとき表示:
- テキスト入力: `placeholder="https://www.youtube.com/watch?v=..."`
- 「Generate Chord Draft」ボタン（既存ファイル解析と同じテキスト）
- URL検証に `extractVideoId()` を使用（[lib/youtube.ts](frontend/lib/youtube.ts) を import）

### 2-4. URL解析ハンドラ `handleAnalyzeUrl`

新しい関数を追加。既存の `handleAnalyze` と同じパターン（free tier gate → POST → poll）:

```typescript
const handleAnalyzeUrl = async () => {
    // 1. URL検証（extractVideoId で妥当性チェック）
    // 2. Free tier gate (canAnalyze)
    // 3. setStatus('downloading')
    // 4. POST /analyze/url に FormData { url, mode: 'EARLY_ACCESS' } を送信
    //    ※ analyzeYoutube() は使わない（202 job_id レスポンスを直接処理）
    // 5. 成功 → job_id を取得 → setStatus('analyzing') → pollJob() 開始
    // 6. エラー → YouTube固有エラーメッセージ（403/429）を表示
};
```

### 2-5. ローディング状態の更新

`status === 'downloading'` の場合にダウンロード中UIを表示:

```
Downloading from YouTube... (スピナー、プログレスバーなし)
```

`status === 'analyzing'` は既存のまま（プログレスバー付き）。

### 2-6. `pollJob` 結果からの `audio_url` 取得

既存の `pollJob` 内の `done` 処理（[page.tsx:247-268](frontend/app/early-access/page.tsx#L247-L268)）を修正:

```typescript
if (jobStatus === "done") {
    const r = await fetch(...);
    const resultData: AnalysisResult = await r.json();
    setResult(resultData);
    // URL解析の場合、バックエンドから返されるaudio_urlを使用
    if (resultData.audio_url) {
        const normalized = resultData.audio_url.startsWith('http')
            ? resultData.audio_url
            : `${base}${resultData.audio_url}`;
        setAudioUrl(normalized);
    }
    setProgress(100);
    setStatus('ready');
    ...
}
```

### 2-7. エラーメッセージのYouTube対応

`handleAnalyzeUrl` の `catch` ブロックでYouTube固有エラーを処理:
- **429**: 「YouTubeのレート制限に達しました。数分後に再度お試しください。」
- **403**: 「この動画にはアクセスできません。別の動画をお試しいただくか、MP3をダウンロードしてファイルアップロードをご利用ください。」
- その他: 汎用エラーメッセージ

### 2-8. リセット処理

`inputMode` 切り替え時に状態をクリーンアップ:
- file/url 切り替え時に `setResult(null)`, `setError(null)`, `setStatus('idle')` 等をリセット

### 2-9. audioUrl のクリーンアップ

ファイルアップロード時は `URL.createObjectURL` → `URL.revokeObjectURL` で管理。
URL解析時はバックエンドURLなので `revokeObjectURL` 不要。既存の `useEffect` クリーンアップでblob URLかどうか判定:

```typescript
useEffect(() => {
    return () => {
        if (audioUrl && audioUrl.startsWith('blob:')) {
            URL.revokeObjectURL(audioUrl);
        }
    };
}, [audioUrl]);
```

## 再利用する既存コード

| コード | ファイル | 用途 |
|-------|---------|------|
| `extractVideoId()` | [lib/youtube.ts](frontend/lib/youtube.ts) | URL検証 |
| `ResultDisplay` | [components/ResultDisplay.tsx](frontend/components/ResultDisplay.tsx) | 結果表示（変更不要） |
| `pollJob()` | [early-access/page.tsx](frontend/app/early-access/page.tsx#L202) | ジョブポーリング（`audio_url` 取得を追加） |
| `cleanup_temp_dir()` | [backend/main.py](backend/main.py#L39) | 6時間TTLで音声ファイル自動削除 |

## Step 3: バグ修正 — yt-dlp 403エラーが500として返される問題

### 問題

スクリーンショットで確認: `POST /analyze/url` が **500 Internal Server Error** を返している。

**原因の流れ:**
1. `download_youtube_audio` で yt-dlp が YouTube から 403 エラーを受ける
2. yt-dlp が `DownloadError` を発生させる（メッセージ例: `"ERROR: [youtube] ...: HTTP Error 403: Forbidden"`）
3. [main.py:1091](backend/main.py#L1091) のパターンマッチは特定の文字列のみ対応:
   - `"Sign in to confirm you're not a bot"`, `"cookies"`, `"Music Premium"`, `"Private video"`
4. 一般的な `"HTTP Error 403"` や `"Forbidden"` はマッチしない → `raise e` (line 1096)
5. `_process_analyze_url` の `except Exception` で 500 に変換される
6. フロントエンドがdetail中の "403" を検出 → `YOUTUBE_ACCESS_DENIED` 表示

### 修正箇所

#### 3-1. `download_youtube_audio` のパターンマッチ拡張

[main.py:1091](backend/main.py#L1091) の 403 判定条件を拡張:

```python
# 403 Forbidden (Login/Bot/Privacy)
if ("Sign in to confirm you're not a bot" in msg
    or "confirm you're not a bot" in msg
    or "cookies" in msg
    or "This video is only available to Music Premium members" in msg
    or "Private video" in msg
    or "HTTP Error 403" in msg
    or "Forbidden" in msg):
    raise HTTPException(
        status_code=403,
        detail="YouTube Access Denied (Login/Cookies required). Please try a different video or upload cookies.txt."
    )
```

#### 3-2. フロントエンドのエラー表示改善

現在、500 でもエラー詳細に "403" が含まれると `YOUTUBE_ACCESS_DENIED` として表示されるが、これはユーザーに対して正しい情報を伝えている。しかし、バックエンド修正後は適切な 403 ステータスが返るようになるため、フロントエンド側の修正は不要。

#### 3-3. yt-dlp バージョン更新の推奨

YouTube のボット検出は頻繁に変更されるため、yt-dlp を最新バージョンに更新することで解決する可能性が高い:

```bash
pip install --upgrade yt-dlp
```

## 検証方法

1. **バックエンド単体テスト**:
   - `POST /analyze/url` にYouTube URLを送信 → `job_id` 取得
   - `/analyze/status/{job_id}` をポーリング → `done` 確認
   - `/analyze/result/{job_id}` → レスポンスに `audio_url` フィールドが含まれることを確認
   - `audio_url` のパスにファイルが存在し、`GET /temp/{filename}` でアクセス可能なことを確認

2. **フロントエンド統合テスト**:
   - `/early-access` ページにアクセス
   - 「YouTube URL」タブに切り替え
   - YouTube URLを入力して解析実行
   - ダウンロード中 → 解析中のステータス遷移を確認
   - 解析完了後、ResultDisplayで音声再生とコード同期が機能することを確認
   - 「File Upload」タブに戻してファイルアップロードが正常に動作することを確認

3. **エラーケース**:
   - 無効なURL入力 → バリデーションエラー表示
   - 存在しない動画 → 適切なエラーメッセージ
   - Free tier制限到達 → CTAモーダル表示
