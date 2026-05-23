# /labページの解析進捗表示の改善

## 問題の概要

`/lab`ページで音源解析が完了せず、進捗も表示されない問題。現在の実装は：
- ファイルアップロード解析が180秒タイムアウトの同期リクエスト
- 進捗表示がスピナーのみで進捗率不明
- YouTube解析のみがジョブポーリングを使用

## 解決策

ファイルアップロード解析も、YouTube解析と同じ非同期ジョブシステムを使用し、進捗バーとハング検出機能を追加します。

## 実装内容

### 1. `frontend/lib/api.ts` の変更

`analyzeAudio()`関数をジョブベースのポーリングに変更：

- ジョブIDを取得して非同期ポーリングを実行
- `onProgress`コールバックで進捗を通知
- スタートアップチェック（15秒で開始確認）
- 進捗ストール検出（120秒更新なしでタイムアウト）
- 全体タイムアウト（5分）
- キャンセル対応（`AbortSignal`）

### 2. `frontend/app/lab/page.tsx` の変更

進捗表示とキャンセル機能を追加：

- `progress`状態変数を追加
- `AbortController`でキャンセル対応
- 進捗バーUIの追加（スピナー + パーセンテージ + プログレスバー）
- `handleFileSelect`と`handleUrlSelect`を更新

## 修正するファイル

1. `c:\Users\USER\.gemini\antigravity\guitar-tab\frontend\lib\api.ts` - `analyzeAudio()`関数を書き換え
2. `c:\Users\USER\.gemini\antigravity\guitar-tab\frontend\app\lab\page.tsx` - 進捗表示を追加

## 参考ファイル

- `frontend/app/early-access/page.tsx` - 進捗バーの実装パターン
- `frontend/app/workspace/page.tsx` - ストール検出の実装パターン
- `backend/main.py` - ジョブステータスエンドポイントの確認

## 検証方法

1. ファイルをアップロードして進捗バーが表示されること
2. 進捗率が更新されること
3. タイムアウト時に適切なエラーメッセージが表示されること
4. 解析完了時に結果が正しく表示されること