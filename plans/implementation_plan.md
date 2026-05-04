# Phase 1: クリエイター向けアレンジ空間 コアローカル機能

既存BandScore（Next.js 16 + FastAPI）をベースに、プロ向けクラウド型アレンジワークスペースの Phase 1 を実装する。既存のコード検出・BPM検出・Web Audio API再生機能を活かしつつ、UI/UXを一新し、タップアライメント・コード編集・テキスト出力機能を追加する。

## User Review Required

> [!IMPORTANT]
> **大規模なUI/UXリニューアルを含みます。** 既存の白背景Light UIからダークモード基調のプレミアムデザインに完全移行します。既存の`/lab`ページは残しつつ、新しい`/workspace`ルートを追加する方針です。

> [!WARNING]
> **WaveSurfer.jsの追加は新規依存です。** `wavesurfer.js`パッケージをnpm installで追加します。バンドルサイズが増加しますが、波形表示のコアライブラリとして必要不可欠です。

> [!IMPORTANT]
> **Phase 1の範囲確認：** Phase 2（クラウド共有）、Phase 3（AIサジェスト）は今回のスコープ外です。Phase 1ではローカル完結の機能に集中します。

---

## Proposed Changes

### デザインシステム・CSS基盤

新しいダークモード・グラスモーフィズムデザインシステムを構築し、既存の[globals.css](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/app/globals.css)を拡張する。

#### [MODIFY] [globals.css](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/app/globals.css)
- ダークモードカラーパレット（`--bg-primary`, `--bg-surface`, `--accent`等）をCSS変数で定義
- グラスモーフィズム効果（`backdrop-filter: blur()`）のユーティリティクラス
- 波形表示用のカスタムスタイル
- アニメーション・トランジション定義（`@keyframes`）
- タップアライメント用のパルスアニメーション
- 既存のbar-highlighted等のスタイルはダークテーマに適合するよう更新

---

### WaveSurfer.js 統合・波形表示

#### [NEW] [WaveformDisplay.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/WaveformDisplay.tsx)
- WaveSurfer.jsの初期化・波形描画
- 小節線（bar markers）の波形上オーバーレイ描画
- 再生位置カーソルの同期表示
- ピンチズーム・スクロール対応
- タップアライメントのチェックポイント表示

#### package.json への wavesurfer.js 追加
```bash
npm install wavesurfer.js
```

---

### タップ・アライメント機能

生演奏のテンポ揺らぎに対応するため、ユーザーが小節頭をタップしてグリッドを補正する機能。

#### [NEW] [TapAlignment.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/TapAlignment.tsx)
- タップボタンUI（大きなタッチターゲット、パルスアニメーション）
- タップした時刻 → 最寄りの小節頭に紐づけ
- 以降の小節グリッドを線形補間で自動補正
- タップ履歴の表示・取消し機能

#### [NEW] [useTempoMap.ts](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/hooks/useTempoMap.ts)
- テンポマップの状態管理（チェックポイント配列）
- チェックポイントに基づく各小節の`start_sec`/`end_sec`の再計算
- 初期値はバックエンドから受け取ったBPMベースの等間隔グリッド
- undo/redo対応

---

### コード編集UI

#### [NEW] [ChordEditor.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/ChordEditor.tsx)
- 小節をタップ → コード候補リスト（ポップオーバー）を表示
- ダイアトニックコード優先表示、全コードリスト切替
- コード選択時に即座にWeb Audio APIで1ストローク再生
- コード変更はローカルstateに記録（サーバー不要）

#### [NEW] [ChordPalette.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/ChordPalette.tsx)
- コード候補のグリッド表示コンポーネント
- カテゴリ別タブ（Major, Minor, 7th, sus4, m7, Maj7）
- 選択時のハプティックフィードバック

---

### メインワークスペース

#### [NEW] [page.tsx (workspace)](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/app/workspace/page.tsx)
- 新しいメインワークスペースページ
- WaveformDisplay + コード進行表示 + コントロールパネルの統合レイアウト
- ファイルドロップによる音源読み込み
- 分析結果の表示とインタラクション

#### [NEW] [WorkspaceLayout.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/WorkspaceLayout.tsx)
- ワークスペースの3パネルレイアウト
  - 上段: ヘッダー（プロジェクト名、BPM/Key情報）
  - 中段: 波形＋コード進行表示（メインエリア）
  - 下段: コントロールバー（再生/停止、音量、エクスポート）
- レスポンシブブレイクポイント対応

#### [NEW] [ResultDisplayV2.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/ResultDisplayV2.tsx)
- 既存の[ResultDisplay.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/ResultDisplay.tsx)のコアロジック（Anchor同期、Lookaheadスケジューラ、RAF loop）を継承
- 新デザインシステムに対応した表示
- コード編集UIとの統合
- テンポマップ対応の小節タイミング

---

### テキスト/JSON出力

#### [NEW] [ExportPanel.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/ExportPanel.tsx)
- テキスト出力: コード進行をプレーンテキスト形式でコピー
  - 例: `|C   |Am  |F   |G   |`
- JSON出力: DAW連携用の構造化データ
  - テンポマップ、キー、拍子、全小節のコード・タイミング
- クリップボードコピー + ファイルダウンロード

#### [NEW] [useExport.ts](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/hooks/useExport.ts)
- テキストフォーマットへの変換ロジック
- JSON構造の定義と出力
- ダウンロードトリガー（Blob URL生成）

---

### 既存コンポーネントへの影響

#### [MODIFY] [page.tsx (root)](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/app/page.tsx)
- `/` → `/workspace` へのリダイレクト追加（既存`/demo`リダイレクトの代替）

#### 既存ファイル（変更なし・互換性維持）
- [backend/main.py](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/backend/main.py) — Phase 1では変更なし。既存APIをそのまま活用
- [lib/guitarSound.ts](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/lib/guitarSound.ts) — 既存のままフル活用（playStroke, scheduleStrumPattern等）
- [lib/api.ts](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/lib/api.ts) — 既存のまま活用
- [components/ResultDisplay.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/ResultDisplay.tsx) — V1として残す（`/lab`ページで引き続き使用）
- [components/FileUpload.tsx](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/frontend/components/FileUpload.tsx) — V1として残す
- `/lab`, `/demo`, `/preview` 等の既存ルート — 変更なし

---

## Verification Plan

### 自動テスト

**既存バックエンドテスト（変更不要・回帰確認用）：**
```bash
cd c:\Users\USER\.gemini\antigravity\guitar-tab\backend
python tests\verify_modes.py
python tests\verify_preview_content.py
python tests\verify_simple.py
python tests\verify_bar_timing.py
```

**フロントエンドビルド確認：**
```bash
cd c:\Users\USER\.gemini\antigravity\guitar-tab\frontend
npm run build
```
ビルドエラーがないことを確認（TypeScriptコンパイル + Next.jsビルド）。

### ブラウザテスト（browser_subagent使用）

1. **ワークスペースページの表示確認**
   - `http://localhost:3000/workspace` にアクセス
   - ダークモードUIが正しく表示されること
   - ファイルアップロードエリアが表示されること

2. **音源読み込み + 波形表示**
   - MP3ファイルをアップロード
   - WaveSurfer.jsによる波形が表示されること
   - 分析結果（BPM, Key, コード進行）が表示されること

3. **コード編集**
   - コードをクリック → 候補リストが表示されること
   - 候補コードをクリック → 1ストローク再生されること
   - コードが更新されること

4. **テキスト出力**
   - エクスポートボタンをクリック
   - テキスト/JSONの出力が正しいことを確認

5. **既存ページの回帰テスト**
   - `http://localhost:3000/lab` が従来通り動作すること

### 手動テスト（ユーザー確認）

1. **スマートフォンでのレスポンシブUIの動作確認**
   - ブラウザの開発者ツールでモバイルビューに切り替え可能ですが、実機での確認も推奨
2. **タップアライメントの操作感**
   - 音源を再生しながらタップボタンを押して、小節グリッドが補正される感触を確認
3. **テンポ揺らぎのある実音源での動作確認**
   - リポジトリ内のテスト音源（[tax.mp3](file:///c:/Users/USER/.gemini/antigravity/guitar-tab/tax.mp3)等）で動作検証
