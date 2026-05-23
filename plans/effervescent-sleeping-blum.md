# BPM検出修正計画

## コンテキスト

UVERworld「哀しみはきっと」のBPM検出で、本来105 BPMであるものが、最初のチャンクでは正しく検出されるが、2番目以降のチャンクで162 BPMと誤検出される問題を修正する。

**問題の原因:**
- 2番目以降のチャンクで`forced_bpm=105.0`が渡されている
- しかし、hybrid BPM検出（lines 2357-2424）が`forced_bpm`を尊重せずに実行される
- DL confidence > 0.9 の場合、無条件でDL結果（162 BPM）で上書きしてしまう

## 修正アプローチ

`forced_bpm`が渡されている場合（2番目以降のチャンク）、hybrid BPM検出をスキップして、渡されたBPMをそのまま使用するように修正する。

## 修正ファイル

- `backend/main.py`

## 修正内容

### Line 2357-2424: ハイブリッドBPM検出のスキップ条件追加

```python
# --- ハイブリッド BPM 検出の統合 ---
# 既存の BPM 検出結果とハイブリッド方法を比較し、より良い結果を採用
# ただし、forced_bpm が渡されている場合はスキップ（2番目以降のチャンクでBPMを統一するため）
if forced_bpm is None:
    app_logger.debug("Integrating hybrid BPM detection...")
    try:
        # 元のBPMを中心に±40 BPMの範囲でハイブリッド検出を実行
        bpm_search_min = max(60, int(bpm - 40))
        bpm_search_max = min(240, int(bpm + 40))
        # ... 既存のhybrid BPM検出ロジック（lines 2361-2421）
    except Exception as e:
        app_logger.error(f"[Hybrid BPM] Error in hybrid detection: {e}, using original {bpm:.1f} BPM")
else:
    app_logger.info(f"[Hybrid BPM] Skipping hybrid detection - using forced BPM: {bpm:.1f}")
# --- ハイブリッド BPM 検出ここまで ---
```

**変更点:**
- hybrid BPM検出全体を `if forced_bpm is None:` で囲む
- `forced_bpm`が渡されている場合は、hybrid BPM検出をスキップしてログのみ出力

## 検証方法

1. バックエンドを再起動
2. UVERworld「哀しみはきっと」を解析
3. 以下を確認:
   - 最初のチャンク: 105 BPMを検出（DL confidence 0.993）
   - 2番目以降のチャンク: 105 BPMを維持（162 BPMに上書きされない）
   - ログに `[Hybrid BPM] Skipping hybrid detection - using forced BPM: 105.0` が出力される

## 関連コード

- `backend/main.py:2357-2424` - ハイブリッドBPM検出ロジック
- `backend/main.py:2731` - `forced_bpm`のパス
- `backend/main.py:2018-2022` - `forced_bpm`の使用箇所
