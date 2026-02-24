# ハイライトとコード音の同期修正プラン

**作成日**: 2026-02-10
**ステータス**: 承認済み（未実装）

---

## 問題の概要

音源の再生に合わせてコード名のハイライト + コード音が鳴るが、以下の問題が発生：

1. **小節の頭でハイライト + コード音が鳴らない**
2. **ハイライト + 鳴るタイミングが一定ではない**

特に、コード音は正確に鳴っているが、**ハイライトのタイミングが早くなったり遅くなったりして小節頭に合っていない**。

---

## 根本原因の分析

### 原因1: HTMLAudioElementの`currentTime`の精度問題

**問題箇所**: [ResultDisplay.tsx:353-362](../frontend/components/ResultDisplay.tsx#L353-L362)

```typescript
const currentTime = audioRef.current.currentTime; // ← 精度が低い（数十ms単位でブレる）
const effectiveTime = currentTime - offsetSec;
const uiIndex = findBarIndexByTime(effectiveTime, safeBars);
```

**問題点**:
- `HTMLAudioElement.currentTime`は内部的に動画/音声デコーダーのフレームタイミングに依存
- ブラウザによって更新頻度が異なる:
  - Chrome: 約250ms間隔
  - Firefox: 約100ms間隔
- requestAnimationFrameは60FPS（16ms）で実行されるが、`currentTime`は遅れて更新される

### 原因2: 時刻ソースの二重基準

- **コード音の再生**: `AudioContext.currentTime`（マイクロ秒精度）を使用 → **正確**
- **ハイライト**: `HTMLAudioElement.currentTime`（低精度）を使用 → **不正確**

この二重基準により、**音は正確だがハイライトがズレる**現象が発生。

### 原因3: 既存の修正が不十分

前回の修正で以下を実施したが、ハイライトのズレは解消されなかった：

1. ✅ スケジューラーの遅延判定ロジック改善（-0.05秒 → -0.1秒）
2. ✅ スケジューラー実行間隔短縮（25ms → 16ms）
3. ✅ LOOKAHEAD_SEC延長（0.15秒 → 0.2秒）
4. ✅ strumSec短縮（0.015秒 → 0.012秒）
5. ✅ バックエンドのフレームベースタイミング生成

→ これらはコード音再生の精度を向上させたが、**ハイライトの時刻ソースが低精度のままだった**ため問題が残った。

---

## 修正計画

### 修正1: ハイライトもAudioContextベースの高精度時刻を使用

**対象ファイル**: `frontend/components/ResultDisplay.tsx`
**対象箇所**: L350-368（RAFループ内のハイライト更新ロジック）

**現在のコード**:
```typescript
const loop = () => {
    try {
        if (audioRef.current) {
            const currentTime = audioRef.current.currentTime; // ← 低精度
            const effectiveTime = currentTime - offsetSec;
            const uiIndex = findBarIndexByTime(effectiveTime, safeBars);
            setCurrentBarIndex(prev => {
                if (prev !== uiIndex) return uiIndex;
                return prev;
            });
        }
    } catch (e) {
        console.error("ResultDisplay RAF loop error:", e);
    }
    rafIdRef.current = requestAnimationFrame(loop);
};
```

**修正後のコード**:
```typescript
const loop = () => {
    try {
        if (audioRef.current) {
            // Anchorが無い場合は初期化を試みる
            if (!anchorRef.current) {
                resetAnchor();
            }

            let currentTime: number;

            // AudioContextベースの高精度時刻を使用
            const ctxNow = getAudioContextTime();
            if (ctxNow !== null && anchorRef.current) {
                // Anchorを使って高精度なAudio時刻を計算
                currentTime = anchorRef.current.audio + (ctxNow - anchorRef.current.ctx);
            } else {
                // フォールバック: AudioContextが利用不可の場合のみHTMLAudioElementを使用
                currentTime = audioRef.current.currentTime;
            }

            const effectiveTime = currentTime - offsetSec;
            const uiIndex = findBarIndexByTime(effectiveTime, safeBars);

            setCurrentBarIndex(prev => {
                if (prev !== uiIndex) return uiIndex;
                return prev;
            });
        }
    } catch (e) {
        console.error("ResultDisplay RAF loop error:", e);
    }
    rafIdRef.current = requestAnimationFrame(loop);
};
```

**変更点**:
1. `anchorRef`が無い場合、`resetAnchor()`を呼んで初期化を試みる
2. `AudioContext.currentTime`と`anchorRef`を使って高精度時刻を計算
3. AudioContextが利用不可の場合のみ、`HTMLAudioElement.currentTime`にフォールバック

---

### 修正2: RAFループの依存配列を最適化

**対象ファイル**: `frontend/components/ResultDisplay.tsx`
**対象箇所**: L384（useEffectの依存配列）

**現在のコード**:
```typescript
}, [isPlaying, offsetSec, safeBars, autoChord, chordVolume]);
```

**修正後のコード**:
```typescript
}, [isPlaying, offsetSec, safeBars]);
```

**変更理由**:
- `autoChord`と`chordVolume`はハイライト表示に影響しない
- これらが変更されるたびにRAFループが再起動され、`anchorRef`がリセットされる可能性がある
- 不要な依存を削除することで、Anchorの安定性が向上

---

## 期待される効果

### 1. ハイライトが小節の頭に正確に合う
- AudioContext時刻を使用することで、マイクロ秒精度のタイミング判定が可能
- HTMLAudioElementの更新タイミングに依存しない

### 2. 音とハイライトが完全に同期
- コード音再生とハイライト更新が**同じ時刻ソース**（AudioContext + Anchor）を使用
- 二重基準による不一致が解消

### 3. ブラウザ依存のブレが解消
- Chrome/Firefoxなどブラウザ間の`currentTime`更新頻度の違いの影響を受けない
- より一貫した同期動作

---

## 実装手順

### ステップ1: RAFループの修正
1. `frontend/components/ResultDisplay.tsx`のL350-368を修正
2. Anchor初期化チェックを追加
3. AudioContextベースの時刻計算ロジックを実装
4. HTMLAudioElementへのフォールバックを追加

### ステップ2: 依存配列の最適化
1. `frontend/components/ResultDisplay.tsx`のL384を修正
2. `autoChord`と`chordVolume`を依存配列から削除

### ステップ3: 動作確認
1. フロントエンドを起動（`npm run dev`）
2. デモページまたはプレビューページで音源を再生
3. 以下を確認:
   - ハイライトが小節の頭で正確に切り替わるか
   - コード音とハイライトが同期しているか
   - 複数回再生してタイミングのブレが無いか
4. ブラウザのコンソールで`[Anchor]`ログを確認し、Anchorが正常に機能しているか確認

---

## 注意事項

- この修正は既存の前回修正（スケジューラー改善、バックエンドタイミング精度向上）を**前提**としています
- 前回修正が未適用の場合、まずそちらを適用してからこの修正を実施してください
- AudioContextが利用不可の環境（非常に古いブラウザなど）では、フォールバックとしてHTMLAudioElementを使用

---

## 関連ファイル

- `frontend/components/ResultDisplay.tsx` - メイン修正対象
- `frontend/lib/guitarSound.ts` - AudioContext管理（既存、修正不要）
- `backend/main.py` - タイミング生成（前回修正済み）

---

## 次のアクションアイテム

- [ ] ステップ1: RAFループの修正
- [ ] ステップ2: 依存配列の最適化
- [ ] ステップ3: 動作確認
- [ ] 問題が解決しない場合: デバッグログを追加して時刻差を測定
