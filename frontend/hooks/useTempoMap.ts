'use client';

import { useState, useCallback, useMemo } from 'react';
import { Bar } from '../lib/api';

export interface TempoCheckpoint {
  barIndex: number;    // どの小節に対するチェックポイントか
  audioTimeSec: number; // ユーザーがタップした実際の音源時刻
}

export interface MappedBar extends Bar {
  mapped_start_sec: number;
  mapped_end_sec: number;
}

/**
 * テンポマップ管理hook
 * - バックエンドから受け取ったBPMベースの等間隔グリッドを初期値として保持
 * - ユーザーがタップしたチェックポイントでグリッドを補正
 * - 補正後の start_sec / end_sec を各小節に再計算して提供
 */
export function useTempoMap(bars: Bar[] | null) {
  const [checkpoints, setCheckpoints] = useState<TempoCheckpoint[]>([]);

  // チェックポイントを追加（最寄りの小節に紐づけ）
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

    setCheckpoints(prev => {
      // 同じ小節のチェックポイントがあれば更新
      const filtered = prev.filter(cp => cp.barIndex !== nearestIdx);
      const next = [...filtered, { barIndex: nearestIdx, audioTimeSec }];
      next.sort((a, b) => a.barIndex - b.barIndex);
      return next;
    });

    return nearestIdx;
  }, []);

  // チェックポイントを削除
  const removeCheckpoint = useCallback((barIndex: number) => {
    setCheckpoints(prev => prev.filter(cp => cp.barIndex !== barIndex));
  }, []);

  // 全チェックポイントをリセット
  const resetCheckpoints = useCallback(() => {
    setCheckpoints([]);
  }, []);

  // チェックポイントに基づく補正済み小節タイミングを計算
  const mappedBars = useMemo((): MappedBar[] => {
    if (!bars || bars.length === 0) return [];
    if (checkpoints.length === 0) {
      // チェックポイントなし: 元のタイミングをそのまま使う
      return bars.map(b => ({
        ...b,
        mapped_start_sec: b.start_sec,
        mapped_end_sec: b.end_sec,
      }));
    }

    // チェックポイントのセグメント間を線形補間
    const result: MappedBar[] = bars.map(b => ({ ...b, mapped_start_sec: b.start_sec, mapped_end_sec: b.end_sec }));

    // チェックポイント区間ごとに線形補間
    const anchors: Array<{ barIndex: number; audioTimeSec: number }> = [
      { barIndex: 0, audioTimeSec: bars[0].start_sec }, // 先頭アンカー
      ...checkpoints,
      { barIndex: bars.length - 1, audioTimeSec: bars[bars.length - 1].end_sec }, // 末尾アンカー
    ].sort((a, b) => a.barIndex - b.barIndex);

    for (let seg = 0; seg < anchors.length - 1; seg++) {
      const aStart = anchors[seg];
      const aEnd = anchors[seg + 1];

      const originalStartTime = bars[aStart.barIndex].start_sec;
      const originalEndTime = bars[aEnd.barIndex].start_sec;
      const originalDuration = originalEndTime - originalStartTime;
      const targetDuration = aEnd.audioTimeSec - aStart.audioTimeSec;
      const ratio = originalDuration > 0.001 ? targetDuration / originalDuration : 1.0;

      for (let i = aStart.barIndex; i <= aEnd.barIndex; i++) {
        const offsetFromSegStart = bars[i].start_sec - originalStartTime;
        result[i].mapped_start_sec = aStart.audioTimeSec + offsetFromSegStart * ratio;
        const offsetEndFromSegStart = bars[i].end_sec - originalStartTime;
        result[i].mapped_end_sec = aStart.audioTimeSec + offsetEndFromSegStart * ratio;
      }
    }

    return result;
  }, [bars, checkpoints]);

  return {
    checkpoints,
    mappedBars,
    addCheckpoint,
    removeCheckpoint,
    resetCheckpoints,
  };
}
