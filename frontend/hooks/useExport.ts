'use client';

import { useCallback } from 'react';
import { Bar, AnalysisResult } from '../lib/api';

/**
 * コード進行のエクスポートhook
 * - テキスト形式: |C   |Am  |F   |G   |
 * - JSON形式: DAW連携用の構造化データ
 */
export function useExport() {
  /**
   * コード進行をChordSheetスタイルのテキストに変換
   */
  const buildChordText = useCallback((bars: Bar[], barsPerLine: number = 4): string => {
    if (!bars || bars.length === 0) return '';

    const lines: string[] = [];
    for (let i = 0; i < bars.length; i += barsPerLine) {
      const chunk = bars.slice(i, i + barsPerLine);
      const barStrings = chunk.map(b => {
        const chord = (b.chord || 'N.C.').padEnd(4, ' ');
        return `${chord}`;
      });
      lines.push('| ' + barStrings.join(' | ') + ' |');
    }
    return lines.join('\n');
  }, []);

  /**
   * DAW連携用JSONを構築
   */
  const buildExportJSON = useCallback((result: AnalysisResult, bars: Bar[]): object => {
    return {
      version: '1.0',
      exportedAt: new Date().toISOString(),
      metadata: {
        bpm: result.bpm,
        key: result.key,
        timeSignature: result.time_signature,
        durationSec: result.duration_sec,
        totalBars: bars.length,
      },
      bars: bars.map((b, i) => ({
        index: i + 1,
        chord: b.chord,
        startSec: Number(b.start_sec.toFixed(4)),
        endSec: Number(b.end_sec.toFixed(4)),
        durationSec: Number((b.end_sec - b.start_sec).toFixed(4)),
        tab: b.tab?.frets ?? null,
      })),
    };
  }, []);

  /**
   * テキストをクリップボードにコピー
   */
  const copyToClipboard = useCallback(async (text: string): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // フォールバック: textarea経由
      const el = document.createElement('textarea');
      el.value = text;
      el.style.position = 'fixed';
      el.style.opacity = '0';
      document.body.appendChild(el);
      el.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(el);
      return ok;
    }
  }, []);

  /**
   * JSONファイルとしてダウンロード
   */
  const downloadJSON = useCallback((data: object, filename: string = 'bandscore-export.json') => {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  /**
   * テキストファイルとしてダウンロード
   */
  const downloadText = useCallback((text: string, filename: string = 'chords.txt') => {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  return {
    buildChordText,
    buildExportJSON,
    copyToClipboard,
    downloadJSON,
    downloadText,
  };
}
