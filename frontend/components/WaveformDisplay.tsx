'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { MappedBar } from '../hooks/useTempoMap';

interface WaveformDisplayProps {
  audioUrl: string;
  /** 補正済み小節データ（タイミング情報） */
  bars?: MappedBar[];
  /** 現在のハイライトバーインデックス */
  currentBarIndex?: number;
  /** シーク時のコールバック（秒） */
  onSeek?: (timeSec: number) => void;
  /** 再生状態の変化コールバック */
  onPlayPause?: (playing: boolean) => void;
  /** タップアライメント: タップ時に現在時刻を通知 */
  onTap?: (audioTimeSec: number) => void;
  /** 再生位置の外部注入（秒）- Nullなら制御しない */
  seekTo?: number | null;
  /** 既存の HTMLAudioElement を WaveSurfer の media として共有する（null = 未準備でスキップ） */
  mediaElement?: HTMLAudioElement | null;
  className?: string;
}

export default function WaveformDisplay({
  audioUrl,
  bars = [],
  currentBarIndex = -1,
  onSeek,
  onPlayPause,
  onTap,
  seekTo,
  mediaElement,
  className = '',
}: WaveformDisplayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const lastSeekRef = useRef<number | null>(null);

  // WaveSurfer 初期化
  useEffect(() => {
    if (!containerRef.current) return;
    // mediaElement が明示的に null の場合 = まだ audio 要素が未準備なのでスキップ
    if (mediaElement === null) return;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: 'rgba(124, 110, 245, 0.4)',
      progressColor: 'rgba(124, 110, 245, 0.9)',
      cursorColor: '#4ff7cf',
      cursorWidth: 2,
      height: 80,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      interact: true,
      backend: 'WebAudio',
      ...(mediaElement ? { media: mediaElement } : {}),
    });

    ws.load(audioUrl);

    ws.on('ready', () => {
      setIsReady(true);
      setDuration(ws.getDuration());
    });

    ws.on('play', () => {
      setIsPlaying(true);
      onPlayPause?.(true);
    });

    ws.on('pause', () => {
      setIsPlaying(false);
      onPlayPause?.(false);
    });

    ws.on('timeupdate', (t) => {
      setCurrentTime(t);
    });

    ws.on('seeking', (t) => {
      onSeek?.(t);
    });

    wsRef.current = ws;

    return () => {
      ws.destroy();
      wsRef.current = null;
      setIsReady(false);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl, mediaElement]);

  // 外部からのseekTo制御
  useEffect(() => {
    if (seekTo !== null && seekTo !== undefined && wsRef.current && isReady) {
      if (lastSeekRef.current !== seekTo) {
        lastSeekRef.current = seekTo;
        wsRef.current.seekTo(Math.min(1, Math.max(0, seekTo / wsRef.current.getDuration())));
      }
    }
  }, [seekTo, isReady]);

  const togglePlay = useCallback(() => {
    wsRef.current?.playPause();
  }, []);

  const handleTap = useCallback(() => {
    if (!wsRef.current || !isReady) return;
    const t = wsRef.current.getCurrentTime();
    onTap?.(t);
  }, [isReady, onTap]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  // 小節マーカーのレンダリング用: 各barのstart_secをパーセンテージに変換
  const barMarkers = duration > 0 && bars.length > 0
    ? bars.map((b, i) => ({
        pct: (b.mapped_start_sec / duration) * 100,
        isActive: i === currentBarIndex,
        chord: b.chord,
        index: i,
      }))
    : [];

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      {/* 波形 + バーマーカー */}
      <div className="waveform-container" style={{ minHeight: 80 }}>
        {/* ローディング表示 */}
        {!isReady && (
          <div className="absolute inset-0 flex items-center justify-center z-10 rounded-xl" style={{ background: 'var(--bg-surface)' }}>
            <div className="flex items-center gap-3">
              <div className="animate-spin" style={{ width: 20, height: 20, border: '2px solid var(--border-bright)', borderTopColor: 'var(--accent)', borderRadius: '50%' }} />
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>波形を読み込み中…</span>
            </div>
          </div>
        )}

        {/* WaveSurfer マウントポイント */}
        <div ref={containerRef} style={{ width: '100%' }} />

        {/* 小節線マーカー */}
        {isReady && (
          <div className="absolute inset-0 pointer-events-none" style={{ top: 0 }}>
            {barMarkers.map(m => (
              <div
                key={m.index}
                className="absolute"
                style={{
                  left: `${m.pct}%`,
                  top: 0,
                  bottom: 0,
                  width: m.isActive ? 2 : 1,
                  background: m.isActive
                    ? 'rgba(79, 247, 207, 0.9)'
                    : 'rgba(124, 110, 245, 0.25)',
                  transition: 'background 0.1s ease',
                }}
              />
            ))}
          </div>
        )}
      </div>

      {/* コントロールバー */}
      <div className="flex items-center gap-3">
        {/* 再生/一時停止ボタン */}
        <button
          className="ctrl-btn play-btn"
          onClick={togglePlay}
          disabled={!isReady}
          aria-label={isPlaying ? '一時停止' : '再生'}
        >
          {isPlaying ? (
            /* Pause icon */
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1"/>
              <rect x="14" y="4" width="4" height="16" rx="1"/>
            </svg>
          ) : (
            /* Play icon */
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="5,3 19,12 5,21"/>
            </svg>
          )}
        </button>

        {/* 時刻表示 */}
        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums', minWidth: 80 }}>
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>

        {/* スペーサー */}
        <div style={{ flex: 1 }} />

        {/* タップアライメントボタン（onTapが渡されている場合のみ表示） */}
        {onTap && (
          <button
            className="tap-btn"
            onClick={handleTap}
            disabled={!isReady}
            style={{
              padding: '0.5rem 1.25rem',
              borderRadius: 999,
              border: '1px solid rgba(79,247,207,0.4)',
              background: 'rgba(79,247,207,0.1)',
              color: '#4ff7cf',
              fontWeight: 700,
              fontSize: '0.8rem',
              cursor: isReady ? 'pointer' : 'not-allowed',
              transition: 'all 0.15s ease',
              letterSpacing: '0.05em',
            }}
            aria-label="小節頭にタップ"
          >
            ⬤ TAP
          </button>
        )}
      </div>
    </div>
  );
}
