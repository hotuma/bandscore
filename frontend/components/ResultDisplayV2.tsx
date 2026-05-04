'use client';

import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { AnalysisResult, Bar } from '../lib/api';
import {
  playChordFromTabWithSoundFont, scheduleStrumPattern,
  getAudioContextTime, initAudioContext, setGuitarSoundVolume, preloadGuitar,
  invalidateScheduled,
} from '../lib/guitarSound';
import { useTempoMap } from '../hooks/useTempoMap';
import WaveformDisplay from './WaveformDisplay';
import { ChordPalette, ChordCell } from './ChordEditor';
import ExportPanel from './ExportPanel';

interface ResultDisplayV2Props {
  result: AnalysisResult;
  audioUrl: string | null;
}

// ダイアトニックコード （簡易推定: キー情報から） 
function getDiatonicChords(key: string): string[] {
  const noteNames = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  const isMinor = key.endsWith('m');
  const rootName = isMinor ? key.slice(0, -1) : key;
  const rootIdx = noteNames.indexOf(rootName);
  if (rootIdx === -1) return [];
  
  const majorDegrees = [0,2,4,5,7,9,11];
  const minorDegrees = [0,2,3,5,7,8,10];
  const degrees = isMinor ? minorDegrees : majorDegrees;
  const majorQualities = isMinor
    ? ['m','','','m','m','','']
    : ['','m','m','','','m','m'];
  
  return degrees.map((d, i) => {
    const noteName = noteNames[(rootIdx + d) % 12];
    return noteName + majorQualities[i];
  });
}

export default function ResultDisplayV2({ result, audioUrl }: ResultDisplayV2Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);
  const audioCallbackRef = useCallback((el: HTMLAudioElement | null) => {
    audioRef.current = el;
    setAudioEl(el);
  }, []);
  const currentBarIndexRef = useRef<number>(-1);
  const [isPlaying, setIsPlaying] = useState(false);
  const rafIdRef = useRef<number | null>(null);

  // Scheduler refs (inherited from ResultDisplay v1)
  const anchorRef = useRef<{ audio: number; ctx: number } | null>(null);
  const schedulingBarRef = useRef<number>(0);
  const schedulerTimerRef = useRef<number | null>(null);
  const scheduledUpToRef = useRef<number>(-1);
  const lastScheduledBarRef = useRef<number>(-1);
  const barRefs = useRef<(HTMLDivElement | null)[]>([]);
  const lastPlayedBarRef = useRef<number>(-1);

  // UX state
  const [offsetSec, setOffsetSec] = useState<number>(0);
  const [autoChord, setAutoChord] = useState<boolean>(true);
  const [chordVolume, setChordVolumeState] = useState<number>(0.8);
  const [showExport, setShowExport] = useState(false);
  const [showAlignMode, setShowAlignMode] = useState(false);
  const [tapFeedback, setTapFeedback] = useState(false);
  const [audioDurationSec, setAudioDurationSec] = useState<number>(0);

  // Chord editing state
  const [editingBarIndex, setEditingBarIndex] = useState<number | null>(null);
  const [editAnchorRect, setEditAnchorRect] = useState<DOMRect | null>(null);
  const [chordOverrides, setChordOverrides] = useState<Map<number, { chord: string; frets: string[] }>>(new Map());

  // Tempo map
  const rawBars = useMemo(() => result?.bars ?? [], [result]);
  const { mappedBars, checkpoints, addCheckpoint, resetCheckpoints } = useTempoMap(rawBars);

  // コード上書きを適用した最終バー配列
  const finalBars = useMemo(() => {
    return mappedBars.map((b, i) => {
      const override = chordOverrides.get(i);
      if (override) {
        return {
          ...b,
          chord: override.chord,
          tab: override.frets.length > 0 ? { frets: override.frets } : b.tab,
        };
      }
      return b;
    });
  }, [mappedBars, chordOverrides]);

  // ダイアトニックコード
  const diatonicChords = useMemo(() => getDiatonicChords(result?.key ?? ''), [result?.key]);

  // Preload guitar soundfont
  useEffect(() => { preloadGuitar(); }, []);

  // Audio duration tracking
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onMeta = () => {
      const d = Number(audio.duration);
      if (Number.isFinite(d) && d > 0) setAudioDurationSec(d);
    };
    audio.addEventListener('loadedmetadata', onMeta);
    if (audio.readyState >= 1) onMeta();
    return () => audio.removeEventListener('loadedmetadata', onMeta);
  }, [audioUrl]);

  // Anchor utilities (same as v1)
  const resetAnchor = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const ctxNow = getAudioContextTime();
    if (ctxNow == null) return;
    anchorRef.current = { audio: audio.currentTime, ctx: ctxNow };
  }, []);

  // Binary search for current bar
  const findBarIndexByTime = useCallback((t: number, bars: typeof finalBars): number => {
    if (!bars || bars.length === 0 || !Number.isFinite(t)) return -1;
    const firstS = Number(bars[0]?.mapped_start_sec ?? bars[0]?.start_sec);
    const lastE = Number(bars[bars.length - 1]?.mapped_end_sec ?? bars[bars.length - 1]?.end_sec);
    if (!Number.isFinite(firstS) || !Number.isFinite(lastE)) return -1;
    if (t < firstS) return -1;
    if (t >= lastE) return bars.length - 1;

    let lo = 0, hi = bars.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const s = Number(bars[mid]?.mapped_start_sec ?? bars[mid]?.start_sec);
      const e = Number(bars[mid]?.mapped_end_sec ?? bars[mid]?.end_sec);
      if (!Number.isFinite(s) || !Number.isFinite(e)) return -1;
      if (t < s) hi = mid - 1;
      else if (t >= e) lo = mid + 1;
      else return mid;
    }
    return Math.min(bars.length - 1, Math.max(-1, hi));
  }, []);

  // Audio event listeners
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onPlay = () => {
      setIsPlaying(true);
      lastPlayedBarRef.current = -1;
      initAudioContext();
      resetAnchor();
      const t = audio.currentTime - offsetSec;
      const cur = findBarIndexByTime(t, finalBars);
      schedulingBarRef.current = Math.max(0, cur);
      scheduledUpToRef.current = schedulingBarRef.current - 1;
    };
    const onPause = () => setIsPlaying(false);
    const onEnded = () => { setIsPlaying(false); currentBarIndexRef.current = -1; anchorRef.current = null; };
    const onSeeked = () => {
      invalidateScheduled();
      initAudioContext();
      resetAnchor();
      const t = audio.currentTime - offsetSec;
      const cur = findBarIndexByTime(t, finalBars);
      schedulingBarRef.current = Math.max(0, cur);
      scheduledUpToRef.current = schedulingBarRef.current - 1;
      lastPlayedBarRef.current = -1;
      lastScheduledBarRef.current = -1;
    };
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('ended', onEnded);
    audio.addEventListener('seeked', onSeeked);
    return () => {
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('seeked', onSeeked);
    };
  }, [audioUrl, finalBars, offsetSec, resetAnchor, findBarIndexByTime]);

  // RAF highlight loop
  useEffect(() => {
    if (!isPlaying) {
      if (rafIdRef.current) { cancelAnimationFrame(rafIdRef.current); rafIdRef.current = null; }
      return;
    }
    const loop = () => {
      try {
        if (audioRef.current && !audioRef.current.paused) {
          if (!anchorRef.current) resetAnchor();
          let currentTime: number;
          const ctxNow = getAudioContextTime();
          if (ctxNow !== null && anchorRef.current) {
            currentTime = anchorRef.current.audio + (ctxNow - anchorRef.current.ctx);
            const drift = currentTime - audioRef.current.currentTime;
            if (Math.abs(drift) > 0.03) {
              anchorRef.current = { audio: audioRef.current.currentTime, ctx: ctxNow };
              currentTime = audioRef.current.currentTime;
            }
          } else {
            currentTime = audioRef.current.currentTime;
          }
          const effectiveTime = currentTime - offsetSec;
          const uiIndex = findBarIndexByTime(effectiveTime, finalBars);
          if (currentBarIndexRef.current !== uiIndex) {
            const prevIndex = currentBarIndexRef.current;
            if (prevIndex >= 0) {
              document.querySelector(`[data-bar-index="${prevIndex}"].bar-grid-item`)?.classList.remove('bar-highlighted');
              barRefs.current[prevIndex]?.classList.remove('bar-highlighted');
            }
            const newElGrid = document.querySelector(`[data-bar-index="${uiIndex}"].bar-grid-item`);
            const newElCard = barRefs.current[uiIndex];
            newElGrid?.classList.add('bar-highlighted');
            newElCard?.classList.add('bar-highlighted');
            currentBarIndexRef.current = uiIndex;
            // Auto-scroll to current chord cell
            newElGrid?.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'center' });
          }
        }
      } catch (e) { console.error('[RAF] error:', e); }
      rafIdRef.current = requestAnimationFrame(loop);
    };
    rafIdRef.current = requestAnimationFrame(loop);
    return () => { if (rafIdRef.current) { cancelAnimationFrame(rafIdRef.current); rafIdRef.current = null; } };
  }, [isPlaying, finalBars, offsetSec, resetAnchor, findBarIndexByTime]);

  // Lookahead scheduler
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const startScheduler = () => {
      if (schedulerTimerRef.current !== null) return;
      schedulerTimerRef.current = window.setInterval(tick, 16);
    };
    const stopScheduler = () => {
      if (schedulerTimerRef.current !== null) {
        window.clearInterval(schedulerTimerRef.current);
        schedulerTimerRef.current = null;
      }
    };
    const tick = () => {
      if (!audioRef.current || audioRef.current.paused) return;
      if (!autoChord) return;
      if (!anchorRef.current) { resetAnchor(); if (!anchorRef.current) return; }
      const LOOKAHEAD_SEC = 0.2;
      const ctxNow = getAudioContextTime();
      if (ctxNow == null) return;
      let tBrowser = anchorRef.current.audio + (ctxNow - anchorRef.current.ctx);
      const drift = tBrowser - (audioRef.current?.currentTime ?? tBrowser);
      if (Math.abs(drift) > 0.03) { anchorRef.current = { audio: audioRef.current.currentTime, ctx: ctxNow }; tBrowser = audioRef.current.currentTime; }
      const tAnalysis = tBrowser - offsetSec;
      if (schedulingBarRef.current < 0 || schedulingBarRef.current >= finalBars.length) {
        const cur = findBarIndexByTime(tAnalysis, finalBars);
        schedulingBarRef.current = Math.max(0, cur);
        scheduledUpToRef.current = schedulingBarRef.current - 1;
      }
      while (schedulingBarRef.current < finalBars.length) {
        const barIdx = schedulingBarRef.current;
        if (barIdx <= scheduledUpToRef.current) { schedulingBarRef.current++; continue; }
        const bar = finalBars[barIdx];
        if (!bar) break;
        const barStart = Number(bar.mapped_start_sec ?? bar.start_sec);
        const barEnd = Number(bar.mapped_end_sec ?? bar.end_sec);
        if (!Number.isFinite(barStart) || !Number.isFinite(barEnd) || barEnd < barStart) { scheduledUpToRef.current = barIdx; schedulingBarRef.current++; continue; }
        const delta = barStart - tAnalysis;
        if (delta > LOOKAHEAD_SEC) break;
        if (delta < -0.1) { scheduledUpToRef.current = barIdx; schedulingBarRef.current++; continue; }
        const whenSec = Math.max(ctxNow, ctxNow + delta);
        const frets = bar?.tab?.frets;
        if (frets) { scheduleStrumPattern(frets, whenSec, barEnd - barStart).catch(console.error); }
        scheduledUpToRef.current = barIdx;
        lastScheduledBarRef.current = barIdx;
        schedulingBarRef.current++;
      }
    };
    startScheduler();
    return () => stopScheduler();
  }, [autoChord, finalBars, offsetSec, resetAnchor, findBarIndexByTime]);

  // Volume init
  useEffect(() => { setGuitarSoundVolume(chordVolume); }, []); // eslint-disable-line

  // Bar click handler
  const handleBarClick = useCallback((barIndex: number) => {
    const bar = finalBars[barIndex];
    if (!bar || !audioRef.current) return;
    const frets = bar.tab?.frets;
    if (frets) playChordFromTabWithSoundFont(frets).catch(console.error);
    const start = Number(bar.mapped_start_sec ?? bar.start_sec);
    if (!Number.isFinite(start)) return;
    const targetTime = Math.max(0, start + offsetSec);
    const dur = Number(audioRef.current.duration);
    audioRef.current.currentTime = Number.isFinite(dur) && dur > 0 ? Math.min(targetTime, dur - 0.001) : targetTime;
    lastPlayedBarRef.current = -1;
    resetAnchor();
    schedulingBarRef.current = barIndex;
    scheduledUpToRef.current = barIndex - 1;
    // DOM highlight update
    const prevIdx = currentBarIndexRef.current;
    if (prevIdx >= 0) {
      document.querySelector(`[data-bar-index="${prevIdx}"].bar-grid-item`)?.classList.remove('bar-highlighted');
      barRefs.current[prevIdx]?.classList.remove('bar-highlighted');
    }
    document.querySelector(`[data-bar-index="${barIndex}"].bar-grid-item`)?.classList.add('bar-highlighted');
    barRefs.current[barIndex]?.classList.add('bar-highlighted');
    currentBarIndexRef.current = barIndex;
    audioRef.current.play();
  }, [finalBars, offsetSec, resetAnchor]);

  // Chord override handler
  const handleChordChange = useCallback((barIndex: number, chord: string, frets: string[]) => {
    setChordOverrides(prev => {
      const next = new Map(prev);
      next.set(barIndex, { chord, frets });
      return next;
    });
    setEditingBarIndex(null);
  }, []);

  // Tap alignment handler
  const handleTap = useCallback((audioTimeSec: number) => {
    const nearestIdx = addCheckpoint(audioTimeSec, mappedBars);
    setTapFeedback(true);
    setTimeout(() => setTapFeedback(false), 400);
    return nearestIdx;
  }, [addCheckpoint, mappedBars]);

  const formatDuration = (s: number) => { const m = Math.floor(s / 60); return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`; };

  if (!result) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', position: 'relative' }}>
      {/* === 隠しオーディオ要素（HTMLAudioElement経由の制御） === */}
      {audioUrl && (
        <audio ref={audioCallbackRef} src={audioUrl} preload="auto" style={{ display: 'none' }} />
      )}

      {/* ===== ヘッダー統計カード ===== */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem' }}>
        {[
          { label: 'BPM', value: result.bpm },
          { label: 'Key', value: result.key || '—' },
          { label: 'Bars', value: finalBars.length },
          { label: 'Duration', value: formatDuration(result.duration_sec) },
        ].map(({ label, value }) => (
          <div key={label} className="ws-stat-card">
            <div className="ws-stat-label">{label}</div>
            <div className="ws-stat-value" translate={label === 'Key' ? 'no' : undefined}>{value}</div>
          </div>
        ))}
      </div>

      {/* ===== 波形表示 ===== */}
      {audioUrl && (
        <div className="ws-panel" style={{ padding: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
            <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', fontWeight: 700 }}>Waveform</span>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              {/* アライメントモードトグル */}
              <button
                onClick={() => setShowAlignMode(v => !v)}
                style={{
                  padding: '0.25rem 0.75rem',
                  borderRadius: 999,
                  border: `1px solid ${showAlignMode ? 'rgba(79,247,207,0.5)' : 'var(--border)'}`,
                  background: showAlignMode ? 'rgba(79,247,207,0.1)' : 'var(--bg-elevated)',
                  color: showAlignMode ? '#4ff7cf' : 'var(--text-muted)',
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
              >
                {showAlignMode ? '⬤ Tap Align ON' : 'Tap Align'}
              </button>
              {checkpoints.length > 0 && (
                <button onClick={resetCheckpoints} style={{ padding: '0.25rem 0.5rem', borderRadius: 999, border: '1px solid var(--border)', background: 'none', color: 'var(--text-muted)', fontSize: '0.65rem', cursor: 'pointer' }}>
                  リセット ({checkpoints.length})
                </button>
              )}
            </div>
          </div>
          <WaveformDisplay
            audioUrl={audioUrl}
            bars={finalBars}
            currentBarIndex={currentBarIndexRef.current}
            mediaElement={audioEl}
            onTap={showAlignMode ? handleTap : undefined}
          />
          {tapFeedback && (
            <div style={{ fontSize: '0.7rem', color: '#4ff7cf', textAlign: 'center', marginTop: '0.4rem', animation: 'fadeIn 0.2s ease' }}>
              ✓ アライメントポイントを設定しました
            </div>
          )}
        </div>
      )}

      {/* ===== コントロールバー ===== */}
      <div className="ws-panel" style={{ padding: '0.75rem 1.25rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '1rem' }}>
          {/* Auto Chord */}
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', userSelect: 'none' }}>
            <input
              type="checkbox"
              checked={autoChord}
              onChange={e => setAutoChord(e.target.checked)}
              style={{ width: 14, height: 14, accentColor: 'var(--accent)' }}
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Auto Chord</span>
          </label>

          {/* Volume */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Vol</span>
            <input
              type="range" min={0} max={3} step={0.01} value={chordVolume}
              className="ws-slider" style={{ width: 80 }}
              onChange={e => { const v = Number(e.target.value); setChordVolumeState(v); setGuitarSoundVolume(v); }}
            />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', minWidth: 36 }}>{Math.round(chordVolume * 100)}%</span>
          </div>

          {/* Offset */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Offset</span>
            <button onClick={() => setOffsetSec(p => Math.max(-2.0, Number((p - 0.1).toFixed(1))))}
              style={{ width: 22, height: 22, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-elevated)', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.9rem', lineHeight: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>−</button>
            <span style={{ fontVariantNumeric: 'tabular-nums', fontSize: '0.75rem', color: 'var(--text-secondary)', minWidth: 44, textAlign: 'center' }}>
              {`${offsetSec > 0 ? '+' : ''}${offsetSec.toFixed(1)}s`}
            </span>
            <button onClick={() => setOffsetSec(p => Math.min(2.0, Number((p + 0.1).toFixed(1))))}
              style={{ width: 22, height: 22, borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-elevated)', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.9rem', lineHeight: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>+</button>
          </div>

          <div style={{ flex: 1 }} />

          {/* Export toggle */}
          <button
            onClick={() => setShowExport(v => !v)}
            style={{
              padding: '0.35rem 0.9rem',
              borderRadius: 999,
              border: `1px solid ${showExport ? 'var(--accent)' : 'var(--border)'}`,
              background: showExport ? 'rgba(124,110,245,0.15)' : 'var(--bg-elevated)',
              color: showExport ? 'var(--accent)' : 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            ↗ Export
          </button>

          {/* Chord overrides count badge */}
          {chordOverrides.size > 0 && (
            <div className="ws-badge accent">{chordOverrides.size} 個編集</div>
          )}
        </div>
      </div>

      {/* ===== エクスポートパネル ===== */}
      {showExport && (
        <ExportPanel
          result={result}
          bars={finalBars.map(b => ({
            ...b,
            start_sec: b.mapped_start_sec ?? b.start_sec,
            end_sec: b.mapped_end_sec ?? b.end_sec,
          }))}
        />
      )}

      {/* ===== コード進行グリッド ===== */}
      <div className="ws-panel" style={{ padding: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h3 style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', fontWeight: 700 }}>
            Chord Progression
          </h3>
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>タップ → 編集 / シーク</span>
        </div>

        {/* コードグリッド */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.5rem' }}>
          {finalBars.map((bar, idx) => (
            <ChordCell
              key={`${idx}-${bar.bar}-${bar.chord}`}
              bar={bar}
              barIndex={idx}
              diatonicChords={diatonicChords}
              onChordChange={handleChordChange}
              onBarClick={handleBarClick}
              forwardRef={el => { barRefs.current[idx] = el; }}
              isEditing={editingBarIndex === idx}
              onEditOpen={(barIndex, rect) => {
                setEditingBarIndex(barIndex);
                setEditAnchorRect(rect);
              }}
            />
          ))}
        </div>
      </div>

      {/* ===== コード編集パレット（ポップオーバー） ===== */}
      {editingBarIndex !== null && (
        <div
          style={{
            position: 'fixed',
            top: editAnchorRect ? Math.min(editAnchorRect.bottom + 8, window.innerHeight - 320) : '50%',
            left: editAnchorRect ? Math.max(8, Math.min(editAnchorRect.left, window.innerWidth - 420)) : '50%',
            transform: editAnchorRect ? undefined : 'translate(-50%, -50%)',
            zIndex: 1000,
          }}
        >
          <ChordPalette
            currentChord={finalBars[editingBarIndex]?.chord ?? ''}
            barIndex={editingBarIndex}
            diatonicChords={diatonicChords}
            onChordSelect={handleChordChange}
            onClose={() => setEditingBarIndex(null)}
          />
        </div>
      )}

      {/* パレット表示中のオーバーレイ */}
      {editingBarIndex !== null && (
        <div
          style={{ position: 'fixed', inset: 0, zIndex: 999 }}
          onClick={() => setEditingBarIndex(null)}
        />
      )}
    </div>
  );
}
