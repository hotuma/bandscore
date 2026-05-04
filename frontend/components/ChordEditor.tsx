'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Bar } from '../lib/api';
import { playChordFromTabWithSoundFont } from '../lib/guitarSound';

// 全コードリスト（バックエンドのCHORD_TO_TABと同期）
const ALL_CHORDS = [
  // Major
  'C','C#','D','D#','E','F','F#','G','G#','A','A#','B',
  // Minor
  'Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm',
  // Dominant 7th
  'C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7',
  // Major 7th
  'Cmaj7','C#maj7','Dmaj7','D#maj7','Emaj7','Fmaj7','F#maj7','Gmaj7','G#maj7','Amaj7','A#maj7','Bmaj7',
  // sus4
  'Csus4','C#sus4','Dsus4','D#sus4','Esus4','Fsus4','F#sus4','Gsus4','G#sus4','Asus4','A#sus4','Bsus4',
  // Minor 7th
  'Cm7','C#m7','Dm7','D#m7','Em7','Fm7','F#m7','Gm7','G#m7','Am7','A#m7','Bm7',
];

const CHORD_CATEGORIES: Record<string, string[]> = {
  'Major':  ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'],
  'Minor':  ['Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm'],
  '7th':    ['C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7'],
  'Maj7':   ['Cmaj7','C#maj7','Dmaj7','D#maj7','Emaj7','Fmaj7','F#maj7','Gmaj7','G#maj7','Amaj7','A#maj7','Bmaj7'],
  'sus4':   ['Csus4','C#sus4','Dsus4','D#sus4','Esus4','Fsus4','F#sus4','Gsus4','G#sus4','Asus4','A#sus4','Bsus4'],
  'm7':     ['Cm7','C#m7','Dm7','D#m7','Em7','Fm7','F#m7','Gm7','G#m7','Am7','A#m7','Bm7'],
};

// バックエンドのCHORD_TO_TAB簡易マッピング
const CHORD_TO_FRETS: Record<string, string[]> = {
  'C':['x','3','2','0','1','0'], 'C#':['x','4','6','6','6','4'], 'D':['x','x','0','2','3','2'],
  'D#':['x','6','8','8','8','6'], 'E':['0','2','2','1','0','0'], 'F':['1','3','3','2','1','1'],
  'F#':['2','4','4','3','2','2'], 'G':['3','2','0','0','0','3'], 'G#':['4','6','6','5','4','4'],
  'A':['x','0','2','2','2','0'], 'A#':['x','1','3','3','3','1'], 'B':['x','2','4','4','4','2'],
  'Cm':['x','3','5','5','4','3'], 'C#m':['x','4','6','6','5','4'], 'Dm':['x','x','0','2','3','1'],
  'D#m':['x','6','8','8','7','6'], 'Em':['0','2','2','0','0','0'], 'Fm':['1','3','3','1','1','1'],
  'F#m':['2','4','4','2','2','2'], 'Gm':['3','5','5','3','3','3'], 'G#m':['4','6','6','4','4','4'],
  'Am':['x','0','2','2','1','0'], 'A#m':['x','1','3','3','2','1'], 'Bm':['x','2','4','4','3','2'],
  'C7':['x','3','2','3','1','0'], 'D7':['x','x','0','2','1','2'], 'E7':['0','2','0','1','0','0'],
  'G7':['3','2','0','0','0','1'], 'A7':['x','0','2','0','2','0'], 'B7':['x','2','1','2','0','2'],
  'Cmaj7':['x','3','2','0','0','0'], 'Dmaj7':['x','x','0','2','2','2'], 'Emaj7':['0','2','1','1','0','0'],
  'Fmaj7':['1','3','2','2','1','0'], 'Gmaj7':['3','2','0','0','0','2'], 'Amaj7':['x','0','2','1','2','0'],
  'Csus4':['x','3','3','0','1','1'], 'Dsus4':['x','x','0','2','3','3'], 'Esus4':['0','2','2','2','0','0'],
  'Gsus4':['3','3','0','0','1','3'], 'Asus4':['x','0','2','2','3','0'],
  'Em7':['0','2','0','0','0','0'], 'Am7':['x','0','2','0','1','0'], 'Dm7':['x','x','0','2','1','1'],
  'Bm7':['x','2','0','2','0','2'],
};

interface ChordPaletteProps {
  currentChord: string;
  barIndex: number;
  /** ダイアトニックコードのリスト（キーから計算） */
  diatonicChords?: string[];
  onChordSelect: (barIndex: number, chord: string, frets: string[]) => void;
  onClose: () => void;
  /** パレットを表示する基準位置（ポップオーバー） */
  anchorRect?: DOMRect | null;
}

export function ChordPalette({
  currentChord,
  barIndex,
  diatonicChords = [],
  onChordSelect,
  onClose,
}: ChordPaletteProps) {
  const [activeCategory, setActiveCategory] = useState('Major');
  const palRef = useRef<HTMLDivElement>(null);

  // 外側クリックで閉じる
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (palRef.current && !palRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    setTimeout(() => document.addEventListener('mousedown', handler), 50);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const handleSelect = useCallback(async (chord: string) => {
    const frets = CHORD_TO_FRETS[chord];
    if (frets) {
      // 即時1ストローク再生
      try {
        await playChordFromTabWithSoundFont(frets, { durationSec: 1.5 });
      } catch (e) {
        console.warn('Chord preview failed:', e);
      }
      onChordSelect(barIndex, chord, frets);
    } else {
      // フレット情報がなくても選択だけ通知
      onChordSelect(barIndex, chord, []);
    }
    onClose();
  }, [barIndex, onChordSelect, onClose]);

  const chordsInCategory = CHORD_CATEGORIES[activeCategory] ?? [];
  const diatonicSet = new Set(diatonicChords);

  return (
    <div
      ref={palRef}
      className="chord-palette"
      style={{ padding: '1rem', minWidth: 320, maxWidth: 400, zIndex: 200, position: 'relative' }}
      role="dialog"
      aria-label="コード選択"
    >
      {/* ヘッダー */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 700 }}>
          コードを選択
        </span>
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1rem', lineHeight: 1 }}
          aria-label="閉じる"
        >
          ✕
        </button>
      </div>

      {/* カテゴリタブ */}
      <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.75rem', flexWrap: 'wrap' }}>
        {Object.keys(CHORD_CATEGORIES).map(cat => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            style={{
              padding: '0.2rem 0.6rem',
              borderRadius: 999,
              border: `1px solid ${activeCategory === cat ? 'var(--accent)' : 'var(--border)'}`,
              background: activeCategory === cat ? 'var(--accent)' : 'var(--bg-surface)',
              color: activeCategory === cat ? 'white' : 'var(--text-secondary)',
              fontSize: '0.7rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* コードグリッド */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.4rem' }}>
        {chordsInCategory.map(chord => {
          const isDiatonic = diatonicSet.has(chord);
          const isSelected = chord === currentChord;
          return (
            <button
              key={chord}
              onClick={() => handleSelect(chord)}
              className={`chord-candidate-btn${isDiatonic ? ' diatonic' : ''}${isSelected ? ' selected' : ''}`}
              translate="no"
            >
              {chord}
            </button>
          );
        })}
      </div>

      {/* ダイアトニック凡例 */}
      {diatonicChords.length > 0 && (
        <div style={{ marginTop: '0.75rem', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
          <span style={{ color: '#a89cf7' }}>■</span> ダイアトニックコード
        </div>
      )}
    </div>
  );
}

interface ChordCellProps {
  bar: Bar & { mapped_start_sec?: number; mapped_end_sec?: number };
  barIndex: number;
  diatonicChords?: string[];
  onChordChange: (barIndex: number, chord: string, frets: string[]) => void;
  onBarClick?: (barIndex: number) => void;
  isHighlighted?: boolean;
  forwardRef?: (el: HTMLDivElement | null) => void;
  isEditing?: boolean;
  onEditOpen?: (barIndex: number, rect: DOMRect) => void;
}

/**
 * 小節グリッドの1セル（コードのタップで編集パレットを開く）
 */
export function ChordCell({
  bar,
  barIndex,
  isHighlighted,
  forwardRef,
  onBarClick,
  onEditOpen,
}: ChordCellProps) {
  const cellRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onBarClick?.(barIndex);
    if (onEditOpen && cellRef.current) {
      onEditOpen(barIndex, cellRef.current.getBoundingClientRect());
    }
  };

  return (
    <div
      ref={(el) => {
        (cellRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
        forwardRef?.(el);
      }}
      data-bar-index={barIndex}
      onClick={handleClick}
      className={`bar-grid-item flex flex-col items-center p-2${isHighlighted ? ' bar-highlighted' : ''}`}
    >
      <span className="bar-number" style={{ fontSize: '0.65rem', marginBottom: 2, color: 'var(--text-muted)' }}>
        {bar.bar}
      </span>
      <span className="bar-chord" style={{ fontWeight: 700, fontSize: '0.95rem' }} translate="no">
        {bar.chord}
      </span>
    </div>
  );
}
