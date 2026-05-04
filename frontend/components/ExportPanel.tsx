'use client';

import { useState, useCallback } from 'react';
import { Bar, AnalysisResult } from '../lib/api';
import { useExport } from '../hooks/useExport';

interface ExportPanelProps {
  result: AnalysisResult;
  bars: Bar[];
}

export default function ExportPanel({ result, bars }: ExportPanelProps) {
  const { buildChordText, buildExportJSON, copyToClipboard, downloadJSON, downloadText } = useExport();
  const [copiedText, setCopiedText] = useState(false);
  const [copiedJSON, setCopiedJSON] = useState(false);
  const [showPreview, setShowPreview] = useState<'text' | 'json' | null>(null);

  const chordText = buildChordText(bars, 4);
  const jsonData = buildExportJSON(result, bars);
  const jsonText = JSON.stringify(jsonData, null, 2);

  const handleCopyText = useCallback(async () => {
    const ok = await copyToClipboard(chordText);
    if (ok) {
      setCopiedText(true);
      setTimeout(() => setCopiedText(false), 2000);
    }
  }, [chordText, copyToClipboard]);

  const handleCopyJSON = useCallback(async () => {
    const ok = await copyToClipboard(jsonText);
    if (ok) {
      setCopiedJSON(true);
      setTimeout(() => setCopiedJSON(false), 2000);
    }
  }, [jsonText, copyToClipboard]);

  const handleDownloadText = useCallback(() => {
    downloadText(chordText, `chords-${result.key}-${result.bpm}bpm.txt`);
  }, [chordText, downloadText, result]);

  const handleDownloadJSON = useCallback(() => {
    downloadJSON(jsonData, `bandscore-${result.key}-${result.bpm}bpm.json`);
  }, [jsonData, downloadJSON, result]);

  return (
    <div className="export-panel animate-fadeIn">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <h3 style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', fontWeight: 700 }}>
          Export
        </h3>
        <div className="ws-badge accent">Phase 1</div>
      </div>

      {/* テキスト出力セクション */}
      <div style={{ marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            コード進行テキスト
          </span>
          <button
            onClick={() => setShowPreview(showPreview === 'text' ? null : 'text')}
            style={{ fontSize: '0.65rem', color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none', textDecoration: 'underline' }}
          >
            {showPreview === 'text' ? '隠す' : 'プレビュー'}
          </button>
        </div>

        {showPreview === 'text' && (
          <div className="export-code" style={{ marginBottom: '0.5rem' }}>
            {chordText}
          </div>
        )}

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={handleCopyText}
            style={{
              flex: 1,
              padding: '0.45rem',
              borderRadius: 8,
              border: '1px solid var(--border)',
              background: copiedText ? 'rgba(34,201,138,0.15)' : 'var(--bg-surface)',
              color: copiedText ? 'var(--success)' : 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            {copiedText ? '✓ コピー完了' : '📋 コピー'}
          </button>
          <button
            onClick={handleDownloadText}
            style={{
              flex: 1,
              padding: '0.45rem',
              borderRadius: 8,
              border: '1px solid var(--border)',
              background: 'var(--bg-surface)',
              color: 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            ↓ .txt
          </button>
        </div>
      </div>

      {/* JSON出力セクション */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            JSON（DAW連携用）
          </span>
          <button
            onClick={() => setShowPreview(showPreview === 'json' ? null : 'json')}
            style={{ fontSize: '0.65rem', color: 'var(--text-muted)', cursor: 'pointer', background: 'none', border: 'none', textDecoration: 'underline' }}
          >
            {showPreview === 'json' ? '隠す' : 'プレビュー'}
          </button>
        </div>

        {showPreview === 'json' && (
          <div className="export-code" style={{ marginBottom: '0.5rem' }}>
            {jsonText.slice(0, 600)}{jsonText.length > 600 ? '\n…（省略）' : ''}
          </div>
        )}

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={handleCopyJSON}
            style={{
              flex: 1,
              padding: '0.45rem',
              borderRadius: 8,
              border: '1px solid var(--border)',
              background: copiedJSON ? 'rgba(34,201,138,0.15)' : 'var(--bg-surface)',
              color: copiedJSON ? 'var(--success)' : 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            {copiedJSON ? '✓ コピー完了' : '📋 コピー'}
          </button>
          <button
            onClick={handleDownloadJSON}
            style={{
              flex: 1,
              padding: '0.45rem',
              borderRadius: 8,
              border: '1px solid var(--border)',
              background: 'var(--bg-surface)',
              color: 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
          >
            ↓ .json
          </button>
        </div>
      </div>
    </div>
  );
}
