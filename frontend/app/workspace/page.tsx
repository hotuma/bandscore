'use client';

import React, { useState, useCallback, useRef } from 'react';
import { AnalysisResult } from '../../lib/api';
import { extractVideoId } from '../../lib/youtube';
import ResultDisplayV2 from '../../components/ResultDisplayV2';

export default function WorkspacePage() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState<number>(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [inputMode, setInputMode] = useState<'file' | 'url'>('file');
  const [urlInput, setUrlInput] = useState('');
  const [cookiesFile, setCookiesFile] = useState<File | null>(null);
  const [cookiesSelected, setCookiesSelected] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cookiesInputRef = useRef<HTMLInputElement>(null);
  const blobUrlRef = useRef<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('audio/') && !file.name.match(/\.(mp3|m4a|wav|ogg|flac|aac)$/i)) {
      setError('音声ファイルを選択してください（MP3, M4A, WAV, OGG, FLAC, AACに対応）');
      return;
    }

    // 前のジョブをキャンセル
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const { signal } = controller;

    setError(null);
    setResult(null);
    setProgress(0);
    setIsAnalyzing(true);
    setFileName(file.name);

    // 前回のblob URLを解放
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
    }
    const blobUrl = URL.createObjectURL(file);
    blobUrlRef.current = blobUrl;
    setAudioUrl(blobUrl);

    const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
    const submittedAt = Date.now();

    try {
      // Step 1: ジョブ送信
      const formData = new FormData();
      formData.append('file', file);
      formData.append('mode', 'EARLY_ACCESS');
      const res = await fetch(`${base}/analyze`, { method: 'POST', body: formData, signal });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'アップロードに失敗しました');
      }
      const { job_id } = await res.json();
      if (!job_id) throw new Error('サーバーからジョブIDが返されませんでした');

      // Step 2: ポーリング
      let lastProgress = -1;
      let lastUpdateTime = Date.now();

      while (true) {
        if (signal.aborted) return;
        await new Promise(r => setTimeout(r, 1500));
        if (signal.aborted) return;

        const s = await fetch(`${base}/analyze/status/${job_id}`, { signal });
        if (s.status === 404) throw new Error('ジョブが見つかりません。再度お試しください。');
        const data = await s.json();
        const rawP = typeof data.progress === 'number' ? data.progress : 0;
        // Backend returns progress as 0-100, normalize to 0-1 for display
        const p = rawP > 1 ? rawP / 100 : rawP;
        setProgress(p);

        // スタートアップチェック（15秒経っても started_at がなければ異常）
        if (!data.started_at && Date.now() - submittedAt > 15000) {
          throw new Error('サーバーが応答しません。バックエンドが起動しているか確認してください。');
        }

        // ストールチェック（25秒進捗なしでタイムアウト）
        if (p > lastProgress) {
          lastProgress = p;
          lastUpdateTime = Date.now();
        } else if (Date.now() - lastUpdateTime > 120000) {
          throw new Error('解析がタイムアウトしました。再度お試しください。');
        }

        if (data.status === 'error') throw new Error('解析に失敗しました。');
        if (data.status === 'done') {
          const r = await fetch(`${base}/analyze/result/${job_id}`, { signal });
          if (!r.ok) throw new Error('結果の取得に失敗しました。');
          const resultData = await r.json();
          // audio_url の正規化
          if (resultData.audio_url && typeof resultData.audio_url === 'string' && !resultData.audio_url.startsWith('http')) {
            const path = resultData.audio_url.startsWith('/') ? resultData.audio_url : `/${resultData.audio_url}`;
            resultData.audio_url = `${base}${path}`;
          }
          setResult(resultData);
          setProgress(1.0);
          return;
        }
      }
    } catch (err: any) {
      if (signal.aborted || err.name === 'AbortError') return;
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setAudioUrl(null);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragOver(false), []);

  const processUrl = useCallback(async (url: string) => {
    const videoId = extractVideoId(url);
    if (!videoId) {
      setError('有効な YouTube URL を入力してください');
      return;
    }

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const { signal } = controller;

    setError(null);
    setResult(null);
    setProgress(0);
    setIsAnalyzing(true);
    setFileName(`youtu.be/${videoId}`);

    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    setAudioUrl(null);

    const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
    const submittedAt = Date.now();

    try {
      // Step 1: ジョブ送信
      const formData = new FormData();
      formData.append('url', url);
      formData.append('mode', 'EARLY_ACCESS');
      if (cookiesFile) {
        formData.append('cookies', cookiesFile);
      }

      const res = await fetch(`${base}/analyze/url`, { method: 'POST', body: formData, signal });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'URL 解析の送信に失敗しました');
      }
      const { job_id } = await res.json();
      if (!job_id) throw new Error('サーバーからジョブIDが返されませんでした');

      // Step 2: ポーリング
      let lastProgress = -1;
      let lastUpdateTime = Date.now();

      while (true) {
        if (signal.aborted) return;
        await new Promise(r => setTimeout(r, 1500));
        if (signal.aborted) return;

        const s = await fetch(`${base}/analyze/status/${job_id}`, { signal });
        if (s.status === 404) throw new Error('ジョブが見つかりません。再度お試しください。');
        const data = await s.json();
        const rawP = typeof data.progress === 'number' ? data.progress : 0;
        // Backend returns progress as 0-100, normalize to 0-1 for display
        const p = rawP > 1 ? rawP / 100 : rawP;
        setProgress(p);

        if (!data.started_at && Date.now() - submittedAt > 15000) {
          throw new Error('サーバーが応答しません。バックエンドが起動しているか確認してください。');
        }

        if (p > lastProgress) {
          lastProgress = p;
          lastUpdateTime = Date.now();
        } else if (Date.now() - lastUpdateTime > 120000) {
          throw new Error('解析がタイムアウトしました。再度お試しください。');
        }

        if (data.status === 'error') {
          const errorMsg = data.error || '解析に失敗しました。';
          throw new Error(errorMsg);
        }
        if (data.status === 'done') {
          const r = await fetch(`${base}/analyze/result/${job_id}`, { signal });
          if (!r.ok) throw new Error('結果の取得に失敗しました。');
          const resultData = await r.json();
          if (resultData.audio_url && typeof resultData.audio_url === 'string' && !resultData.audio_url.startsWith('http')) {
            const path = resultData.audio_url.startsWith('/') ? resultData.audio_url : `/${resultData.audio_url}`;
            resultData.audio_url = `${base}${path}`;
          }
          setResult(resultData);
          setAudioUrl(resultData.audio_url ?? null);
          setProgress(1.0);
          return;
        }
      }
    } catch (err: any) {
      if (signal.aborted || err?.name === 'AbortError') return;
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  }, [cookiesFile]);

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)', padding: '1.5rem 1rem' }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>

        {/* ===== ヘッダー ===== */}
        <header style={{ marginBottom: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
              <h1 style={{ fontSize: '1.5rem', fontWeight: 900, margin: 0 }}>
                <span className="text-gradient">BandScore</span>
              </h1>
              <div className="ws-badge accent">Workspace</div>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
              クリエイター向けアレンジ空間 — Phase 1
            </p>
          </div>
          {result && (
            <button
              onClick={() => {
                abortControllerRef.current?.abort();
                setResult(null);
                setAudioUrl(null);
                setFileName(null);
                setProgress(0);
              }}
              style={{
                padding: '0.4rem 1rem',
                borderRadius: 999,
                border: '1px solid var(--border)',
                background: 'var(--bg-elevated)',
                color: 'var(--text-muted)',
                fontSize: '0.75rem',
                cursor: 'pointer',
                fontWeight: 600,
                transition: 'all 0.2s ease',
              }}
            >
              ← 新しい音源を読み込む
            </button>
          )}
        </header>

        {/* ===== 入力エリア（結果がない場合のみ表示） ===== */}
        {!result && !isAnalyzing && (
          <div style={{ marginBottom: '1.5rem' }}>
            {/* タブ切り替え */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem' }}>
              {(['file', 'url'] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => setInputMode(mode)}
                  style={{
                    padding: '0.4rem 1.1rem',
                    borderRadius: 999,
                    border: inputMode === mode ? '1px solid var(--accent)' : '1px solid var(--border)',
                    background: inputMode === mode ? 'rgba(var(--accent-rgb, 99,102,241),0.12)' : 'var(--bg-elevated)',
                    color: inputMode === mode ? 'var(--accent)' : 'var(--text-muted)',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {mode === 'file' ? 'ファイル' : 'URL'}
                </button>
              ))}
            </div>

            {/* ファイルタブ */}
            {inputMode === 'file' && (
              <div
                className={`drop-zone${isDragOver ? ' drag-over' : ''}`}
                style={{ padding: '3.5rem 2rem', textAlign: 'center' }}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*,.mp3,.m4a,.wav,.ogg,.flac,.aac"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
                <div style={{ fontSize: '3rem', marginBottom: '1rem', lineHeight: 1 }}>🎵</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                  音源をドロップ または クリックして選択
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                  MP3, M4A, WAV, OGG, FLAC, AAC に対応
                </div>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
                  {['BPM自動検出', 'コード解析', '波形表示', '1ストローク試聴', 'コード編集', 'JSON出力'].map(feat => (
                    <div key={feat} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)' }} />
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontWeight: 600 }}>{feat}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* URL タブ */}
            {inputMode === 'url' && (
              <div className="drop-zone" style={{ padding: '2rem', textAlign: 'left' }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.75rem' }}>
                  YouTube URL を入力
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <input
                    type="text"
                    value={urlInput}
                    onChange={e => setUrlInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') processUrl(urlInput); }}
                    placeholder="https://www.youtube.com/watch?v=..."
                    style={{
                      flex: 1,
                      padding: '0.55rem 0.9rem',
                      borderRadius: 8,
                      border: '1px solid var(--border-bright)',
                      background: 'var(--bg-overlay)',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                      outline: 'none',
                    }}
                  />
                  <button
                    onClick={() => processUrl(urlInput)}
                    disabled={!urlInput.trim()}
                    style={{
                      padding: '0.55rem 1.2rem',
                      borderRadius: 8,
                      border: 'none',
                      background: 'var(--accent)',
                      color: '#fff',
                      fontSize: '0.82rem',
                      fontWeight: 700,
                      cursor: urlInput.trim() ? 'pointer' : 'not-allowed',
                      opacity: urlInput.trim() ? 1 : 0.5,
                      transition: 'opacity 0.15s',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    解析開始
                  </button>
                </div>

                {/* Advanced Options */}
                <button
                  onClick={() => setShowAdvanced(v => !v)}
                  style={{ fontSize: '0.72rem', color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, fontWeight: 600 }}
                >
                  {showAdvanced ? '▾' : '▸'} Advanced Options
                </button>
                {showAdvanced && (
                  <div style={{ marginTop: '0.75rem', padding: '0.75rem', background: 'var(--bg-overlay)', borderRadius: 8, border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                      限定公開動画の場合、YouTube の cookies.txt をアップロードしてください。
                    </div>
                    <input
                      ref={cookiesInputRef}
                      type="file"
                      accept=".txt"
                      style={{ display: 'none' }}
                      onChange={e => setCookiesFile(e.target.files?.[0] ?? null)}
                    />
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <button
                        onClick={() => cookiesInputRef.current?.click()}
                        style={{
                          padding: '0.35rem 0.8rem',
                          borderRadius: 6,
                          border: cookiesSelected ? '2px solid var(--accent)' : '1px solid var(--border-bright)',
                          background: cookiesSelected ? 'rgba(var(--accent-rgb, 99, 102, 241), 0.12)' : 'var(--bg-elevated)',
                          color: cookiesSelected ? '#fff' : 'var(--text-secondary)',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          cursor: 'pointer',
                          transition: 'all 0.2s ease',
                        }}
                      >
                        {cookiesFile ? '選択済み' : 'cookies.txt を選択'}
                      </button>
                      {cookiesFile && (
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{cookiesFile.name}</span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ===== 解析中 ===== */}
        {isAnalyzing && (
          <div className="ws-panel animate-fadeIn" style={{ padding: '3rem 2rem', textAlign: 'center', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
              <div className="animate-spin" style={{ width: 40, height: 40, border: '3px solid var(--border-bright)', borderTopColor: 'var(--accent)', borderRadius: '50%' }} />
            </div>
            <div style={{ fontWeight: 800, fontSize: '1rem', marginBottom: '0.5rem' }}>
              解析中… {Math.round(progress * 100)}%
            </div>
            {/* プログレスバー */}
            <div style={{ width: '100%', maxWidth: 300, margin: '0 auto 0.75rem', height: 4, background: 'var(--bg-overlay)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{
                width: `${Math.round(progress * 100)}%`,
                height: '100%',
                background: 'var(--accent)',
                borderRadius: 2,
                transition: 'width 0.4s ease',
              }} />
            </div>
            {fileName && (
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{fileName}</div>
            )}
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
              BPM検出・コード解析・タブ生成を実行中
            </div>
          </div>
        )}

        {/* ===== エラー ===== */}
        {error && (
          <div className="animate-fadeIn" style={{ padding: '0.75rem 1rem', background: 'rgba(245,86,74,0.1)', border: '1px solid rgba(245,86,74,0.3)', borderRadius: 12, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '1rem' }}>⚠️</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--error)', fontWeight: 600 }}>{error}</span>
          </div>
        )}

        {/* ===== 解析結果（ResultDisplayV2） ===== */}
        {!isAnalyzing && result && (
          <div className="animate-fadeIn">
            <ResultDisplayV2 result={result} audioUrl={audioUrl} />
          </div>
        )}

        {/* ===== フッター ===== */}
        <footer style={{ marginTop: '3rem', textAlign: 'center' }}>
          <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            BandScore Workspace — Phase 1 / Local First Architecture
          </p>
        </footer>
      </div>
    </div>
  );
}
