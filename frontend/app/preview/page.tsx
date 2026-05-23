'use client';

import React, { useState, useRef, useEffect } from 'react';

import { Loader2 } from 'lucide-react';
import { AnalysisResult, analyzeAudio } from "../../lib/api";
import ResultDisplay from "../../components/ResultDisplay";
import { extractVideoId } from "../../lib/youtube";

type AppStatus = 'idle' | 'uploading' | 'downloading' | 'analyzing' | 'ready' | 'error';
type InputMode = 'file' | 'url';
type ErrorState = {
    code: string;
    message: string;
};

export default function PreviewPage() {
    useEffect(() => {
        console.log("PREVIEW PAGE LOADED - NEW VERSION", { updated: new Date().toISOString() });
    }, []);
    const [inputMode, setInputMode] = useState<InputMode>('file');
    const [youtubeUrl, setYoutubeUrl] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [status, setStatus] = useState<AppStatus>('idle');
    const [error, setError] = useState<ErrorState | null>(null);
    const [progress, setProgress] = useState<number>(0);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const abortPollingRef = useRef<AbortController | null>(null);
    const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15MB (Lower limit for mobile stability)

    useEffect(() => {
        return () => {
            if (audioUrl && audioUrl.startsWith('blob:')) URL.revokeObjectURL(audioUrl);
        };
    }, [audioUrl]);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            if (selectedFile.size > MAX_FILE_SIZE) {
                setError({ code: 'FILE_TOO_LARGE', message: 'Maximum file size is 15MB.' });
                return;
            }
            setFile(selectedFile);
            if (audioUrl) URL.revokeObjectURL(audioUrl);
            setAudioUrl(URL.createObjectURL(selectedFile));
            setError(null);
            setStatus('idle');
            setResult(null);
            setProgress(0);
        }
    };

    const pollJob = async (jobId: string, signal: AbortSignal) => {
        const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
        let lastProgress = -1;
        let fetchFailCount = 0;
        const STALL_SEC = 60;
        const MAX_FETCH_FAIL = 5;

        while (true) {
            if (signal.aborted) return;
            await new Promise(r => setTimeout(r, 1000));
            if (signal.aborted) return;

            try {
                const s = await fetch(`${base}/analyze/status/${jobId}`, { signal });
                if (s.status === 404) throw new Error("JOB_LOST");

                const data = await s.json();
                const jobStatus = data.status;
                const p = typeof data.progress === "number" ? data.progress : 0;
                console.log("[poll] status", { jobId, jobStatus, p });

                // Status fetch success - but wait to reset fail count until logical success confirm

                if (p !== lastProgress) {
                    lastProgress = p;
                }

                // Watchdog: heartbeat check (more robust than progress)
                // data.updated_at is in seconds (Python time.time())
                const hbSec = data.updated_at ?? data.submitted_at;
                if (!hbSec) throw new Error("JOB_NO_HEARTBEAT");

                const stalled = (Date.now() - hbSec * 1000) > STALL_SEC * 1000;

                if (stalled && jobStatus === "analyzing") {
                    throw new Error("JOB_STALLED");
                }

                // started_at is optional; heartbeat (updated_at/submitted_at) is the source of truth

                if (jobStatus === "error") throw new Error(data.error || "ANALYSIS_FAILED_BG");

                if (jobStatus === "done") {
                    const r = await fetch(`${base}/analyze/result/${jobId}`, { signal });
                    console.log("[poll] result fetch", { ok: r.ok, status: r.status });
                    if (!r.ok) throw new Error("RESULT_FETCH_FAILED");
                    const resultData: AnalysisResult = await r.json();
                    console.log("ANALYZE RESULT", resultData);
                    fetchFailCount = 0; // complete success
                    // audio_url 正規化（YouTube解析時にバックエンドURLを絶対URLへ変換）
                    if (resultData.audio_url) {
                        const normalized = resultData.audio_url.startsWith('http')
                            ? resultData.audio_url
                            : `${base}${resultData.audio_url}`;
                        setAudioUrl(normalized);
                    }
                    setProgress(100);
                    setResult(resultData);
                    setStatus('ready');
                    return;
                }
                fetchFailCount = 0; // Valid status update received
                setProgress(p);
            } catch (e: any) {
                if (signal.aborted || e.name === "AbortError") return;
                console.error("Polling error:", e);

                fetchFailCount += 1;
                if (fetchFailCount >= MAX_FETCH_FAIL) {
                    setError({
                        code: 'NETWORK_UNSTABLE',
                        message: 'Network or backend is unstable. Please retry.'
                    });
                    setStatus('error');
                    return;
                }

                if (e?.message === "JOB_LOST") {
                    setError({ code: 'JOB_LOST', message: 'Job was lost. Please retry.' });
                    setStatus('error');
                    return;
                }
                if (e?.message === "JOB_STALLED") {
                    setError({ code: 'JOB_STALLED', message: 'Analysis stalled. Please retry.' });
                    setStatus('error');
                    return;
                }
                // otherwise keep looping a bit (transient)
            }
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;

        if (abortPollingRef.current) abortPollingRef.current.abort();
        const controller = new AbortController();
        abortPollingRef.current = controller;

        setStatus('analyzing');
        setError(null);
        setProgress(0);

        try {
            // 1. Pre-load check (Mobile/iCloud optimization)
            // Read to array buffer to ensure file is fully local before POST
            // This prevents "upload stall" when device tries to stream from cloud
            const buffer = await file.arrayBuffer();
            console.log("[upload] preloaded bytes", buffer.byteLength);

            const safeFile = new File([buffer], file.name, {
                type: file.type || 'audio/mpeg',
                lastModified: file.lastModified,
            });

            // INIT auto-retry (avoid user churn)
            const MAX_INIT_RETRY = 3;
            let lastErr: any = null;
            let response: any = null;

            for (let attempt = 1; attempt <= MAX_INIT_RETRY; attempt++) {
                try {
                    console.log(`[upload] start analyzeAudio attempt ${attempt}`);
                    // initiate analysis with 25s timeout (Fail fast -> Retry)
                    response = await analyzeAudio(safeFile, 'PREVIEW', {
                        signal: controller.signal,
                        timeoutMs: 25000,
                    });
                    console.log("[upload] analyzeAudio returned", response);

                    if (response?.job_id) break;
                    throw new Error("No Job ID returned");
                } catch (e: any) {
                    console.warn(`Init attempt ${attempt} failed:`, e);
                    lastErr = e;
                    // backoff: 0.8s, 1.6s, 3.2s
                    const delayMs = 800 * Math.pow(2, attempt - 1);
                    await new Promise(r => setTimeout(r, delayMs));
                }
            }

            if (!response?.job_id) throw lastErr || new Error("INIT_FAILED");

            if (response.job_id) {
                pollJob(response.job_id, controller.signal);
            } else {
                throw new Error("No Job ID returned");
            }

        } catch (err: any) {
            console.error('Analysis failed:', err);
            setError({
                code: err.code || 'INIT_FAILED',
                message: err.message || 'Failed to start analysis.'
            });
            setStatus('error');
        }
    };

    const handleAnalyzeUrl = async () => {
        const trimmed = youtubeUrl.trim();
        if (!trimmed) return;

        const videoId = extractVideoId(trimmed);
        if (!videoId) {
            setError({ code: 'INVALID_URL', message: 'Invalid YouTube URL. Use https://www.youtube.com/watch?v=... format.' });
            return;
        }

        if (abortPollingRef.current) abortPollingRef.current.abort();
        const controller = new AbortController();
        abortPollingRef.current = controller;

        setStatus('downloading');
        setProgress(0);
        setError(null);
        setResult(null);
        setAudioUrl(null);

        try {
            const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
            const formData = new FormData();
            formData.append('url', trimmed);
            const res = await fetch(`${base}/analyze/url/preview`, {
                method: 'POST',
                body: formData,
                signal: controller.signal,
            });

            if (!res.ok) {
                const errorData = await res.json().catch(() => ({}));
                const msg = errorData.detail || `Status ${res.status}`;
                if (res.status === 429 || msg.includes("429") || msg.includes("Rate Limit")) {
                    throw { code: 'YOUTUBE_RATE_LIMIT', message: 'YouTube rate limit reached. Please try again in a few minutes.' };
                } else if (res.status === 403 || msg.includes("403") || msg.includes("Access Denied")) {
                    throw { code: 'YOUTUBE_ACCESS_DENIED', message: 'Cannot access this video. Try another video or upload an MP3 file instead.' };
                }
                throw { code: `HTTP_${res.status}`, message: msg };
            }

            const { job_id } = await res.json();
            console.log("URL preview job started:", job_id);

            setStatus('analyzing');
            pollJob(job_id, controller.signal);

        } catch (err: any) {
            if (err.name === "AbortError") return;
            console.error('URL submission failed:', err);
            if (err.code && err.message) {
                setError(err);
            } else {
                setError({ code: 'SUBMISSION_FAILED', message: err.message || 'Failed to start URL analysis.' });
            }
            setStatus('error');
        }
    };

    return (
        <div className="min-h-screen bg-neutral-950 text-white p-8 font-sans">
            <div className="max-w-4xl mx-auto space-y-8">

                {/* HEADER */}
                <header className="border-b border-neutral-800 pb-6 relative">
                    <div className="absolute top-0 right-0 bg-yellow-500/10 text-yellow-500 border border-yellow-500/50 px-3 py-1 rounded text-xs font-bold tracking-wider uppercase">
                        Development Preview
                    </div>
                    <h1 className="text-3xl font-bold text-neutral-200">
                        BandScore Preview
                    </h1>
                    <p className="text-neutral-400 mt-2 text-sm max-w-2xl">
                        Verify the analysis engine on your own files. <br />
                        <span className="text-neutral-500">Note: Preview mode is limited to 60 seconds and does not include chord export.</span>
                    </p>
                </header>

                {/* INPUT MODE TABS */}
                <div className="flex border-b border-neutral-800">
                    <button
                        onClick={() => {
                            if (inputMode === 'url') {
                                setInputMode('file');
                                setError(null);
                                if (status !== 'ready') { setStatus('idle'); setProgress(0); }
                            }
                        }}
                        className={`px-6 py-3 text-sm font-medium transition-colors ${inputMode === 'file' ? 'text-yellow-500 border-b-2 border-yellow-500' : 'text-neutral-500 hover:text-neutral-300'}`}
                    >
                        File Upload
                    </button>
                    <button
                        onClick={() => {
                            if (inputMode === 'file') {
                                setInputMode('url');
                                setError(null);
                                if (status !== 'ready') { setStatus('idle'); setProgress(0); }
                            }
                        }}
                        className={`px-6 py-3 text-sm font-medium transition-colors ${inputMode === 'url' ? 'text-yellow-500 border-b-2 border-yellow-500' : 'text-neutral-500 hover:text-neutral-300'}`}
                    >
                        YouTube URL
                    </button>
                </div>

                {/* UPLOAD SECTION */}
                {inputMode === 'file' && (
                <section className={`transition-opacity duration-500 ${status === 'analyzing' ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
                    <div className="bg-neutral-900/50 rounded-xl border border-neutral-800 p-8 text-center border-dashed hover:border-teal-500/50 transition-colors">
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".mp3,.wav,.m4a,audio/*"
                            onChange={handleFileSelect}
                            className="hidden"
                        />

                        {!file ? (
                            <div className="space-y-4 cursor-pointer" onClick={() => fileInputRef.current?.click()}>
                                <div className="mx-auto w-16 h-16 bg-neutral-800 rounded-full flex items-center justify-center text-2xl text-neutral-400">
                                    📂
                                </div>
                                <div>
                                    <p className="text-lg font-medium text-neutral-200">Select Audio File to Preview</p>
                                    <p className="text-sm text-neutral-500">MP3, WAV, M4A (Max 15MB)</p>
                                </div>
                            </div>
                        ) : (
                            <div className="flex items-center justify-between bg-neutral-800 rounded-lg p-4 max-w-md mx-auto">
                                <div className="flex items-center space-x-3 overflow-hidden">
                                    <div className="text-2xl">🎵</div>
                                    <div className="truncate text-left">
                                        <p className="text-sm font-medium text-white truncate">{file.name}</p>
                                        <p className="text-xs text-neutral-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                    </div>
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        if (abortPollingRef.current) abortPollingRef.current.abort();
                                        setFile(null);
                                        setResult(null);
                                        if (audioUrl) URL.revokeObjectURL(audioUrl);
                                        setAudioUrl(null);
                                        setStatus('idle');
                                        setProgress(0);
                                        setError(null);
                                    }}
                                    className="p-2 hover:bg-neutral-700 rounded-full text-neutral-400 hover:text-red-400 transition-colors"
                                >
                                    ✕
                                </button>
                            </div>
                        )}

                        {file && status !== 'analyzing' && status !== 'ready' && (
                            <button
                                onClick={handleAnalyze}
                                className="mt-6 px-8 py-3 bg-neutral-700 hover:bg-neutral-600 text-white font-medium rounded-lg shadow-lg transition-all"
                            >
                                Run Preview Analysis
                            </button>
                        )}
                    </div>
                </section>
                )}

                {/* YOUTUBE URL INPUT */}
                {inputMode === 'url' && (
                <section className={`transition-opacity duration-500 ${(status === 'downloading' || status === 'analyzing') ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
                    <div className="bg-neutral-900/50 rounded-xl border border-neutral-800 p-8 border-dashed hover:border-yellow-500/50 transition-colors">
                        <div className="max-w-lg mx-auto space-y-4">
                            <div className="flex gap-3">
                                <input
                                    type="url"
                                    placeholder="https://www.youtube.com/watch?v=..."
                                    value={youtubeUrl}
                                    onChange={(e) => setYoutubeUrl(e.target.value)}
                                    className="flex-1 px-4 py-3 bg-neutral-800 border border-neutral-700 rounded-lg text-white placeholder-neutral-500 focus:outline-none focus:border-yellow-500 transition-colors"
                                />
                            </div>
                            {status !== 'downloading' && status !== 'analyzing' && status !== 'ready' && (
                                <button
                                    onClick={handleAnalyzeUrl}
                                    disabled={!youtubeUrl.trim()}
                                    className="w-full px-8 py-3 bg-neutral-700 hover:bg-neutral-600 text-white font-medium rounded-lg shadow-lg transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Run Preview Analysis
                                </button>
                            )}
                        </div>
                    </div>
                </section>
                )}

                {/* DOWNLOADING STATE (YouTube) */}
                {status === 'downloading' && (
                    <div className="text-center py-12">
                        <Loader2 className="h-12 w-12 animate-spin text-yellow-500 mx-auto mb-4" />
                        <h3 className="text-xl font-bold mb-2">Downloading from YouTube...</h3>
                        <p className="text-gray-400">
                            This may take 10-30 seconds depending on the video.
                        </p>
                    </div>
                )}

                {/* LOADING STATE */}
                {status === 'analyzing' && (
                    <div className="text-center py-12">
                        <Loader2 className="h-12 w-12 animate-spin text-yellow-500 mx-auto mb-4" />
                        <h3 className="text-xl font-bold mb-2">Analyzing (Preview Mode)...</h3>
                        <p className="text-gray-400 mb-6">
                            Processing up to 60 seconds of audio.
                        </p>
                        {/* Progress Bar */}
                        <div className="max-w-md mx-auto">
                            <div className="flex items-center justify-between text-xs text-gray-400 mb-2 font-mono">
                                <span>PROCESSING</span>
                                <span>{Math.floor(progress)}%</span>
                            </div>
                            <div className="h-1.5 w-full bg-neutral-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-yellow-500 transition-all duration-300 ease-out"
                                    style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
                                />
                            </div>
                        </div>
                    </div>
                )}

                {/* ERROR STATE */}
                {error && (
                    <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 flex items-start space-x-3 text-red-200">
                        <span className="text-xl">⚠️</span>
                        <div className="flex-1">
                            <h3 className="font-bold text-red-400 text-sm tracking-wide">{error.code}</h3>
                            <p className="text-sm mt-1 mb-3">{error.message}</p>

                            {/* Reload Hint */}
                            <div className="text-xs text-red-200/80 bg-red-950/30 p-3 rounded mb-3 border border-red-500/20">
                                <p className="font-semibold text-red-100 mb-1">Analysis Stuck?</p>
                                <ul className="list-disc pl-4 space-y-1 mb-2">
                                    <li>Try <span className="font-bold text-white">Retry Analysis</span> first.</li>
                                    <li>If it fails again, please <span className="font-bold text-white">Reload Page</span>.</li>
                                </ul>
                                <p className="opacity-75">
                                    (Mobile connections may pause uploads in background)
                                </p>
                            </div>

                            <div className="flex gap-3">
                                <button
                                    onClick={inputMode === 'url' ? handleAnalyzeUrl : handleAnalyze}
                                    className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-100 text-xs font-bold uppercase tracking-wider rounded transition-colors"
                                >
                                    Retry Analysis
                                </button>
                                <button
                                    onClick={() => window.location.reload()}
                                    className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 text-xs font-bold uppercase tracking-wider rounded transition-colors"
                                >
                                    Reload Page
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {/* PREVIEW RESULT */}
                {status === 'ready' && result && (
                    <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className="flex items-center justify-between mb-8">
                            <h2 className="text-xl font-bold text-white">Analysis Result <span className="text-sm font-normal text-yellow-500 ml-2">(Preview Limit Applied)</span></h2>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                            <div className="bg-neutral-800 p-4 rounded-lg">
                                <p className="text-xs text-neutral-400 uppercase tracking-wider mb-1">Analyzed Duration</p>
                                <p className="text-2xl font-mono text-white">{result.analyzed_duration_sec}s <span className="text-sm text-neutral-500">/ 60s max</span></p>
                            </div>
                            <div className="bg-neutral-800 p-4 rounded-lg">
                                <p className="text-xs text-neutral-400 uppercase tracking-wider mb-1">Key</p>
                                <p className="text-2xl font-mono text-white">{result.key}</p>
                            </div>
                        </div>

                        {(result?.bars?.length ?? 0) > 0 && audioUrl ? (
                            <div className="mt-4">
                                <ResultDisplay result={result} audioUrl={audioUrl} />
                            </div>
                        ) : (
                            <div className="text-center py-12 text-neutral-400">
                                No chords detected. Please try another file.
                            </div>
                        )}


                    </div>
                )}
            </div>
        </div>
    );
}
