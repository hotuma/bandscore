'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Loader2 } from 'lucide-react';
import ResultDisplay from "../../components/ResultDisplay";
import { AnalysisResult, analyzeAudio } from "../../lib/api";

type AppStatus = 'idle' | 'uploading' | 'analyzing' | 'ready' | 'error';
type ErrorState = {
    code: string;
    message: string;
};

export default function PreviewPage() {
    const router = useRouter();
    const [file, setFile] = useState<File | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [status, setStatus] = useState<AppStatus>('idle');
    const [error, setError] = useState<ErrorState | null>(null);
    const [progress, setProgress] = useState<number>(0);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const abortPollingRef = useRef<AbortController | null>(null);
    const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
    useEffect(() => {
        return () => {
            if (audioUrl) URL.revokeObjectURL(audioUrl);
        };
    }, [audioUrl]);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            if (selectedFile.size > MAX_FILE_SIZE) {
                setError({ code: 'FILE_TOO_LARGE', message: 'Maximum file size is 20MB.' });
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

    const pollJob = async (jobId: string, signal: AbortSignal, submittedAt: number) => {
        const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
        let lastProgress = -1;
        let lastUpdateTime = Date.now();

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

                setProgress(p);

                // Timeout checks
                if (!data.started_at && (Date.now() - submittedAt > 15000)) throw new Error("JOB_NOT_STARTED");

                if (p > lastProgress) {
                    lastProgress = p;
                    lastUpdateTime = Date.now();
                } else if (Date.now() - lastUpdateTime > 30000) {
                    throw new Error("JOB_STALLED");
                }

                if (jobStatus === "error") throw new Error(data.error || "ANALYSIS_FAILED_BG");

                if (jobStatus === "done") {
                    const r = await fetch(`${base}/analyze/result/${jobId}`, { signal });
                    if (!r.ok) throw new Error("RESULT_FETCH_FAILED");
                    const resultData: AnalysisResult = await r.json();
                    setResult(resultData);
                    setStatus('ready');
                    return;
                }
            } catch (e: any) {
                if (signal.aborted || e.name === "AbortError") return;
                console.error("Polling error:", e);
                setError({
                    code: 'ANALYSIS_ERROR',
                    message: typeof e.message === 'string' ? e.message : 'Analysis failed during processing.'
                });
                setStatus('error');
                return;
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
            // Initiate Analysis
            // analyzeAudio calls /analyze/preview internally if mode is PREVIEW
            // and returns { job_id: "..." } technically
            const response: any = await analyzeAudio(file, 'PREVIEW');

            if (response.job_id) {
                pollJob(response.job_id, controller.signal, Date.now());
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

                {/* UPLOAD SECTION */}
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
                                    <p className="text-sm text-neutral-500">MP3, WAV, M4A (Max 20MB)</p>
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
                                        setFile(null);
                                        setResult(null);
                                        if (audioUrl) URL.revokeObjectURL(audioUrl);
                                        setAudioUrl(null);
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
                        <div>
                            <h3 className="font-bold text-red-400 text-sm tracking-wide">{error.code}</h3>
                            <p className="text-sm mt-1">{error.message}</p>
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
                                <p className="text-xs text-neutral-400 uppercase tracking-wider mb-1">BPM</p>
                                <p className="text-2xl font-mono text-white">{result.bpm}</p>
                            </div>
                            <div className="bg-neutral-800 p-4 rounded-lg">
                                <p className="text-xs text-neutral-400 uppercase tracking-wider mb-1">Key</p>
                                <p className="text-2xl font-mono text-white">{result.key || 'Unknown'}</p>
                            </div>
                            <div className="bg-neutral-800 p-4 rounded-lg">
                                <p className="text-xs text-neutral-400 uppercase tracking-wider mb-1">Chords</p>
                                <p className="text-sm text-neutral-400">Preview up to 60 seconds.</p>
                            </div>
                        </div>

                        {(result?.bars?.length ?? 0) > 0 ? (
                            <div className="mt-8">
                                <ResultDisplay result={result} audioUrl={audioUrl} />
                            </div>
                        ) : (
                            <div className="text-center py-12 text-neutral-400">
                                解析結果が空でした。別の音源でお試しください。
                            </div>
                        )}

                        <div className="mt-8 pt-8 border-t border-neutral-800 text-center">
                            <p className="text-neutral-400 mb-4">Want to see the full chords and export features?</p>
                            <button
                                onClick={() => router.push('/early-access')}
                                className="px-8 py-3 bg-gradient-to-r from-teal-500 to-blue-500 hover:from-teal-400 hover:to-blue-400 text-white font-bold rounded-lg shadow-lg hover:shadow-teal-500/20 transition-all"
                            >
                                Go to Early Access
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
