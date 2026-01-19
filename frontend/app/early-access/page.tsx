'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Loader2 } from 'lucide-react';
import ResultDisplay from "../../components/ResultDisplay";
import { AnalysisResult } from "../../lib/api";
import { analysisResultToTimedChords } from "../../lib/chordTimeline";

// Types matching Backend Contract


type AppStatus = 'idle' | 'uploading' | 'analyzing' | 'ready' | 'error';

type ErrorState = {
    code: string;
    message: string;
};

// Mapping from Backend (simplified copy for MVP preview)


// --- LocalStorage Usage Tracking ---
type EaUsage = {
    analysisCount: number;
    firstAnalysisAt?: number; // epoch ms
    isEarlyAccess?: boolean;  // paid flag (MVP)
};

const EA_USAGE_KEY = "ea_usage_v1";

function loadEaUsage(): EaUsage {
    if (typeof window === "undefined") return { analysisCount: 0 };
    try {
        const raw = localStorage.getItem(EA_USAGE_KEY);
        if (!raw) return { analysisCount: 0 };
        const parsed = JSON.parse(raw) as EaUsage;
        return {
            analysisCount: Number(parsed.analysisCount ?? 0),
            firstAnalysisAt: typeof parsed.firstAnalysisAt === "number" ? parsed.firstAnalysisAt : undefined,
            isEarlyAccess: Boolean(parsed.isEarlyAccess ?? false),
        };
    } catch {
        return { analysisCount: 0 };
    }
}

function saveEaUsage(next: EaUsage) {
    if (typeof window === "undefined") return;
    localStorage.setItem(EA_USAGE_KEY, JSON.stringify(next));
}

// --- Limitation Rules ---
const FREE_ANALYSIS_LIMIT = 2;

function canAnalyze(usage: EaUsage): { ok: true } | { ok: false; reason: "LIMIT_REACHED" } {
    if (usage.isEarlyAccess) return { ok: true };
    if (usage.analysisCount >= FREE_ANALYSIS_LIMIT) return { ok: false, reason: "LIMIT_REACHED" };
    return { ok: true };
}

type CtaReason = "EXPORT" | "LIMIT_REACHED";

// --- Onboarding Logic ---
type OnboardingStep = "WELCOME" | "UPLOAD" | "PLAY" | "EDIT" | "EXPORT" | "DONE";
const EA_ONBOARDING_KEY = "ea_onboarding_v1";

type OnboardingState = { completed: boolean };

function loadOnboarding(): OnboardingState {
    if (typeof window === "undefined") return { completed: false };
    try {
        const raw = localStorage.getItem(EA_ONBOARDING_KEY);
        if (!raw) return { completed: false };
        return JSON.parse(raw);
    } catch {
        return { completed: false };
    }
}

function completeOnboarding() {
    if (typeof window === "undefined") return;
    localStorage.setItem(EA_ONBOARDING_KEY, JSON.stringify({ completed: true }));
}

function InlineHint({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <div className={`text-sm text-teal-300 bg-teal-500/10 border border-teal-500/30 rounded px-3 py-2 ${className}`}>
            {children}
        </div>
    );
}

// CSS Content Hack: The text is put in `data-text` and displayed via `before:content-[attr(data-text)]`
// This removes the text from the DOM content tree, bypassing most translation engines.

export default function EarlyAccessPage() {
    const router = useRouter();
    const [file, setFile] = useState<File | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [status, setStatus] = useState<AppStatus>('idle');
    const [progress, setProgress] = useState<number>(0); // Added progress state
    const [error, setError] = useState<ErrorState | null>(null);

    // Free Tier / CTA State
    const [eaUsage, setEaUsage] = useState<EaUsage>({ analysisCount: 0 });
    const [ctaOpen, setCtaOpen] = useState(false);
    const [ctaReason, setCtaReason] = useState<CtaReason>("EXPORT");

    // Onboarding State
    const [onboardingStep, setOnboardingStep] = useState<OnboardingStep | null>(null);

    useEffect(() => {
        // Load Usage
        const u = loadEaUsage();

        // CHECK FOR UNLOCK (Option C)
        // We use window.location here for minimal diff / avoiding Suspense requirement on this file
        if (typeof window !== "undefined") {
            const params = new URLSearchParams(window.location.search);
            if (params.get("paid") === "1") {
                console.log("EA_UNLOCKED_VIA_PAYMENT");
                u.isEarlyAccess = true;
                saveEaUsage(u);

                // Clean URL
                router.replace("/early-access");
            }
        }

        setEaUsage(u);
    }, [router]);

    // Init Onboarding if Paid (Separate effect to rely on stable state)
    useEffect(() => {
        if (!eaUsage.isEarlyAccess) return;
        const ob = loadOnboarding();
        if (!ob.completed) setOnboardingStep("WELCOME");
    }, [eaUsage.isEarlyAccess]);

    // Onboarding Transitions
    useEffect(() => {
        if (onboardingStep === "UPLOAD" && file) {
            setOnboardingStep("PLAY");
        }
    }, [onboardingStep, file]);

    useEffect(() => {
        // Advanced from PLAY to EDIT when analysis is ready
        // Note: This makes the "Just listen" step appear during the analysis/idle phase after upload
        if (onboardingStep === "PLAY" && status === "ready") {
            setOnboardingStep("EXPORT");
        }
    }, [onboardingStep, status]);

    useEffect(() => {
        if (onboardingStep === "DONE") {
            const t = setTimeout(() => setOnboardingStep(null), 500);
            return () => clearTimeout(t);
        }
    }, [onboardingStep]);

    // Logging: EA_CTA_OPEN
    useEffect(() => {
        if (ctaOpen) {
            console.log("EA_CTA_OPEN", { reason: ctaReason });
        }
    }, [ctaOpen, ctaReason]);


    // Editing State (Removed for ResultDisplay refactor)
    // Auto Preview & Volume State (Removed for ResultDisplay refactor)

    // Refs
    const fileInputRef = useRef<HTMLInputElement>(null);
    const abortPollingRef = useRef<AbortController | null>(null);

    // Constants
    const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB


    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];

            // Basic Frontend Validation
            if (selectedFile.size > MAX_FILE_SIZE) {
                setError({ code: 'FILE_TOO_LARGE', message: 'Maximum file size is 20MB.' });
                return;
            }

            setFile(selectedFile);
            setAudioUrl(URL.createObjectURL(selectedFile));
            setError(null);
            setStatus('idle');
            setProgress(0);
            setResult(null);
        }
    };

    const pollJob = async (jobId: string, signal: AbortSignal, submittedAt: number) => {
        const base = process.env.NEXT_PUBLIC_API_BASE_URL;

        let lastProgress = -1;
        let lastUpdateTime = Date.now();

        while (true) {
            if (signal.aborted) return;

            // Wait 1.5s
            await new Promise(r => setTimeout(r, 1500));

            if (signal.aborted) return;

            // Check status
            try {
                const s = await fetch(`${base}/analyze/status/${jobId}`, { signal });
                if (s.status === 404) throw new Error("JOB_LOST");

                const data = await s.json();
                const jobStatus = data.status;
                const p = typeof data.progress === "number" ? data.progress : 0;

                console.log("progress", p, "status", jobStatus);
                setProgress(p);

                // STARTUP CHECK (Correction 2)
                // If 15s passed and server hasn't acknowledged start (no started_at), assume thread died
                const startedAt = data.started_at;
                if (!startedAt && (Date.now() - submittedAt > 15000)) {
                    throw new Error("JOB_NOT_STARTED");
                }

                // Stalled Check
                if (p > lastProgress) {
                    lastProgress = p;
                    lastUpdateTime = Date.now();
                } else {
                    // Progress hasn't moved
                    if (Date.now() - lastUpdateTime > 25000) { // 25s timeout
                        throw new Error("JOB_STALLED");
                    }
                }

                if (jobStatus === "error") throw new Error("ANALYSIS_FAILED_BG");
                if (jobStatus === "done") {
                    // Fetch Result
                    const r = await fetch(`${base}/analyze/result/${jobId}`, { signal });
                    if (!r.ok) throw new Error("RESULT_FETCH_FAILED");
                    const resultData: AnalysisResult = await r.json();

                    setResult(resultData);
                    setProgress(100);
                    setStatus('ready');

                    // Increment Usage
                    setEaUsage((prev) => {
                        const next: EaUsage = {
                            ...prev,
                            analysisCount: (prev.analysisCount ?? 0) + 1,
                            firstAnalysisAt: prev.firstAnalysisAt ?? Date.now(),
                        };
                        saveEaUsage(next);
                        return next;
                    });
                    return; // Done
                }
                // If processing, loop continues
            } catch (e: any) {
                if (signal.aborted || e.name === "AbortError") return;

                console.error("Polling error:", e);
                if (e.message === "JOB_LOST" || e.message === "JOB_STALLED" || e.message === "JOB_NOT_STARTED") {
                    setError({
                        code: "SERVER_RESTARTED",
                        message: "サーバーが再起動した可能性があるため、解析が中断されました。もう一度お試しください。"
                    });
                } else if (e.message === "ANALYSIS_FAILED_BG") {
                    setError({ code: "ANALYSIS_FAILED", message: "Audio analysis failed on the server." });
                } else {
                    setError({ code: "NETWORK_ERROR", message: "Connection lost during polling." });
                }
                setStatus('error');
                return; // Stop polling
            }
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;

        // FREE TIER GATE
        const gate = canAnalyze(eaUsage);
        if (!gate.ok) {
            setCtaReason("LIMIT_REACHED");
            setCtaOpen(true);
            return;
        }

        if (!process.env.NEXT_PUBLIC_API_BASE_URL) {
            setError({ code: 'CONFIG_ERROR', message: 'Configuration Error: API Base URL not set.' });
            return;
        }

        if (abortPollingRef.current) {
            abortPollingRef.current.abort();
        }
        const controller = new AbortController();
        abortPollingRef.current = controller;

        setStatus('analyzing');
        setProgress(0);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            // Connect to Backend
            const base = process.env.NEXT_PUBLIC_API_BASE_URL;
            const url = `${base}/analyze`;
            const submittedAt = Date.now();
            console.log("Submitting job to:", url);

            const res = await fetch(url, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });

            if (!res.ok) {
                // Handle immediate errors (validation etc)
                let errorData;
                try { errorData = await res.json(); } catch (e) { }

                if (errorData?.detail?.error) {
                    throw errorData.detail.error;
                } else {
                    throw { code: 'HTTP_ERROR', message: `Status ${res.status}` };
                }
            }

            const { job_id } = await res.json();
            console.log("Job started:", job_id);

            // Start Polling
            pollJob(job_id, controller.signal, submittedAt);

        } catch (err: any) {
            if (err.name === "AbortError") return;
            console.error('Submission failed:', err);
            if (err.code && err.message) {
                setError(err);
            } else {
                setError({ code: 'SUBMISSION_FAILED', message: 'Failed to start analysis.' });
            }
            setStatus('error');
        }
    };





    // EXPORT LOGIC
    const downloadFile = (content: string, filename: string, type: string) => {
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleExportJSON = () => {
        if (!eaUsage.isEarlyAccess) {
            setCtaReason("EXPORT");
            setCtaOpen(true);
            return;
        }

        if (!result) return;

        // Onboarding Advance
        if (onboardingStep === "EXPORT") {
            completeOnboarding();
            setOnboardingStep("DONE");
        }

        downloadFile(JSON.stringify(result, null, 2), `analysis-${Date.now()}.json`, 'application/json');
    };

    const handleExportTXT = () => {
        if (!eaUsage.isEarlyAccess) {
            setCtaReason("EXPORT");
            setCtaOpen(true);
            return;
        }

        if (!result) return;

        // Onboarding Advance
        if (onboardingStep === "EXPORT") {
            completeOnboarding();
            setOnboardingStep("DONE");
        }

        const timeline = analysisResultToTimedChords(result);
        const lines = timeline.map(c => `${c.startSec.toFixed(3)}\t${c.endSec.toFixed(3)}\t${c.name}`).join('\n');
        downloadFile(lines, `chords-${Date.now()}.txt`, 'text/plain');
    };

    // Safe Audio Cleanup
    useEffect(() => {
        return () => {
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
        };
    }, [audioUrl]);

    return (
        <div className="min-h-screen bg-neutral-950 text-white p-8 font-sans">
            <div className="max-w-4xl mx-auto space-y-8">

                {/* HEADER */}
                <header className="border-b border-neutral-800 pb-6">
                    <p className="text-xs text-neutral-500 mb-2">
                        表示が「午前」などに変わる場合は、ブラウザのページ翻訳をOFFにしてください。
                    </p>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-transparent">
                        Early Access: Audio Analysis
                    </h1>
                    <p className="text-neutral-400 mt-2 text-sm">
                        Upload your own audio files to generate chord sheets.
                        <span className="block text-yellow-500/80 mt-1">
                            ⚠ Accuracy is experimental. This tool is offline-first; files are processed temporarily and deleted.
                        </span>
                    </p>
                </header>

                {/* UPLOAD SECTION */}
                {/* STEP 1: UPLOAD HINT */}
                {onboardingStep === "UPLOAD" && (
                    <div className="max-w-xl mx-auto mb-2">
                        <InlineHint>Step 1: 音源を1曲アップロードしてください</InlineHint>
                    </div>
                )}
                {/* STEP 2: PLAY HINT */}
                {onboardingStep === "PLAY" && (
                    <div className="max-w-xl mx-auto mb-2">
                        <InlineHint>まずは、合っているかどうかを気にせず再生してください (解析完了をお待ちください)</InlineHint>
                    </div>
                )}
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
                                    <p className="text-lg font-medium text-neutral-200">Click to Select Audio File</p>
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
                                    onClick={(e) => { e.stopPropagation(); setFile(null); setAudioUrl(null); }}
                                    className="p-2 hover:bg-neutral-700 rounded-full text-neutral-400 hover:text-red-400 transition-colors"
                                >
                                    ✕
                                </button>
                            </div>
                        )}

                        {file && status !== 'analyzing' && status !== 'ready' && (
                            <button
                                onClick={handleAnalyze}
                                className="mt-6 px-8 py-3 bg-teal-600 hover:bg-teal-500 text-white font-medium rounded-lg shadow-lg hover:shadow-teal-500/20 transition-all active:scale-95"
                            >
                                Generate Chord Draft
                            </button>
                        )}
                    </div>
                </section>

                {/* LOADING STATE */}
                {status === 'analyzing' && (
                    <div className="text-center py-12">
                        <Loader2 className="h-12 w-12 animate-spin text-teal-500 mx-auto mb-4" />
                        <h3 className="text-xl font-bold mb-2">Analyzing Audio...</h3>
                        <p className="text-gray-400">
                            Detecting chords, beats, and key signature. <br />
                            This may take a moment depending on the file size.
                        </p>

                        {/* Progress Bar */}
                        <div className="max-w-md mx-auto mt-6">
                            <div className="flex items-center justify-between text-xs text-gray-400 mb-2 font-mono">
                                <span>PROCESSING</span>
                                <span>{Math.floor(progress)}%</span>
                            </div>
                            <div className="h-1.5 w-full bg-neutral-800 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-teal-500 to-emerald-400 transition-all duration-300 ease-out"
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

                {/* RESULTS SECTION */}
                {status === 'ready' && (
                    (result?.bars?.length ?? 0) > 0 ? (
                        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                            <ResultDisplay result={result!} audioUrl={audioUrl} />

                            {/* Actions Footer */}
                            <div className="flex justify-end items-center space-x-4 pt-8 border-t border-neutral-800">
                                {/* STEP 4: EXPORT HINT */}
                                {onboardingStep === "EXPORT" && (
                                    <span className="text-sm text-teal-400 animate-pulse">
                                        この下書きを DAW に持っていけます &rarr;
                                    </span>
                                )}
                                <button
                                    onClick={handleExportTXT}
                                    className="px-4 py-2 text-sm text-neutral-400 hover:text-white transition-colors border border-neutral-800 hover:border-neutral-600 rounded-lg">
                                    Export Text
                                </button>
                                <button
                                    onClick={handleExportJSON}
                                    className="px-4 py-2 text-sm text-neutral-400 hover:text-white transition-colors border border-neutral-800 hover:border-neutral-600 rounded-lg">
                                    Export JSON
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="text-center py-12 text-neutral-400">
                            解析結果が空でした。別の音源でお試しください。
                        </div>
                    )
                )}

                {/* CTA MODAL */}
                {ctaOpen && (
                    <div role="dialog" aria-modal="true" className="fixed inset-0 z-[100] flex items-center justify-center">
                        <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={() => setCtaOpen(false)} />
                        <div className="relative w-full max-w-lg rounded-xl bg-white p-6 shadow-2xl text-neutral-900 mx-4">
                            {ctaReason === "EXPORT" ? (
                                <>
                                    <h2 className="text-xl font-bold">この下書きを、DAW に持っていきませんか？</h2>
                                    <p className="mt-3 text-sm text-gray-600">
                                        今作ったコード下書きは、このままでは外に持ち出せません。Early Access に参加すると Export できます。
                                    </p>
                                </>
                            ) : (
                                <>
                                    <h2 className="text-xl font-bold">2曲分の下書きを作成しました</h2>
                                    <p className="mt-3 text-sm text-gray-600">
                                        ここまでで精度感と編集の流れは十分に判断できます。続けて使うなら Early Access をご検討ください。
                                    </p>
                                </>
                            )}

                            <div className="mt-4 rounded-lg bg-gray-50 p-4 text-sm space-y-2 border border-gray-100">
                                <div className="flex items-center"><span className="w-16 font-semibold text-gray-500">価格</span> ¥1,980（Early Access）</div>
                                <div className="flex items-center"><span className="w-16 font-semibold text-gray-500">内容</span> 解析・編集・Export 制限解除</div>
                                <div className="flex items-center"><span className="w-16 font-semibold text-gray-500">安心</span> 7日以内返金可</div>
                            </div>

                            <div className="mt-6 flex gap-3">
                                <button className="flex-1 rounded-lg bg-black px-4 py-3 text-white font-bold hover:bg-neutral-800 transition-colors" onClick={() => {
                                    setCtaOpen(false);
                                    const from = ctaReason === "EXPORT" ? "export" : "limit";
                                    console.log("EA_CTA_CLICK", { reason: ctaReason, to: `/waitlist?from=${from}` });
                                    router.push(`/waitlist?from=${from}`);
                                }}>
                                    Early Access に参加する
                                </button>
                                <button className="rounded-lg border border-gray-300 px-4 py-3 text-sm text-gray-600 hover:bg-gray-50 transition-colors" onClick={() => setCtaOpen(false)}>
                                    今回はここまでにする
                                </button>
                            </div>

                            <p className="mt-4 text-xs text-gray-400 text-center">
                                ※ Early Access は開発途中の版です。完璧な解析は行いません。
                            </p>
                        </div>
                    </div>
                )}

                {/* ONBOARDING MODAL (Step 0) */}
                {onboardingStep === "WELCOME" && (
                    <div role="dialog" aria-modal="true" className="fixed inset-0 z-[100] flex items-center justify-center">
                        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
                        <div className="relative w-full max-w-md rounded-xl bg-neutral-900 border border-neutral-700 p-6 shadow-2xl text-white mx-4">
                            <h2 className="text-xl font-bold mb-4">Early Access へようこそ</h2>
                            <div className="space-y-3 text-neutral-300 mb-6">
                                <p>
                                    このツールは <strong className="text-teal-400">正確な答え</strong> を出すものではありません。
                                </p>
                                <p>
                                    下書きを最速で作り、あとはあなたの耳で仕上げるためのものです。
                                </p>
                            </div>
                            <div className="flex justify-end gap-3">
                                <button
                                    onClick={() => {
                                        completeOnboarding();
                                        setOnboardingStep(null);
                                    }}
                                    className="px-4 py-2 text-sm text-neutral-500 hover:text-white transition-colors"
                                >
                                    Skip
                                </button>
                                <button
                                    onClick={() => setOnboardingStep("UPLOAD")}
                                    className="px-6 py-2 bg-teal-600 hover:bg-teal-500 text-white font-bold rounded-lg transition-colors"
                                >
                                    始める
                                </button>
                            </div>
                        </div>
                    </div>
                )}

            </div>
        </div>
    );
}
