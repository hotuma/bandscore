'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Trash2, AlertCircle, Loader2 } from 'lucide-react';
import { playChordFromTab, setChordVolume } from '../../lib/chordAudio';

// Types matching Backend Contract
type Chord = {
    startSec: number;
    endSec: number;
    name: string;
    confidence: number;
    frets?: string[]; // Optional if we map it later, but backend currently returns name-only in main list, mapped from bars
};

type AnalysisMeta = {
    durationSec: number;
    bpm: number;
    key: string;
};

type AnalysisResponse = {
    chords: Chord[];
    meta: AnalysisMeta;
};

type AppStatus = 'idle' | 'uploading' | 'analyzing' | 'ready' | 'error';

type ErrorState = {
    code: string;
    message: string;
};

// Mapping from Backend (simplified copy for MVP preview)
const CHORD_TO_TAB: Record<string, string[]> = {
    "C": ["x", "3", "2", "0", "1", "0"],
    "C#": ["x", "4", "6", "6", "6", "4"],
    "D": ["x", "x", "0", "2", "3", "2"],
    "D#": ["x", "6", "8", "8", "8", "6"],
    "E": ["0", "2", "2", "1", "0", "0"],
    "F": ["1", "3", "3", "2", "1", "1"],
    "F#": ["2", "4", "4", "3", "2", "2"],
    "G": ["3", "2", "0", "0", "0", "3"],
    "G#": ["4", "6", "6", "5", "4", "4"],
    "A": ["x", "0", "2", "2", "2", "0"],
    "A#": ["x", "1", "3", "3", "3", "1"],
    "B": ["x", "2", "4", "4", "4", "2"],
    "Cm": ["x", "3", "5", "5", "4", "3"],
    "C#m": ["x", "4", "6", "6", "5", "4"],
    "Dm": ["x", "x", "0", "2", "3", "1"],
    "D#m": ["x", "6", "8", "8", "7", "6"],
    "Em": ["0", "2", "2", "0", "0", "0"],
    "Fm": ["1", "3", "3", "1", "1", "1"],
    "F#m": ["2", "4", "4", "2", "2", "2"],
    "Gm": ["3", "5", "5", "3", "3", "3"],
    "G#m": ["4", "6", "6", "4", "4", "4"],
    "Am": ["x", "0", "2", "2", "1", "0"],
    "A#m": ["x", "1", "3", "3", "2", "1"],
    "Bm": ["x", "2", "4", "4", "3", "2"],
};

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

export default function EarlyAccessPage() {
    const router = useRouter();
    const [file, setFile] = useState<File | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [chords, setChords] = useState<Chord[]>([]);
    const [meta, setMeta] = useState<AnalysisMeta | null>(null);
    const [status, setStatus] = useState<AppStatus>('idle');
    const [progress, setProgress] = useState<number>(0); // Added progress state
    const [error, setError] = useState<ErrorState | null>(null);
    const [currentTime, setCurrentTime] = useState(0);

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
            setOnboardingStep("EDIT");
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


    // Editing State
    const [editingIndex, setEditingIndex] = useState<number | null>(null);

    // Auto Preview & Volume State
    const [autoPreview, setAutoPreview] = useState(false);
    const [sourceVolume, setSourceVolume] = useState(0.5);
    const [chordVolume, setChordVolumeState] = useState(0.5);

    const prevChordIndexRef = useRef<number>(-1);
    const audioRef = useRef<HTMLAudioElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const abortPollingRef = useRef<AbortController | null>(null);

    // Constants
    const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
    const ALLOWED_TYPES = ['audio/mpeg', 'audio/wav', 'audio/x-m4a', 'audio/mp4', 'audio/x-wav'];

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
            setChords([]);
            setMeta(null);
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
                // If 15s passed and progress is still < 1%, assume thread never started (Render killed it)
                if (p < 1 && (Date.now() - submittedAt > 15000)) {
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
                    const r = await fetch(`${base}/analyze/result/${jobId}`);
                    if (!r.ok) throw new Error("RESULT_FETCH_FAILED");
                    const resultData: AnalysisResponse = await r.json();

                    setChords(resultData.chords);
                    setMeta(resultData.meta);
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
            pollJob(job_id, controller.signal, Date.now());

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

    const playPreview = async (chordName: string) => {
        const trimmed = chordName.trim();
        if (CHORD_TO_TAB[trimmed]) {
            await playChordFromTab(CHORD_TO_TAB[trimmed]);
        }
    };

    const duckAudio = () => {
        if (!audioRef.current) return;
        const originalVol = sourceVolume;
        // Duck
        audioRef.current.volume = Math.max(0, originalVol * 0.3);

        // Restore after short delay (e.g. 150ms)
        setTimeout(() => {
            if (audioRef.current) {
                // Check if user changed volume during ducking? 
                // For MVP, just restore to current state sourceVolume
                // But we must read the *latest* sourceVolume if possible, or ref?
                // Actually, accessing state in timeout closure uses captured value.
                // Ref pattern is safer but maybe overkill. Let's use simple logic:
                // We will rely on the useEffect below to sync volume, so we just set it back?
                // Actually, if we just set it back it might be stale.
                // Better: Trigger a re-apply or just set it. 
                // Let's rely on a ref for sourceVolume to be safe or just use the volume property.
            }
        }, 150);

        // Actually, the useEffect [sourceVolume] will force update. 
        // We need to bypass the effect or make sure effect doesn't override ducking?
        // Simpler: The effect sets volume. Ducking sets it temporarily. 
        // If effect fires during duck (unlikely unless user drags slider), it resets.
        // We will just restore to 'sourceVolume' (closure capture).
        // Since playPreview is short, 150ms is fine.
    };

    // Restore volume after ducking needs to happen carefully. 
    // Let's use a timeout refs to avoid race conditions?
    const duckTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const triggerDucking = () => {
        if (!audioRef.current) return;

        if (duckTimeoutRef.current) clearTimeout(duckTimeoutRef.current);

        // Milder ducking: 70% of current volume instead of 20%
        audioRef.current.volume = sourceVolume * 0.7;

        duckTimeoutRef.current = setTimeout(() => {
            if (audioRef.current) {
                audioRef.current.volume = sourceVolume;
            }
            duckTimeoutRef.current = null;
        }, 200); // 200ms duck
    };

    const handleTimeUpdate = () => {
        if (audioRef.current) {
            const time = audioRef.current.currentTime;
            setCurrentTime(time);

            // Auto Preview Logic
            if (autoPreview && !audioRef.current.paused && chords.length > 0) {
                const activeIndex = chords.findIndex(c => time >= c.startSec && time < c.endSec);

                if (activeIndex !== -1 && activeIndex !== prevChordIndexRef.current) {
                    // New Chord entered
                    const chord = chords[activeIndex];
                    playPreview(chord.name);
                    triggerDucking();
                    prevChordIndexRef.current = activeIndex;
                } else if (activeIndex === -1) {
                    prevChordIndexRef.current = -1;
                }
            }
        }
    };

    // Volume Effects

    useEffect(() => {
        if (audioRef.current) {
            audioRef.current.volume = sourceVolume;
        }
    }, [sourceVolume]);

    useEffect(() => {
        setChordVolume(chordVolume);
    }, [chordVolume]);

    const formatTime = (sec: number) => {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    // EDITING LOGIC
    const startEditing = (index: number) => {
        setEditingIndex(index);
        // Focus will be handled by autoFucus on input
    };

    const saveChord = async (index: number, newName: string) => {
        // Basic validation
        const trimmed = newName.trim();
        if (!trimmed || trimmed.length > 12) {
            // Revert or alert? For MVP just revert if empty, keep if valid
            setEditingIndex(null);
            return;
        }

        const updated = [...chords];
        updated[index] = { ...updated[index], name: trimmed };
        setChords(updated);
        setEditingIndex(null);

        // Instant Preview
        // Better fallback: simplistic parsing
        if (CHORD_TO_TAB[trimmed]) {
            await playChordFromTab(CHORD_TO_TAB[trimmed]);
        } else {
            // Just play a default click or generic chord if unknown, to confirm action
        }

        // Onboarding Advance
        if (onboardingStep === "EDIT") {
            setOnboardingStep("EXPORT");
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

        // Onboarding Advance
        if (onboardingStep === "EXPORT") {
            completeOnboarding();
            setOnboardingStep("DONE");
        }

        const data = {
            meta: meta,
            chords: chords
        };
        downloadFile(JSON.stringify(data, null, 2), `chords-${Date.now()}.json`, 'application/json');
    };

    const handleExportTXT = () => {
        if (!eaUsage.isEarlyAccess) {
            setCtaReason("EXPORT");
            setCtaOpen(true);
            return;
        }

        // Onboarding Advance
        if (onboardingStep === "EXPORT") {
            completeOnboarding();
            setOnboardingStep("DONE");
        }

        const lines = chords.map(c => `${c.startSec.toFixed(3)}\t${c.endSec.toFixed(3)}\t${c.name}`).join('\n');
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
                {status === 'ready' && chords.length > 0 && (
                    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">

                        {/* Audio Player */}
                        <div className="bg-neutral-900 rounded-xl p-4 border border-neutral-800 sticky top-4 z-50 shadow-2xl backdrop-blur-md bg-neutral-900/90">
                            <div className="flex items-center space-x-4 mb-2 px-2">
                                {/* Auto Preview Toggle */}
                                <label className="flex items-center space-x-2 cursor-pointer group">
                                    <div className={`w-10 h-6 flex items-center bg-gray-700 rounded-full p-1 duration-300 ease-in-out ${autoPreview ? 'bg-teal-600' : ''}`}>
                                        <div className={`bg-white w-4 h-4 rounded-full shadow-md transform duration-300 ease-in-out ${autoPreview ? 'translate-x-4' : ''}`} />
                                    </div>
                                    <input
                                        type="checkbox"
                                        className="hidden"
                                        checked={autoPreview}
                                        onChange={(e) => setAutoPreview(e.target.checked)}
                                    />
                                    <span className="text-sm text-neutral-300 group-hover:text-white select-none">Auto Chord Preview</span>
                                </label>

                                <div className="h-4 w-px bg-neutral-700 mx-2"></div>

                                {/* Volume Controls */}
                                <div className="flex items-center space-x-4">
                                    <div className="flex items-center space-x-2">
                                        <span className="text-xs text-neutral-400">Music</span>
                                        <input
                                            type="range" min="0" max="1" step="0.05"
                                            value={sourceVolume}
                                            onChange={(e) => setSourceVolume(parseFloat(e.target.value))}
                                            className="w-20 h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                                        />
                                    </div>
                                    <div className="flex items-center space-x-2">
                                        <span className="text-xs text-neutral-400">Chords</span>
                                        <input
                                            type="range" min="0" max="1" step="0.05"
                                            value={chordVolume}
                                            onChange={(e) => setChordVolumeState(parseFloat(e.target.value))}
                                            className="w-20 h-1 bg-neutral-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
                                        />
                                    </div>
                                </div>
                            </div>

                            <audio
                                ref={audioRef}
                                src={audioUrl!}
                                controls
                                className="w-full h-10 invert-[.9] opacity-80"
                                onTimeUpdate={handleTimeUpdate}
                            />
                            <div className="flex justify-between text-xs text-neutral-500 px-2 mt-2 font-mono">
                                <span>{formatTime(currentTime)}</span>
                                <span>{meta?.bpm ? `${Math.round(meta.bpm)} BPM` : 'Unknown BPM'} • {meta?.key || 'Unknown Key'}</span>
                                <span>{formatTime(meta?.durationSec || 0)}</span>
                            </div>
                        </div>

                        {/* Chord Timeline (Grid) */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
                            {chords.map((chord, idx) => {
                                const isActive = currentTime >= chord.startSec && currentTime < chord.endSec;
                                const isEditing = editingIndex === idx;

                                return (
                                    <div
                                        key={`${chord.startSec}-${idx}`}
                                        className={`
                         relative p-3 rounded-lg border transition-all duration-200 group
                         ${isActive
                                                ? 'bg-teal-500/20 border-teal-500 shadow-[0_0_15px_rgba(20,184,166,0.3)] scale-105 z-10'
                                                : 'bg-neutral-800/50 border-neutral-800 hover:border-neutral-600 hover:bg-neutral-800'}
                       `}
                                        onClick={(e) => {
                                            // Only seek if not editing
                                            if (!isEditing && !e.defaultPrevented) {
                                                if (audioRef.current) {
                                                    audioRef.current.currentTime = chord.startSec;
                                                    audioRef.current.play();
                                                }
                                            }
                                        }}
                                    >
                                        {/* STEP 3: EDIT TOOLTIP */}
                                        {onboardingStep === "EDIT" && idx === 0 && (
                                            <div className="absolute -top-12 left-1/2 -translate-x-1/2 z-50 whitespace-nowrap">
                                                <div className="bg-teal-500 text-white text-xs font-bold px-2 py-1 rounded shadow-lg animate-bounce">
                                                    ここをクリックして編集できます
                                                    <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 border-4 border-transparent border-t-teal-500"></div>
                                                </div>
                                            </div>
                                        )}
                                        {isEditing ? (
                                            <input
                                                autoFocus
                                                className="bg-transparent text-xl font-bold text-white w-full outline-none border-b border-teal-500"
                                                defaultValue={chord.name}
                                                onBlur={(e) => saveChord(idx, e.target.value)}
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter') {
                                                        e.currentTarget.blur();
                                                    }
                                                }}
                                                onClick={(e) => e.stopPropagation()}
                                            />
                                        ) : (
                                            <span
                                                className={`text-xl font-bold block cursor-pointer hover:text-teal-400 ${isActive ? 'text-teal-300' : 'text-neutral-200'}`}
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    e.preventDefault(); // Prevent Seek
                                                    startEditing(idx);
                                                }}
                                            >
                                                {chord.name}
                                            </span>
                                        )}

                                        <span className="text-[10px] text-neutral-500 font-mono absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                                            {formatTime(chord.startSec)}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>

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
