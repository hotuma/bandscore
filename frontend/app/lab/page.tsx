'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import FileUpload from '../../components/FileUpload';
import ResultDisplay from '../../components/ResultDisplay';
import { analyzeAudio, analyzeYoutube, AnalysisResult } from '../../lib/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
if (!API_BASE_URL) {
    throw new Error("NEXT_PUBLIC_API_BASE_URL is not set");
}

// Note: For client components, we can't export metadata directly in the same file if we want it to work easily 
// with the simplified 'use client' pattern sometimes, but in App Router 'use client' pages can't export metadata.
// Since we are inside a layout that might handle metadata, or we can use a separate layout.tsx for /lab.
// However, the user requirement says "add noindex". 
// A 'use client' page CANNOT export metadata. 
// Solution: We should move the page logic to a component or keep this as 'use client' and add a layout.tsx in /lab 
// OR just Accept that we can't do metadata in this file easily without removing 'use client'.
// BETTER APPROACH for this specific task:
// Create app/lab/layout.tsx to handle the noindex metadata for all /lab routes.
// The user asked to add noindex to /lab. existing /lab/login already has it.
// So I will creating app/lab/layout.tsx is the cleanest way.

export default function LabHome() {
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);

    const handleFileSelect = async (file: File) => {
        if (isAnalyzing) return;

        setError(null);
        setResult(null);
        setIsAnalyzing(true);

        // Create URL for playback locally (prefer local blob for uploads for speed)
        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
        }
        const blobUrl = URL.createObjectURL(file);
        setAudioUrl(blobUrl);

        try {
            const data = await analyzeAudio(file);
            setResult(data);
            // If we wanted to use the server URL: setAudioUrl(`${API_BASE_URL}${data.audio_url}`);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
            setAudioUrl(null);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleUrlSelect = async (url: string, cookiesFile?: File) => {
        console.log("[UI] analyzeYoutube called", { url, hasCookies: !!cookiesFile, isAnalyzing });
        if (isAnalyzing) {
            console.warn("[UI] blocked because isAnalyzing=true");
            return;
        }

        setError(null);
        setResult(null);
        setIsAnalyzing(true);
        setAudioUrl(null); // Clear previous

        try {
            const data = await analyzeYoutube(url, cookiesFile);
            console.log("[UI] analyzeYoutube success");
            setResult(data);
            if (data.audio_url) {
                // api.ts has already normalized this to an absolute URL
                setAudioUrl(data.audio_url);
            }
        } catch (err: any) {
            console.error("[UI] analyzeYoutube error", err);
            // Clean error presentation
            let msg = err instanceof Error ? err.message : 'An error occurred';
            if (msg.includes("403")) {
                msg = "Access Denied by YouTube. This video may require login. Please use the 'Advanced Options' to upload cookies.txt, or download the MP3 and upload it directly.";
            } else if (msg.includes("429")) {
                msg = "Too many requests to YouTube. Please wait a few minutes and try again.";
            }
            setError(msg);
        } finally {
            setIsAnalyzing(false);
        }
    };

    return (
        <main className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-extrabold text-gray-900 mb-2 tracking-tight">
                        BandScore Lab
                    </h1>
                    <p className="text-lg text-gray-600 mb-4">
                        Internal Audio Analysis Tools
                    </p>
                    <Link
                        href="/lab/training"
                        className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                        </svg>
                        BeatNet-Plus Training
                    </Link>
                </div>

                <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
                    <FileUpload
                        onFileSelect={handleFileSelect}
                        onUrlSelect={handleUrlSelect}
                        isLoading={isAnalyzing}
                    />

                    {error && (
                        <div className="mt-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-center">
                            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                            {error}
                        </div>
                    )}

                    {isAnalyzing && (
                        <div className="mt-12 text-center py-12">
                            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent mb-4"></div>
                            <p className="text-gray-500 font-medium animate-pulse">Analyzing audio...</p>
                            <p className="text-xs text-gray-400 mt-2">This might take a few seconds depending on the file size.</p>
                        </div>
                    )}

                    {!isAnalyzing && result && <ResultDisplay result={result} audioUrl={audioUrl} />}
                </div>

                <div className="mt-12 text-center text-gray-400 text-sm">
                    <p>&copy; 2025 BandScore Lab. Restricted Access.</p>
                </div>
            </div>
        </main>
    );
}
