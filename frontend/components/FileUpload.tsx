'use client';

import React, { useState, ChangeEvent } from 'react';

interface FileUploadProps {
    onFileSelect: (file: File) => void;
    onUrlSelect: (url: string, cookiesFile?: File) => void;
    isLoading?: boolean;
}

export default function FileUpload({ onFileSelect, onUrlSelect, isLoading = false }: FileUploadProps) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [url, setUrl] = useState('');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [cookieFile, setCookieFile] = useState<File | null>(null);

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setSelectedFile(file);
            onFileSelect(file);
        }
    };

    const handleCookieChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setCookieFile(e.target.files[0]);
        }
    };

    const handleUrlSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (url.trim() && !isLoading) {
            onUrlSelect(url.trim(), cookieFile || undefined);
        }
    };

    return (
        <div className="space-y-6">
            {/* File Drop Zone */}
            <div className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-gray-300 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors">
                <label htmlFor="audio-upload" className={`cursor-pointer text-center w-full ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}>
                    <span className="text-gray-600 font-medium block">
                        {selectedFile ? selectedFile.name : 'Click to upload MP3'}
                    </span>
                    <input
                        id="audio-upload"
                        type="file"
                        accept=".mp3,audio/*"
                        className="hidden"
                        onChange={handleFileChange}
                        disabled={isLoading}
                    />
                </label>
            </div>

            <div className="relative flex items-center justify-center">
                <div className="border-t w-full border-gray-200"></div>
                <span className="bg-white px-2 text-sm text-gray-500 absolute">OR</span>
            </div>

            {/* URL Input */}
            <form onSubmit={handleUrlSubmit} className="space-y-4">
                <div className="flex gap-2">
                    <input
                        type="url"
                        placeholder="Paste YouTube URL"
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none disabled:opacity-50 disabled:bg-gray-100"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        required
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        className="px-6 py-2 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={!url.trim() || isLoading}
                    >
                        {isLoading ? 'Analyzing...' : 'Analyze'}
                    </button>
                </div>

                {/* Advanced Options Toggle */}
                <div className="pt-2">
                    <button
                        type="button"
                        onClick={() => setShowAdvanced(!showAdvanced)}
                        className="text-sm text-gray-500 hover:text-gray-700 underline focus:outline-none"
                    >
                        {showAdvanced ? 'Hide Advanced Options' : 'Show Advanced Options (for login-restricted videos)'}
                    </button>

                    {showAdvanced && (
                        <div className="mt-3 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Upload cookies.txt (Use only if analysis fails due to login)
                            </label>
                            <input
                                type="file"
                                accept=".txt"
                                onChange={handleCookieChange}
                                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                                disabled={isLoading}
                            />
                            {cookieFile && (
                                <p className="mt-1 text-sm text-gray-600">Selected: {cookieFile.name}</p>
                            )}
                            <div className="mt-2 text-xs text-yellow-800 space-y-1">
                                <p>⚠️ <strong>Security Warning:</strong></p>
                                <ul className="list-disc list-inside ml-1">
                                    <li>Cookies contain your login session.</li>
                                    <li>Used <strong>temporarily</strong> for this analysis only.</li>
                                    <li>Deleted immediately from server after processing.</li>
                                    <li>Do not use on shared computers. If unsure, please upload MP3 instead.</li>
                                </ul>
                            </div>
                        </div>
                    )}
                </div>
            </form>
        </div>
    );
}
