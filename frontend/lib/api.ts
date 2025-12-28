export interface Tab {
    frets: string[];
}

export interface Bar {
    bar: number;
    chord: string;
    tab: Tab;
}

export interface AnalysisResult {
    bpm: number;
    duration_sec: number;
    time_signature: string;
    key: string;
    bars: Bar[];
    audio_url?: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

async function fetchWithTimeout(resource: string, options: RequestInit & { timeout?: number }) {
    const { timeout = 180000, ...fetchOptions } = options;

    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(resource, {
            ...fetchOptions,
            signal: controller.signal,
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        if (error instanceof DOMException && error.name === 'AbortError') {
            throw new Error(`Request timed out after ${timeout / 1000}s`);
        }
        throw error;
    }
}

function normalizeAnalysisResult(data: any): AnalysisResult {
    if (data.audio_url && typeof data.audio_url === 'string' && !data.audio_url.startsWith('http')) {
        // Ensure absolute URL for audio
        const path = data.audio_url.startsWith('/') ? data.audio_url : `/${data.audio_url}`;
        data.audio_url = `${API_URL}${path}`;
    }
    return data as AnalysisResult;
}

export async function analyzeAudio(file: File): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetchWithTimeout(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
        timeout: 180000, // 3 minutes
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Analysis failed');
    }

    const data = await response.json();
    return normalizeAnalysisResult(data);
}

export async function analyzeYoutube(url: string, cookiesFile?: File): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append("url", url);
    if (cookiesFile) {
        formData.append("cookies", cookiesFile);
    }

    const response = await fetchWithTimeout(`${API_URL}/analyze/url`, {
        method: 'POST',
        // Note: Do NOT set Content-Type header here, let browser set it with boundary for FormData
        body: formData,
        timeout: 240000, // 4 minutes
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Analysis failed');
    }

    const data = await response.json();
    return normalizeAnalysisResult(data);
}
