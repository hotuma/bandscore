export interface Tab {
    frets: string[];
}

export interface Bar {
    bar: number;
    chord: string;
    tab: Tab | null;
    start_sec: number;
    end_sec: number;
}

export type AnalyzeMode = 'PREVIEW' | 'EARLY_ACCESS' | 'FULL';

export interface AnalysisResult {
    bpm: number;
    duration_sec: number;
    time_signature: string;
    key: string;
    bars: Bar[] | null; // Nullable for Preview
    audio_url?: string;
    beat_times?: number[]; // すべてのビートのタイムスタンプ（秒）
    // New fields
    mode?: AnalyzeMode;
    is_preview?: boolean;
    analyzed_duration_sec?: number;
    export_allowed?: boolean;
}

const API_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

async function fetchWithTimeout(resource: string, options: RequestInit & { timeout?: number }) {
    const { timeout = 180000, ...fetchOptions } = options;
    const timeoutController = new AbortController();
    const id = setTimeout(() => timeoutController.abort(), timeout);

    // If caller provided a signal, make it also abort the timeoutController.
    const externalSignal = fetchOptions.signal as AbortSignal | undefined;
    const onExternalAbort = () => timeoutController.abort();

    if (externalSignal) {
        if (externalSignal.aborted) {
            timeoutController.abort();
        } else {
            externalSignal.addEventListener('abort', onExternalAbort, { once: true });
        }
    }

    try {
        const response = await fetch(resource, {
            ...fetchOptions,
            // Always use the timeoutController signal (it will also be aborted by externalSignal)
            signal: timeoutController.signal,
        });
        clearTimeout(id);
        return response;
    } catch (error: any) {
        clearTimeout(id);
        if (error instanceof DOMException && error.name === 'AbortError') {
            throw new Error(`Request aborted or timed out after ${timeout / 1000}s`);
        }
        throw error;
    } finally {
        if (externalSignal) {
            externalSignal.removeEventListener('abort', onExternalAbort);
        }
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

export interface AnalyzeAudioOptions {
    signal?: AbortSignal;
    onProgress?: (progress: number) => void; // 0-1 scale
    timeoutMs?: number;
}

export async function analyzeAudio(
    file: File,
    mode: AnalyzeMode = 'EARLY_ACCESS',
    opts: AnalyzeAudioOptions = {}
): Promise<AnalysisResult> {
    const { signal, onProgress, timeoutMs = 300000 } = opts;

    const formData = new FormData();
    formData.append('file', file);

    let url = `${API_URL}/analyze`;
    if (mode === 'PREVIEW') {
        url = `${API_URL}/analyze/preview`;
    } else {
        formData.append('mode', mode);
    }

    // Step 1: Submit job (backend returns 202 with job_id)
    const submitResponse = await fetchWithTimeout(url, {
        method: 'POST',
        body: formData,
        timeout: 60000, // 60s for upload only
        signal,
    });

    if (!submitResponse.ok) {
        const errorData = await submitResponse.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Analysis failed');
    }

    const submitData = await submitResponse.json();
    const jobId = submitData.job_id;

    // Backward compatibility: if no job_id, return direct result
    if (!jobId) {
        return normalizeAnalysisResult(submitData);
    }

    // Step 2: Poll for completion
    const pollInterval = 1500; // 1.5 seconds
    const startTime = Date.now();
    let lastProgress = -1;
    let lastUpdateTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
        if (signal?.aborted) throw new Error('Cancelled');

        await new Promise(r => setTimeout(r, pollInterval));
        if (signal?.aborted) throw new Error('Cancelled');

        const statusRes = await fetch(`${API_URL}/analyze/status/${jobId}`, { signal });
        if (statusRes.status === 404) {
            throw new Error('Job not found. Please try again.');
        }
        const statusData = await statusRes.json();

        // Update progress (normalize 0-100 to 0-1)
        const rawProgress = typeof statusData.progress === 'number' ? statusData.progress : 0;
        const progress = rawProgress > 1 ? rawProgress / 100 : rawProgress;
        onProgress?.(progress);

        // Startup check: if no started_at after 15s, job is stuck
        if (!statusData.started_at && Date.now() - startTime > 15000) {
            throw new Error('Server is not responding. Backend may not be running.');
        }

        // Stall check: if progress hasn't moved in 120s, timeout
        if (progress > lastProgress) {
            lastProgress = progress;
            lastUpdateTime = Date.now();
        } else if (Date.now() - lastUpdateTime > 120000) {
            throw new Error('Analysis timed out (no progress for 2 minutes). Please try again.');
        }

        if (statusData.status === 'error') {
            throw new Error(statusData.error || 'Analysis failed on server');
        }

        if (statusData.status === 'done') {
            const resultRes = await fetch(`${API_URL}/analyze/result/${jobId}`, { signal });
            if (!resultRes.ok) throw new Error('Failed to fetch analysis result');
            const resultData = await resultRes.json();
            return normalizeAnalysisResult(resultData);
        }
    }

    throw new Error('Analysis timed out. Please try again.');
}

export async function analyzeYoutube(url: string, cookiesFile?: File, mode: AnalyzeMode = 'EARLY_ACCESS'): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append("url", url);
    if (cookiesFile) {
        formData.append("cookies", cookiesFile);
    }

    let endpoint = `${API_URL}/analyze/url`;
    if (mode === 'PREVIEW') {
        endpoint = `${API_URL}/analyze/url/preview`;
        // Do NOT append mode
    } else {
        formData.append("mode", mode);
    }

    // Step 1: Submit job (backend returns 202 with job_id)
    const response = await fetchWithTimeout(endpoint, {
        method: 'POST',
        // Note: Do NOT set Content-Type header here, let browser set it with boundary for FormData
        body: formData,
        timeout: 240000, // 4 minutes
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Analysis failed');
    }

    const submitData = await response.json();
    const jobId = submitData.job_id;

    if (!jobId) {
        // If response is not a job submission (e.g. direct result), return as-is
        return normalizeAnalysisResult(submitData);
    }

    // Step 2: Poll for completion
    const maxPollTime = 300000; // 5 minutes
    const pollInterval = 2000; // 2 seconds
    const startTime = Date.now();

    while (Date.now() - startTime < maxPollTime) {
        await new Promise(r => setTimeout(r, pollInterval));

        const statusRes = await fetch(`${API_URL}/analyze/status/${jobId}`);
        if (statusRes.status === 404) {
            throw new Error('Job not found. Please try again.');
        }
        const statusData = await statusRes.json();

        if (statusData.status === 'error') {
            throw new Error(statusData.error || 'Analysis failed');
        }

        if (statusData.status === 'done') {
            // Step 3: Fetch result
            const resultRes = await fetch(`${API_URL}/analyze/result/${jobId}`);
            if (!resultRes.ok) {
                throw new Error('Failed to fetch analysis result');
            }
            const resultData = await resultRes.json();
            return normalizeAnalysisResult(resultData);
        }
    }

    throw new Error('Analysis timed out. Please try again.');
}

// ============================================================================
// BPM調整用API関数
// ============================================================================

export interface BPMAdjustRequest {
    bars: Bar[];
    original_bpm: number;
    adjusted_bpm: number;
}

export interface BPMAdjustResponse {
    adjusted_bars: Bar[];
    original_bpm: number;
    adjusted_bpm: number;
    adjustment_ratio: number;
}

export async function adjustBPM(request: BPMAdjustRequest): Promise<BPMAdjustResponse> {
    const response = await fetch(`${API_URL}/bpm/adjust`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'BPM adjustment failed');
    }

    return await response.json();
}

// ============================================================================
// BPM調整用API関数ここまで
// ============================================================================
