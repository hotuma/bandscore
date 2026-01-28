import React, { useRef, useState, useEffect } from 'react';
import { AnalysisResult } from '../lib/api';
import { setChordVolume } from '../lib/chordAudio';
import { playChordFromTabWithSoundFont } from '../lib/guitarSound';
import { useMemo } from 'react';
// import { analysisResultToTimedChords } from '../lib/chordTimeline'; // unused


interface ResultDisplayProps {
    result: AnalysisResult;
    audioUrl: string | null;
}

export default function ResultDisplay({ result, audioUrl }: ResultDisplayProps) {
    const audioRef = useRef<HTMLAudioElement>(null);
    const [currentBarIndex, setCurrentBarIndex] = useState<number>(-1);
    // UI State
    const [isPlaying, setIsPlaying] = useState(false);
    // Refs
    const rafIdRef = useRef<number | null>(null);

    // Auto-scroll suppression
    const suppressScrollRef = useRef<number>(0);
    const lastUserActionRef = useRef<number>(0);

    // UX State
    const [offsetSec, setOffsetSec] = useState<number>(0);
    const [autoScroll, setAutoScroll] = useState<boolean>(true);
    const [autoChord, setAutoChord] = useState<boolean>(true);
    const [chordVolume, setChordVolumeState] = useState<number>(0.8);
    const [showDebug, setShowDebug] = useState<boolean>(false);

    // Automatically apply a small offset for remote (server-processed) audio
    // This compensates for MP3 encoding/decoding latency differences between librosa and browser
    useEffect(() => {
        if (audioUrl && !audioUrl.startsWith('blob:')) {
            setOffsetSec(0.2); // Start with +0.2s for URLs (found to be a good average for yt-dlp mp3s)
        } else {
            setOffsetSec(0);
        }
    }, [audioUrl]);

    // Refs for auto-scrolling
    const barRefs = useRef<(HTMLDivElement | null)[]>([]);
    const lastPlayedBarRef = useRef<number>(-1);

    const isPreview = result?.is_preview === true || result?.mode === 'PREVIEW';

    // Format duration (seconds -> mm:ss)
    const formatDuration = (seconds: number) => {
        const min = Math.floor(seconds / 60);
        const sec = Math.floor(seconds % 60);
        return `${min}:${sec.toString().padStart(2, '0')}`;
    };

    // Safe bars access (renamed to safeBars to ensure no confusion/stale usage)
    const safeBars = useMemo(() => result?.bars ?? [], [result]);

    // Track actual audio duration for accurate sync
    const [audioDurationSec, setAudioDurationSec] = useState<number>(0);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const onLoadedMeta = () => {
            const d = Number(audio.duration);
            if (Number.isFinite(d) && d > 0) setAudioDurationSec(d);
        };

        audio.addEventListener("loadedmetadata", onLoadedMeta);
        // Handle case where metadata is already loaded
        if (audio.readyState >= 1) {
            onLoadedMeta();
        }

        return () => audio.removeEventListener("loadedmetadata", onLoadedMeta);
    }, [audioUrl]);

    // Calculate seconds per bar robustly (Total Duration / Number of Bars)
    // This relies on the backend returning a fixed-grid "bars" array.
    const secondsPerBar = useMemo(() => {
        const n = safeBars.length;
        if (!n) return 1;

        const audioDur = audioDurationSec;
        // Preview uses analyzed duration (audio might be longer)
        const analyzedDur = Number(result?.analyzed_duration_sec ?? 0);
        const fallbackDur = Number(result?.duration_sec ?? 0);
        const analysisDur = (Number.isFinite(analyzedDur) && analyzedDur > 0) ? analyzedDur : fallbackDur;

        const dur = isPreview
            ? analysisDur
            : ((Number.isFinite(audioDur) && audioDur > 0) ? audioDur : analysisDur);

        if (!dur) return 1;
        return dur / n;
    }, [safeBars.length, result?.duration_sec, result?.analyzed_duration_sec, audioDurationSec, isPreview]);


    // --- Audio Event Listeners (State Management Only) ---
    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const onPlay = () => setIsPlaying(true);
        const onPause = () => setIsPlaying(false);
        const onEnded = () => {
            setIsPlaying(false);
            setCurrentBarIndex(-1);
        };

        audio.addEventListener('play', onPlay);
        audio.addEventListener('pause', onPause);
        audio.addEventListener('ended', onEnded);

        // Initial state check
        if (!audio.paused && !audio.ended) {
            setIsPlaying(true);
        }

        return () => {
            audio.removeEventListener('play', onPlay);
            audio.removeEventListener('pause', onPause);
            audio.removeEventListener('ended', onEnded);
        };
    }, [audioUrl]); // Re-bind if audio source changes

    // Preview: Stop playback if it exceeds analyzed range to prevent sync drift or errors
    useEffect(() => {
        if (!isPreview) return;
        const audio = audioRef.current;
        if (!audio) return;
        const limit = Number(result?.analyzed_duration_sec ?? result?.duration_sec ?? 0);
        if (!Number.isFinite(limit) || limit <= 0) return;

        const onTimeUpdate = () => {
            if (audio.currentTime >= limit) {
                audio.pause();
                audio.currentTime = limit;
                setIsPlaying(false);
                setCurrentBarIndex(safeBars.length ? safeBars.length - 1 : -1);
                suppressScrollRef.current = Date.now() + 1000;
            }
        };
        audio.addEventListener('timeupdate', onTimeUpdate);

        // Prevent manual seeking beyond limit
        const onSeeking = () => {
            if (audio.currentTime >= limit) {
                audio.currentTime = limit;
                // If dragging, this keeps it clamped
            }
        };
        audio.addEventListener('seeking', onSeeking);

        return () => {
            audio.removeEventListener('timeupdate', onTimeUpdate);
            audio.removeEventListener('seeking', onSeeking);
        };
    }, [isPreview, result?.analyzed_duration_sec, result?.duration_sec, safeBars.length]);

    // Ensure AutoChord re-triggers after seek
    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;
        const onSeeked = () => { lastPlayedBarRef.current = -1; };
        audio.addEventListener('seeked', onSeeked);
        return () => audio.removeEventListener('seeked', onSeeked);
    }, [audioUrl]);

    // --- Main Sync Loop (The "King" RAF) ---
    useEffect(() => {
        if (!isPlaying) {
            // Cleanup checks
            if (rafIdRef.current) {
                cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
            }
            return;
        }

        const loop = () => {
            try {
                if (audioRef.current) {
                    const currentTime = audioRef.current.currentTime;
                    // Changed to SUBTRACT offset to delay the visual relative to audio
                    // offsetSec is typically positive latency compensation
                    const effectiveTime = currentTime - offsetSec;

                    // Calculate current bar (0-indexed)
                    let index = -1;

                    if (effectiveTime > 0) {
                        index = Math.floor(effectiveTime / secondsPerBar);
                    }

                    // Clamp index to valid range
                    if (safeBars.length === 0) {
                        index = -1;
                    } else if (index >= safeBars.length) {
                        index = safeBars.length - 1;
                    }

                    if (index < 0) {
                        index = -1;
                    }

                    setCurrentBarIndex(prev => {
                        if (prev !== index) return index;
                        return prev;
                    });
                }
            } catch (e) {
                console.error("ResultDisplay RAF loop error:", e);
            }
            rafIdRef.current = requestAnimationFrame(loop);
        };

        rafIdRef.current = requestAnimationFrame(loop);

        return () => {
            if (rafIdRef.current) {
                cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
            }
        };
    }, [isPlaying, result, offsetSec, secondsPerBar, safeBars]);

    // Auto-scroll effect
    useEffect(() => {
        const now = Date.now();
        if (
            autoScroll &&
            currentBarIndex >= 0 &&
            barRefs.current[currentBarIndex] &&
            now > suppressScrollRef.current
        ) {
            barRefs.current[currentBarIndex]?.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
                inline: 'center'
            });
        }
    }, [currentBarIndex, autoScroll]);

    // Handle user scroll to suppress auto-scroll temporarily
    useEffect(() => {
        const handleScroll = () => {
            // Only suppress if currently playing and auto-scroll is on
            if (isPlaying && autoScroll) {
                suppressScrollRef.current = Date.now() + 2000; // Suppress for 2s
            }
        };
        window.addEventListener('wheel', handleScroll);
        window.addEventListener('touchmove', handleScroll);
        return () => {
            window.removeEventListener('wheel', handleScroll);
            window.removeEventListener('touchmove', handleScroll);
        }
    }, [isPlaying, autoScroll]);

    // Auto Chord Playback (Synchronized with visual grid)
    useEffect(() => {
        if (!autoChord) return;

        if (currentBarIndex < 0) return;

        // Debounce/Check if playing?
        // If we want it strictly during playback:
        if (!isPlaying) return;

        // Prevent re-triggering the same bar repeatedly
        if (lastPlayedBarRef.current === currentBarIndex) return;
        lastPlayedBarRef.current = currentBarIndex;

        const bar = safeBars[currentBarIndex];
        const frets = bar?.tab?.frets;
        if (!frets) return;

        // Sustain control: Play for the duration of the bar (sustainSec)
        const sustainSec = Math.min(
            1.8,                         // Upper limit (prevent too long sustain)
            Math.max(0.25, secondsPerBar * 0.95) // Lower limit (prevent staccato), 0.95 for legato
        );

        playChordFromTabWithSoundFont(frets, {
            durationSec: sustainSec,
            gain: chordVolume
        }).catch((e) => {
            console.error("Failed to play chord", e);
        });
    }, [autoChord, isPlaying, currentBarIndex, safeBars, secondsPerBar, chordVolume]);

    // Initialize Volume (side-effect)
    useEffect(() => {
        setChordVolume(chordVolume);
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const handleBarClick = (barIndex: number) => {
        const bar = safeBars[barIndex];
        if (!bar) return;
        const frets = bar.tab?.frets;

        console.log("Clicked bar", bar.bar, "chord", bar.chord, "frets", frets);

        // Play chord sound
        if (frets) {
            playChordFromTabWithSoundFont(frets, { gain: chordVolume }).catch((e) => {
                console.error("Failed to play chord", e);
            });
        }

        if (audioRef.current) {
            // Adjust seek time by subtracting offset so effective time matches bar start
            // Adjust seek time by subtracting offset so effective time matches bar start
            const targetTime = (barIndex * secondsPerBar) - offsetSec;
            audioRef.current.currentTime = Math.max(0, targetTime);
            lastPlayedBarRef.current = -1; // ensure chord triggers after manual seek
            audioRef.current.play();
        }
    };

    return (
        <div className="mt-8 space-y-8 relative">
            {/* Audio Player & Controls */}
            {audioUrl && (
                <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex flex-col items-center gap-4">
                    <audio
                        ref={audioRef}
                        src={audioUrl}
                        controls
                        className="w-full max-w-md"
                    />

                    {/* Playback Controls */}
                    <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-gray-600 bg-gray-50 px-4 py-2 rounded-lg border border-gray-100">
                        {/* Auto Scroll Toggle */}
                        <label className="flex items-center gap-2 cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={autoScroll}
                                onChange={(e) => setAutoScroll(e.target.checked)}
                                className="rounded text-blue-600 focus:ring-blue-500"
                            />
                            <span className="font-medium">Auto Scroll</span>
                        </label>

                        <div className="w-px h-4 bg-gray-300 hidden sm:block"></div>

                        {/* Auto Chord Toggle */}
                        <label className="flex items-center gap-2 cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={autoChord}
                                onChange={(e) => setAutoChord(e.target.checked)}
                                className="rounded text-blue-600 focus:ring-blue-500"
                            />
                            <span className="font-medium">Auto Chord</span>
                        </label>

                        <div className="w-px h-4 bg-gray-300 hidden sm:block"></div>

                        {/* Volume Slider */}
                        <div className="flex items-center gap-2">
                            <span className="font-medium">Vol</span>
                            <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.01}
                                value={chordVolume}
                                onChange={(e) => {
                                    const v = Number(e.target.value);
                                    setChordVolumeState(v);
                                    setChordVolume(v);
                                }}
                                className="w-20"
                            />
                            <span className="w-8 text-center">{Math.round(chordVolume * 100)}%</span>
                        </div>

                        <div className="w-px h-4 bg-gray-300 hidden sm:block"></div>

                        {/* Offset Adjustment */}
                        <div className="flex items-center gap-3">
                            <span className="font-medium">Offset:</span>
                            <button
                                onClick={() => setOffsetSec(prev => Math.max(-0.5, Number((prev - 0.1).toFixed(1))))}
                                className="w-6 h-6 flex items-center justify-center bg-white border border-gray-300 rounded hover:bg-gray-50 font-mono"
                            >-</button>
                            <span className="font-mono w-12 text-center">{`${offsetSec > 0 ? '+' : ''}${offsetSec.toFixed(1)}s`}</span>
                            <button
                                onClick={() => setOffsetSec(prev => Math.min(0.5, Number((prev + 0.1).toFixed(1))))}
                                className="w-6 h-6 flex items-center justify-center bg-white border border-gray-300 rounded hover:bg-gray-50 font-mono"
                            >+</button>
                        </div>

                        <div className="w-px h-4 bg-gray-300 hidden sm:block"></div>

                        {/* Debug Toggle */}
                        <label className="flex items-center gap-2 cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={showDebug}
                                onChange={(e) => setShowDebug(e.target.checked)}
                                className="rounded text-gray-500 focus:ring-gray-400"
                            />
                            <span className="text-xs text-gray-400">Debug</span>
                        </label>
                    </div>
                </div>
            )}

            {/* Header Stats */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex justify-around items-center">
                <div className="text-center">
                    <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-1">BPM</div>
                    <div className="text-3xl font-black text-gray-800">{result.bpm}</div>
                </div>
                <div className="w-px h-12 bg-gray-100"></div>
                <div className="text-center">
                    <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-1">Duration</div>
                    <div className="text-3xl font-black text-gray-800">{formatDuration(result.duration_sec)}</div>
                </div>
                <div className="w-px h-12 bg-gray-100"></div>
                <div className="text-center">
                    <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-1">Signature</div>
                    <div className="text-3xl font-black text-gray-800">{result.time_signature}</div>
                </div>
                <div className="w-px h-12 bg-gray-100"></div>
                <div className="text-center">
                    <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-1">Bars</div>
                    <div className="text-3xl font-black text-gray-800">{safeBars.length}</div>
                </div>
            </div>

            {/* Chord Progression Overview (Horizontal) */}
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4 tracking-wider">Chord Progression</h3>
                <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-2">
                    {safeBars.map((bar, index) => (
                        <div
                            key={`${index}-${bar.bar}-${bar.chord}`}
                            onClick={() => handleBarClick(index)}
                            className={`flex flex-col items-center p-2 rounded border cursor-pointer transition-colors duration-200 ${index === currentBarIndex
                                ? 'bg-blue-600 border-blue-600 text-white shadow-md scale-110 transform z-10'
                                : 'bg-gray-50 border-gray-100 hover:bg-gray-100'
                                }`}
                        >
                            <span className={`text-[10px] mb-1 ${index === currentBarIndex ? 'text-blue-100' : 'text-gray-400'}`}>{bar.bar}</span>
                            <span className={`font-bold text-sm sm:text-base ${index === currentBarIndex ? 'text-white' : 'text-gray-700'}`} translate="no">{bar.chord}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Detailed TAB View (Cards) */}
            <div>
                <h3 className="text-sm font-bold text-gray-400 uppercase mb-4 tracking-wider">Guitar TABs</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {safeBars.map((bar, index) => (
                        <div
                            key={`${index}-${bar.bar}-${bar.chord}`}
                            ref={el => { barRefs.current[index] = el; }}
                            onClick={() => handleBarClick(index)}
                            className={`rounded-xl border shadow-sm overflow-hidden transition-all duration-200 cursor-pointer ${index === currentBarIndex
                                ? 'border-blue-500 ring-2 ring-blue-200 shadow-lg transform scale-[1.02]'
                                : 'bg-white border-gray-200 hover:shadow-md'
                                }`}
                        >
                            <div className={`px-4 py-2 border-b flex justify-between items-center ${index === currentBarIndex ? 'bg-blue-50 border-blue-100' : 'bg-gray-50 border-gray-100'
                                }`}>
                                <span className={`text-xs font-bold ${index === currentBarIndex ? 'text-blue-600' : 'text-gray-500'}`}>Bar {bar.bar}</span>
                                <span className={`text-lg font-black ${index === currentBarIndex ? 'text-blue-700' : 'text-blue-600'}`} translate="no">{bar.chord}</span>
                            </div>
                            <div className={`p-4 flex justify-center ${index === currentBarIndex ? 'bg-blue-50/30' : 'bg-white'}`}>
                                {bar.tab ? (
                                    <div className={`font-mono text-sm leading-relaxed p-3 rounded border inline-block ${index === currentBarIndex
                                        ? 'bg-white border-blue-200 text-gray-800'
                                        : 'bg-gray-50 border-gray-100 text-gray-600'
                                        }`}>
                                        <div className="flex gap-2">
                                            <span className="text-gray-400 select-none">e|</span>
                                            <span className="font-bold">{bar.tab.frets[5] === 'x' ? 'x' : bar.tab.frets[5]}</span>
                                            <span className="text-gray-300 select-none">---</span>
                                        </div>
                                        <div className="flex gap-2">
                                            <span className="text-gray-400 select-none">B|</span>
                                            <span className="font-bold">{bar.tab.frets[4] === 'x' ? 'x' : bar.tab.frets[4]}</span>
                                            <span className="text-gray-300 select-none">---</span>
                                        </div>
                                        <div className="flex gap-2">
                                            <span className="text-gray-400 select-none">G|</span>
                                            <span className="font-bold">{bar.tab.frets[3] === 'x' ? 'x' : bar.tab.frets[3]}</span>
                                            <span className="text-gray-300 select-none">---</span>
                                        </div>
                                        <div className="flex gap-2">
                                            <span className="text-gray-400 select-none">D|</span>
                                            <span className="font-bold">{bar.tab.frets[2] === 'x' ? 'x' : bar.tab.frets[2]}</span>
                                            <span className="text-gray-300 select-none">---</span>
                                        </div>
                                        <div className="flex gap-2">
                                            <span className="text-gray-400 select-none">A|</span>
                                            <span className="font-bold">{bar.tab.frets[1] === 'x' ? 'x' : bar.tab.frets[1]}</span>
                                            <span className="text-gray-300 select-none">---</span>
                                        </div>
                                        <div className="flex gap-2">
                                            <span className="text-gray-400 select-none">E|</span>
                                            <span className="font-bold">{bar.tab.frets[0] === 'x' ? 'x' : bar.tab.frets[0]}</span>
                                            <span className="text-gray-300 select-none">---</span>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="h-32 flex items-center justify-center text-gray-400 text-sm italic">
                                        No TAB
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Debug Overlay */}
            {showDebug && (
                <div className="fixed bottom-4 left-4 bg-black/80 text-white p-4 rounded-lg text-xs font-mono z-50 shadow-xl backdrop-blur-sm">
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                        <span className="text-gray-400">Time:</span>
                        <span>{audioRef.current?.currentTime.toFixed(3)}s</span>

                        <span className="text-gray-400">Effective:</span>
                        <span className="text-green-400">{(audioRef.current ? audioRef.current.currentTime - offsetSec : 0).toFixed(3)}s</span>

                        <span className="text-gray-400">Bar Idx:</span>
                        <span className="text-blue-400">{currentBarIndex}</span>

                        <span className="text-gray-400">Sec/Bar:</span>
                        <span>{secondsPerBar.toFixed(3)}s</span>

                        <span className="text-gray-400">Offset:</span>
                        <span className="text-yellow-400">{offsetSec.toFixed(2)}s</span>

                        <span className="text-gray-400">AudioDur:</span>
                        <span>{audioDurationSec.toFixed(3)}s</span>
                        <span className="text-gray-400">AnalysisDur:</span>
                        <span>
                            {(
                                isPreview
                                    ? (Number(result?.analyzed_duration_sec ?? result?.duration_sec ?? 0))
                                    : (Number(audioDurationSec || result?.duration_sec || 0))
                            ).toFixed(3)}s
                        </span>
                    </div>
                </div>
            )}

            {/* Floating Player Controls (Visible when AutoScroll is ON) */}
            {autoScroll && (
                <div className="fixed bottom-6 right-6 z-[100] flex flex-col gap-2 items-end">
                    <div className="bg-gray-900/90 backdrop-blur text-white p-3 rounded-2xl shadow-2xl border border-gray-700 flex items-center gap-4 transition-all hover:bg-gray-900">
                        {/* Stop Button (Primary) */}
                        <button
                            onClick={() => {
                                if (audioRef.current) {
                                    audioRef.current.pause();
                                    audioRef.current.currentTime = 0;
                                    setIsPlaying(false);
                                    audioRef.current.currentTime = 0;
                                    setIsPlaying(false);
                                    lastPlayedBarRef.current = -1;
                                    suppressScrollRef.current = Date.now() + 1000;
                                }
                            }}
                            className="flex flex-col items-center gap-1 group px-2"
                            title="Stop"
                        >
                            <div className="w-8 h-8 flex items-center justify-center bg-red-500/20 rounded-full group-hover:bg-red-500 transition-colors">
                                <div className="w-3 h-3 bg-red-500 rounded-sm group-hover:bg-white transition-colors"></div>
                            </div>
                            <span className="text-[10px] font-bold text-red-400 group-hover:text-red-300">STOP</span>
                        </button>

                        <div className="w-px h-8 bg-gray-700"></div>

                        {/* Play/Pause */}
                        <button
                            onClick={() => {
                                if (audioRef.current) {
                                    if (isPlaying) {
                                        audioRef.current.pause();
                                    } else {
                                        audioRef.current.play();
                                    }
                                    suppressScrollRef.current = Date.now() + 1000;
                                }
                            }}
                            className="flex flex-col items-center gap-1 min-w-[3rem]"
                        >
                            {isPlaying ? (
                                <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
                            ) : (
                                <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
                            )}
                            <span className="text-[10px] text-gray-400">{isPlaying ? 'PAUSE' : 'PLAY'}</span>
                        </button>

                        <div className="w-px h-8 bg-gray-700"></div>

                        {/* Toggle Scroll */}
                        <button
                            onClick={() => {
                                setAutoScroll(!autoScroll);
                                suppressScrollRef.current = Date.now() + 1000;
                            }}
                            className={`flex flex-col items-center gap-1 px-2 ${autoScroll ? 'text-blue-400' : 'text-gray-400'}`}
                        >
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
                            <span className="text-[10px]">SCROLL ON</span>
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
