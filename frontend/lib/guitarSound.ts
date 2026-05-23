"use client";

import Soundfont from "soundfont-player";

// Use 'any' for the instrument type to avoid import issues if type definitions are missing or named differently
type GuitarInstrument = any;

let audioContext: AudioContext | null = null;
let masterGain: GainNode | null = null;
let guitarPromise: Promise<GuitarInstrument | null> | null = null;
let _scheduleToken = 0;

/** シーク時に呼び出す: 進行中の scheduleStrumPattern ループをキャンセル */
export function invalidateScheduled(): void {
    _scheduleToken++;
}

export function ensureAudioContext(): AudioContext {
    if (!audioContext) {
        const AC = window.AudioContext || (window as any).webkitAudioContext;
        audioContext = new AC();
    }
    if (!masterGain) {
        masterGain = audioContext.createGain();
        masterGain.gain.value = 0.8;
        masterGain.connect(audioContext.destination);
    }
    return audioContext;
}

/**
 * Initialize AudioContext & Guitar instrument only on the client side.
 */
export function getGuitar(): Promise<GuitarInstrument | null> {
    if (typeof window === "undefined") {
        // SSR check
        return Promise.resolve(null);
    }

    const ctx = ensureAudioContext();

    if (!guitarPromise) {
        console.log("[SoundFont] Loading acoustic_guitar_steel...");
        guitarPromise = Soundfont.instrument(ctx, "acoustic_guitar_steel", {
            destination: masterGain!,
        })
            .then((instrument) => {
                console.log("[SoundFont] Loaded successfully");
                return instrument;
            })
            .catch((err) => {
                console.error("[SoundFont] Failed to load:", err);
                guitarPromise = null; // Allow retry on next call
                return null;
            });
    }

    return guitarPromise;
}

/**
 * Preload guitar soundfont. Call early (e.g. on component mount) to avoid
 * delays when the first chord needs to play.
 */
export function preloadGuitar(): void {
    if (typeof window === "undefined") return;
    getGuitar();
}

/**
 * Set the master volume for guitar chord playback.
 * Accepts values 0..N (values > 1.0 amplify beyond default level).
 */
export function setGuitarSoundVolume(volume: number): void {
    if (!audioContext || !masterGain) return;
    const v = Math.max(0, volume);
    masterGain.gain.setTargetAtTime(v, audioContext.currentTime, 0.01);
}

/**
 * Initialize AudioContext without loading the guitar instrument.
 * This is useful for timing/synchronization even when chord playback is disabled.
 */
export function initAudioContext(): AudioContext | null {
    if (typeof window === "undefined") {
        return null;
    }

    ensureAudioContext();
    console.log("[AudioContext] Initialized for synchronization");

    // Resume if suspended (browser autoplay policy)
    if (audioContext!.state === "suspended") {
        audioContext!.resume().catch((e) => {
            console.warn("[AudioContext] Resume failed:", e);
        });
    }

    return audioContext;
}

/**
 * Get the current time from the shared AudioContext (if initialized).
 * Used for synchronizing external schedulers.
 */
export function getAudioContextTime(): number | null {
    return audioContext ? audioContext.currentTime : null;
}


// Standard Tuning E2 A2 D3 G3 B3 E4 MIDI notes
const STANDARD_TUNING_MIDI = [40, 45, 50, 55, 59, 64];

/**
 * Turn frets array (e.g. ["3","2","0","0","0","3"] or "x") into MIDI notes.
 */
export function fretsToMidiNotes(
    frets: Array<number | string | null | undefined>,
    tuningMidi: number[] = STANDARD_TUNING_MIDI
): number[] {
    const notes: number[] = [];

    const numStrings = Math.min(frets.length, tuningMidi.length);

    for (let i = 0; i < numStrings; i++) {
        const f = frets[i];

        // Treat as muted
        if (f === "x" || f === "X" || f === "-" || f === null || f === undefined || f === "") {
            continue;
        }

        const fretNum =
            typeof f === "string"
                ? parseInt(f, 10)
                : f;

        if (Number.isNaN(fretNum) || fretNum < 0) continue;

        const midi = tuningMidi[i] + fretNum;
        notes.push(midi);
    }

    return notes;
}

export type PlayChordOptions = {
    durationSec?: number;
    gain?: number;
    whenSec?: number; // AudioContext absolute time
    strumSec?: number; // Stagger time per string (default 0.02)
};

export type StrokeDirection = 'down' | 'up';

export interface StrokeEvent {
    direction: StrokeDirection;
    offsetSec: number; // 小節先頭からの相対時間
    gain: number;      // アクセント係数
}

/**
 * Build a strum pattern for a bar of the given duration.
 * Returns DDUD pattern for standard bars, single down for short bars.
 */
function buildStrokeEvents(barDurationSec: number): StrokeEvent[] {
    if (!Number.isFinite(barDurationSec) || barDurationSec < 0.5) {
        return [{ direction: 'down', offsetSec: 0, gain: 1.0 }];
    }

    const base: Array<[StrokeDirection, number]> = [
        ['down', 1.0],
        ['down', 0.7],
        ['up',   0.6],
        ['down', 0.85],
    ];

    // 長いバー (> 3s) はパターンを2倍繰り返す
    const pattern = barDurationSec > 3.0 ? [...base, ...base] : base;
    const n = pattern.length;

    return pattern.map(([direction, gain], i) => ({
        direction,
        gain,
        offsetSec: (i / n) * barDurationSec,
    }));
}

/**
 * Play a single stroke (down or up) on the guitar.
 * Down: all strings low→high. Up: top 3 strings (G,B,E) high→low, softer.
 */
export async function playStroke(
    frets: Array<number | string | null | undefined>,
    direction: StrokeDirection,
    options?: PlayChordOptions
): Promise<void> {
    const guitar = await getGuitar();
    if (!guitar || !audioContext) return;

    if (audioContext.state === "suspended") {
        try { await audioContext.resume(); } catch (_) {}
    }

    const baseWhen = typeof options?.whenSec === "number" ? options.whenSec : audioContext.currentTime;
    const duration = options?.durationSec ?? 0.8;
    const gain = options?.gain ?? 1.0;

    if (direction === 'down') {
        const notes = fretsToMidiNotes(frets);
        notes.forEach((midi, idx) => {
            guitar.play(midi, baseWhen + idx * 0.012, { duration, gain });
        });
    } else {
        // アップ: 上3弦 (index 3,4,5 = G3,B3,E4) のみを高→低順
        const upFrets = Array.from({ length: Math.max(frets.length, 6) }, (_, i) =>
            i >= 3 ? (frets[i] ?? null) : null
        );
        const notes = fretsToMidiNotes(upFrets).reverse();
        notes.forEach((midi, idx) => {
            guitar.play(midi, baseWhen + idx * 0.010, { duration, gain: gain * 0.75 });
        });
    }
}

/**
 * Schedule a full strum pattern for one bar.
 * @param frets       Tab frets for the bar's chord
 * @param barStartCtxSec  AudioContext absolute time of bar start
 * @param barDurationSec  Duration of the bar in seconds
 */
export async function scheduleStrumPattern(
    frets: Array<number | string | null | undefined>,
    barStartCtxSec: number,
    barDurationSec: number
): Promise<void> {
    const token = _scheduleToken;
    const strokes = buildStrokeEvents(barDurationSec);
    for (let i = 0; i < strokes.length; i++) {
        if (_scheduleToken !== token) return;
        const stroke = strokes[i];
        const whenSec = barStartCtxSec + stroke.offsetSec;
        const nextOffset = strokes[i + 1]?.offsetSec ?? barDurationSec;
        const durationSec = Math.max(0.15, nextOffset - stroke.offsetSec);
        await playStroke(frets, stroke.direction, { whenSec, durationSec, gain: stroke.gain });
    }
}

/**
 * Play a chord from TAB frets using SoundFont.
 * @param frets e.g. [3,2,0,0,0,3] / ["3","2","0","0","0","3"]
 * @param options Options for playback (duration, etc.)
 */
export async function playChordFromTabWithSoundFont(
    frets: Array<number | string | null | undefined>,
    options?: PlayChordOptions
): Promise<void> {
    const guitar = await getGuitar();
    if (!guitar || !audioContext) return;

    const midiNotes = fretsToMidiNotes(frets);
    if (midiNotes.length === 0) return;

    // Resume context if suspended (e.g. due to user gesture policy)
    if (audioContext.state === "suspended") {
        try {
            await audioContext.resume();
        } catch (e) {
            console.warn("AudioContext resume failed:", e);
        }
    }

    const duration = options?.durationSec ?? 2.0;
    const gain = options?.gain ?? 1.0;
    const strum = options?.strumSec ?? 0.02;

    // NEW: scheduled playback support
    const baseWhen =
        typeof options?.whenSec === "number"
            ? options.whenSec
            : audioContext.currentTime;

    midiNotes.forEach((midi, idx) => {
        const stagger = idx * strum;
        // soundfont-player handles 'duration' by scheduling noteOff
        guitar.play(midi, baseWhen + stagger, {
            duration: duration,
            gain: gain,
        });
    });
}
