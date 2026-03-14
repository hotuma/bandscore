"use client";

import Soundfont from "soundfont-player";

// Use 'any' for the instrument type to avoid import issues if type definitions are missing or named differently
type GuitarInstrument = any;

let audioContext: AudioContext | null = null;
let masterGain: GainNode | null = null;
let guitarPromise: Promise<GuitarInstrument | null> | null = null;

function ensureAudioContext(): AudioContext {
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
