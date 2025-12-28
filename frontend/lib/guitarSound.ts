"use client";

import Soundfont from "soundfont-player";

// Use 'any' for the instrument type to avoid import issues if type definitions are missing or named differently
type GuitarInstrument = any;

let audioContext: AudioContext | null = null;
let guitarPromise: Promise<GuitarInstrument | null> | null = null;

/**
 * Initialize AudioContext & Guitar instrument only on the client side.
 */
export function getGuitar(): Promise<GuitarInstrument | null> {
    if (typeof window === "undefined") {
        // SSR check
        return Promise.resolve(null);
    }

    if (!audioContext) {
        const AC = window.AudioContext || (window as any).webkitAudioContext;
        audioContext = new AC();
    }

    if (!guitarPromise) {
        guitarPromise = Soundfont.instrument(audioContext!, "acoustic_guitar_steel")
            .then((instrument) => instrument)
            .catch((err) => {
                console.error("Failed to load guitar soundfont:", err);
                return null;
            });
    }

    return guitarPromise;
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

/**
 * Play a chord from TAB frets using SoundFont.
 * @param frets e.g. [3,2,0,0,0,3] / ["3","2","0","0","0","3"]
 * @param durationSec Duration in seconds
 */
export async function playChordFromTabWithSoundFont(
    frets: Array<number | string | null | undefined>,
    durationSec: number = 2.0
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

    const now = audioContext.currentTime;

    midiNotes.forEach((midi, idx) => {
        // Slight stagger for a realistic "strum" effect (20ms per string)
        const stagger = idx * 0.02;
        guitar.play(midi, now + stagger, {
            duration: durationSec,
        });
    });
}
