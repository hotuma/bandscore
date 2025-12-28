import { AnalysisResult } from './api';

export type TimedChord = {
    chordName: string;
    startSec: number;
    endSec: number;
    frets: (string)[]; // API returns string[] for frets (e.g. "x", "3")
};

export function analysisResultToTimedChords(
    analysis: AnalysisResult
): TimedChord[] {
    const { bpm, bars } = analysis;
    // Assuming 4/4 time signature if not specified or parsed differently
    // The current API result has time_signature as string "4/4"
    const timeSignatureParts = analysis.time_signature.split('/');
    const numerator = parseInt(timeSignatureParts[0], 10) || 4;

    const secondsPerBeat = 60 / bpm;
    const barDurationSec = secondsPerBeat * numerator;

    // Currently no audioOffsetSec in AnalysisResult, assuming 0
    const offsetSec = 0;

    const timeline: TimedChord[] = [];

    bars.forEach((bar, index) => {
        // In the current API, every bar has a chord, but we check just in case
        if (!bar.chord) return;

        const startSec = offsetSec + index * barDurationSec;
        const endSec = startSec + barDurationSec;

        timeline.push({
            chordName: bar.chord,
            startSec,
            endSec,
            frets: bar.tab?.frets || [],
        });
    });

    return timeline;
}
