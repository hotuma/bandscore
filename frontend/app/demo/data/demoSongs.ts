import type { TimedChord } from "@/lib/chordTimeline";

const demoTimeline: TimedChord[] = [
    { name: "N.C.", startSec: 0.0, endSec: 2.15, frets: ["x", "x", "x", "x", "x", "x"] },
    { name: "E", startSec: 2.15, endSec: 4.8, frets: ["0", "2", "2", "1", "0", "0"] },
    { name: "G#m", startSec: 4.8, endSec: 7.2, frets: ["4", "6", "6", "4", "4", "4"] },
    { name: "A", startSec: 7.2, endSec: 9.6, frets: ["x", "0", "2", "2", "2", "0"] },
    { name: "B", startSec: 9.6, endSec: 12.0, frets: ["x", "2", "4", "4", "4", "2"] },
    { name: "E", startSec: 12.0, endSec: 14.4, frets: ["0", "2", "2", "1", "0", "0"] },
    { name: "G#m", startSec: 14.4, endSec: 16.8, frets: ["4", "6", "6", "4", "4", "4"] },
    { name: "A", startSec: 16.8, endSec: 19.2, frets: ["x", "0", "2", "2", "2", "0"] },
    { name: "Am", startSec: 19.2, endSec: 21.6, frets: ["x", "0", "2", "2", "1", "0"] },

    { name: "E", startSec: 21.6, endSec: 24.0, frets: ["0", "2", "2", "1", "0", "0"] },
    { name: "G#m", startSec: 24.0, endSec: 26.4, frets: ["4", "6", "6", "4", "4", "4"] },
    { name: "A", startSec: 26.4, endSec: 28.8, frets: ["x", "0", "2", "2", "2", "0"] },
    { name: "B", startSec: 28.8, endSec: 31.2, frets: ["x", "2", "4", "4", "4", "2"] },
    { name: "C#m", startSec: 31.2, endSec: 33.6, frets: ["x", "4", "6", "6", "5", "4"] },
    { name: "G#m", startSec: 33.6, endSec: 36.0, frets: ["4", "6", "6", "4", "4", "4"] },
    { name: "A", startSec: 36.0, endSec: 38.4, frets: ["x", "0", "2", "2", "2", "0"] },
    { name: "B", startSec: 38.4, endSec: 40.8, frets: ["x", "2", "4", "4", "4", "2"] },
    { name: "E", startSec: 40.8, endSec: 45.0, frets: ["0", "2", "2", "1", "0", "0"] },
];

export const demoSongs = [
    {
        videoId: "JegJ6cSsUgg",
        title: "Demo Song (E Major Progression)",
        supportedUrl: "https://www.youtube.com/watch?v=JegJ6cSsUgg",
        defaultOffsetSec: 0,
        timeline: demoTimeline,
    },
] as const;
