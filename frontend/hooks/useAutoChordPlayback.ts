"use client";

import { useEffect, useRef } from "react";
import { playChordFromTab } from "@/lib/chordAudio";
import type { TimedChord } from "@/lib/chordTimeline";

/**
 * 再生中の audio.currentTime に合わせて、
 * コードが変わった時だけ 1 回 playChordFromTab を呼ぶフック
 *
 * @param timingOffsetSec  負の値で「早めに」コードを鳴らす（秒）
 */
export function useAutoChordPlayback(
    audioElement: HTMLAudioElement | null,
    chordTimeline: TimedChord[],
    enabled: boolean,
    timingOffsetSec: number = 0
) {
    const currentIndexRef = useRef<number | null>(null);
    const prevTimeRef = useRef<number>(0);
    const lastTriggerTimeRef = useRef<number>(-999);

    useEffect(() => {
        if (!audioElement) return;
        if (!enabled) return;
        if (!chordTimeline || !chordTimeline.length) return;

        let rafId: number;

        const tick = () => {
            rafId = requestAnimationFrame(tick);

            if (audioElement.paused) return;

            const rawTime = audioElement.currentTime;
            const t = rawTime + timingOffsetSec;

            const prevTime = prevTimeRef.current;
            const timeline = chordTimeline;
            if (!timeline) return;
            const n = timeline.length;

            // --- シーク（大きく時間が飛んだ）を検出 ---
            const jumped = Math.abs(t - prevTime) > 0.2; // 200ms 以上ならシークとみなす
            prevTimeRef.current = t;

            let idx = currentIndexRef.current ?? -1;

            if (jumped || idx < 0 || idx >= n) {
                // シーク時は 0 から探し直し
                idx = timeline.findIndex(
                    (c) => t >= c.startSec && t < c.endSec
                );
            } else {
                // 通常時は「前後にだけ」進める

                // 前に進める（次のコード開始を超えたら進む）
                while (idx + 1 < n && t >= timeline[idx + 1].startSec) {
                    idx++;
                }

                // 巻き戻した場合は戻す
                while (idx > 0 && t < timeline[idx].startSec) {
                    idx--;
                }

                // それでも範囲外なら再検索
                if (
                    idx < 0 ||
                    idx >= n ||
                    !(t >= timeline[idx].startSec && t < timeline[idx].endSec)
                ) {
                    idx = timeline.findIndex(
                        (c) => t >= c.startSec && t < c.endSec
                    );
                }
            }

            if (idx === -1) {
                currentIndexRef.current = null;
                return;
            }

            // 本当に「今のコード index」が idx
            const prevIdx = currentIndexRef.current;

            if (idx !== prevIdx) {
                // クールタイム（直近150msは鳴らさない）
                const now = rawTime;
                const minInterval = 0.15;
                if (now - lastTriggerTimeRef.current < minInterval) {
                    currentIndexRef.current = idx; // index だけ進めて鳴らさない
                    return;
                }

                const chord = timeline[idx];

                // playChordFromTab accepts string[] (e.g. ["x", "3", ...])
                playChordFromTab(chord.frets).catch((e) =>
                    console.error("Auto playback error:", e)
                );

                currentIndexRef.current = idx;
                lastTriggerTimeRef.current = now;
            }
        };

        rafId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafId);
    }, [audioElement, chordTimeline, enabled, timingOffsetSec]);
}
