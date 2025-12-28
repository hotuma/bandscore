let audioCtx: AudioContext | null = null;
let masterGain: GainNode | null = null;

const BASE_STRING_FREQS = [
    82.41,  // 6弦 E2
    110.0,  // 5弦 A2
    146.83, // 4弦 D3
    196.0,  // 3弦 G3
    246.94, // 2弦 B3
    329.63, // 1弦 E4
];

interface GuitarStringNote {
    stringIndex: number;
    fret: number | null;
}

function getAudioContext(): AudioContext | null {
    if (typeof window === 'undefined') return null;
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();

        // Initialize Master Gain
        masterGain = audioCtx.createGain();
        masterGain.gain.value = 0.8; // Default volume
        masterGain.connect(audioCtx.destination);
    }
    return audioCtx;
}

export function setChordVolume(volume: number) {
    const ctx = getAudioContext();
    if (!ctx || !masterGain) return;

    const v = Math.max(0, Math.min(1, volume));
    masterGain.gain.setTargetAtTime(v, ctx.currentTime, 0.01);
}

function fretToFrequency(stringIndex: number, fret: number): number {
    const baseFreq = BASE_STRING_FREQS[stringIndex];
    return baseFreq * Math.pow(2, fret / 12);
}

/**
 * TAB配列（["x","3","2","0","1","0"]など）からコード音を鳴らす
 * 既存のインターフェースを維持しつつ、内部で playGuitarChord を呼ぶラッパー
 */
export async function playChordFromTab(frets: string[]): Promise<void> {
    if (!frets || frets.length !== 6) {
        console.warn("playChordFromTab: invalid frets", frets);
        return;
    }

    // string[] -> GuitarStringNote[] 変換
    const voicing: GuitarStringNote[] = frets.map((fretStr, index) => {
        if (fretStr === "x" || fretStr === "X") {
            return { stringIndex: index, fret: null };
        }
        let n = parseInt(fretStr, 10);
        if (Number.isNaN(n) || n < 0) {
            n = 0;
        }
        return { stringIndex: index, fret: n };
    });

    await playGuitarChord(voicing);
}

/**
 * よりリアルなギター音を合成して鳴らす
 */
export async function playGuitarChord(
    voicing: GuitarStringNote[],
    durationSec: number = 1.5
): Promise<void> {
    const ctx = getAudioContext();
    if (!ctx || !masterGain) return;

    if (ctx.state === "suspended") {
        try {
            await ctx.resume();
        } catch (e) {
            console.warn("Failed to resume AudioContext", e);
            return;
        }
    }

    const now = ctx.currentTime;

    voicing.forEach((note, idx) => {
        if (note.fret == null) return; // ミュートは鳴らさない

        const freq = fretToFrequency(note.stringIndex, note.fret);

        // ストローク感：弦ごとに少しだけ時間をずらす（7〜17ms）
        const strumDelay = 0.007 * idx + (Math.random() * 0.01);
        const startTime = now + strumDelay;

        // ルートとなるオシレーター（triangle波）
        const osc1 = ctx.createOscillator();
        osc1.type = "triangle";
        osc1.frequency.value = freq;

        // 1オクターブ上の倍音用オシレーター
        const osc2 = ctx.createOscillator();
        osc2.type = "triangle";
        osc2.frequency.value = freq * 2;
        osc2.detune.value = 3; // ちょい太さ出し

        // ボディ感を出すフィルター（高域少しカット）
        const filter = ctx.createBiquadFilter();
        filter.type = "lowpass";
        filter.frequency.value = 2300;
        filter.Q.value = 0.8;

        // 音量エンベロープ
        const gain = ctx.createGain();

        // 「ピッキングっぽい」エンベロープ
        const attack = 0.004;          // 0.005 -> 0.004
        const decay = durationSec * 0.25; // 0.3 -> 0.25
        const sustainLevel = 0.25;        // 0.3 -> 0.25
        const release = durationSec * 0.9;// 1.0 -> 0.9

        const g = gain.gain;
        g.cancelScheduledValues(startTime);
        g.setValueAtTime(0.0001, startTime);
        g.exponentialRampToValueAtTime(0.9, startTime + attack);           // ピックの瞬間
        g.exponentialRampToValueAtTime(sustainLevel, startTime + decay);   // すぐに少し落ちる
        g.exponentialRampToValueAtTime(0.0001, startTime + release);       // その後ゆっくり減衰

        // ピックノイズ（ごく短いホワイトノイズ）
        const noiseBuffer = ctx.createBuffer(1, ctx.sampleRate * 0.03, ctx.sampleRate); // 30ms
        const data = noiseBuffer.getChannelData(0);
        for (let i = 0; i < data.length; i++) {
            data[i] = (Math.random() * 2 - 1) * 0.6;
        }
        const noiseSource = ctx.createBufferSource();
        noiseSource.buffer = noiseBuffer;

        const noiseFilter = ctx.createBiquadFilter();
        noiseFilter.type = "bandpass";
        noiseFilter.frequency.value = freq * 1.6;
        noiseFilter.Q.value = 3;

        const noiseGain = ctx.createGain();
        noiseGain.gain.setValueAtTime(0.30, startTime);
        noiseGain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.04);

        // 接続
        osc1.connect(filter);
        osc2.connect(filter);
        filter.connect(gain);
        noiseSource.connect(noiseFilter);
        noiseFilter.connect(noiseGain);
        noiseGain.connect(gain);

        // Connect to Master Gain instead of destination
        if (masterGain) {
            gain.connect(masterGain);
        }

        // 再生・停止
        osc1.start(startTime);
        osc2.start(startTime);
        noiseSource.start(startTime);

        const stopTime = startTime + durationSec + 0.3;
        osc1.stop(stopTime);
        osc2.stop(stopTime);
        noiseSource.stop(startTime + 0.05); // ノイズは超短く
    });
}
