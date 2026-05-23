'use client';

import React, { useState, useEffect, useRef } from 'react';

interface BeatsEditorProps {
  audioUrl: string;
  onBeatsChange: (beats: Array<[number, number]>) => void;
  initialBeats?: Array<[number, number]>;
}

export default function BeatsEditor({
  audioUrl,
  onBeatsChange,
  initialBeats = []
}: BeatsEditorProps) {
  const [beats, setBeats] = useState<Array<[number, number]>>(initialBeats);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    onBeatsChange(beats);
  }, [beats, onBeatsChange]);

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || duration === 0) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const clickTime = (x / rect.width) * duration;

    // Check if clicking near an existing beat (to toggle it)
    const clickThreshold = 0.1; // 100ms
    const existingIndex = beats.findIndex(([t]) => Math.abs(t - clickTime) < clickThreshold);

    if (existingIndex !== -1) {
      // Toggle beat type or remove if it's the only one
      const [time, type] = beats[existingIndex];
      const newBeats = [...beats];
      if (type === 1) {
        // Remove downbeat, convert to regular beat
        newBeats[existingIndex] = [time, 0];
      } else {
        // Remove beat entirely
        newBeats.splice(existingIndex, 1);
      }
      setBeats(newBeats.sort((a, b) => a[0] - b[0]));
    } else {
      // Add new beat (Shift+click for downbeat)
      const newType = e.shiftKey ? 1 : 0;
      setBeats([...beats, [clickTime, newType]] as Array<[number, number]>);
    }
  };

  const handleExportBeats = () => {
    const beatsText = beats.map(([t, type]) => `${t.toFixed(6)} ${type}`).join('\n');
    const blob = new Blob([beatsText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'annotations.beats';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportBeats = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const newBeats: Array<[number, number]> = [];

      text.split('\n').forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 1) {
          const time = parseFloat(parts[0]);
          const type = parts.length > 1 ? parseInt(parts[1]) : 0;
          if (!isNaN(time)) {
            newBeats.push([time, type]);
          }
        }
      });

      setBeats(newBeats.sort((a, b) => a[0] - b[0]));
    };
    reader.readAsText(file);
  };

  // Draw waveform and beats
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = '#f3f4f6';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Draw time markers
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px sans-serif';
    for (let t = 0; t <= duration; t += 10) {
      const x = (t / duration) * rect.width;
      ctx.fillText(`${t}s`, x, rect.height - 5);
    }

    // Draw beats
    beats.forEach(([time, type]) => {
      const x = (time / duration) * rect.width;
      const y = type === 1 ? 10 : 20; // Downbeats higher

      ctx.beginPath();
      ctx.arc(x, y, type === 1 ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = type === 1 ? '#ef4444' : '#3b82f6';
      ctx.fill();
    });

    // Draw playhead
    const playheadX = (currentTime / duration) * rect.width;
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, rect.height);
    ctx.stroke();

  }, [beats, currentTime, duration]);

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Beat Annotations</h3>
        <div className="flex gap-2">
          <label className="px-3 py-1 text-sm bg-gray-100 rounded cursor-pointer hover:bg-gray-200">
            Import
            <input
              type="file"
              accept=".beats,.txt"
              onChange={handleImportBeats}
              className="hidden"
            />
          </label>
          <button
            onClick={handleExportBeats}
            className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
          >
            Export .beats
          </button>
        </div>
      </div>

      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        className="w-full mb-4"
        controls
      />

      <div className="mb-2 text-sm text-gray-600">
        <span className="font-medium">Controls:</span> Click to add beat, Shift+click for downbeat, click on beat to toggle/remove
      </div>

      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="w-full h-24 border rounded cursor-crosshair"
      />

      <div className="mt-4 flex items-center justify-between text-sm text-gray-600">
        <div>
          <span className="font-medium">Beats:</span> {beats.filter(b => b[1] === 0).length} regular,
          <span className="font-medium ml-2">Downbeats:</span> {beats.filter(b => b[1] === 1).length}
        </div>
        <div>
          Time: {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
        </div>
      </div>
    </div>
  );
}
