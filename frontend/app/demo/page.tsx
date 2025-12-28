'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { extractVideoId } from '../../lib/youtube';
import { demoSongs } from './data/demoSongs';
import type { TimedChord } from '../../lib/chordTimeline';

// Declare YT global for TypeScript
declare global {
  interface Window {
    YT: any;
    onYouTubeIframeAPIReady: () => void;
  }
}

export default function DemoPage() {
  // Logic State
  const [inputUrl, setInputUrl] = useState('');
  const [activeSong, setActiveSong] = useState<typeof demoSongs[number] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlayerReady, setIsPlayerReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  // Synchro State
  const [currentTime, setCurrentTime] = useState(0);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [offsetSec, setOffsetSec] = useState(0);
  const [autoScroll, setAutoScroll] = useState(true);

  // Refs
  const playerRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);
  const requestRef = useRef<number>(0);
  const suppressScrollRef = useRef<number>(0);

  // Load Youtube API
  useEffect(() => {
    if (!window.YT) {
      const tag = document.createElement('script');
      tag.src = "https://www.youtube.com/iframe_api";
      const firstScriptTag = document.getElementsByTagName('script')[0];
      firstScriptTag.parentNode?.insertBefore(tag, firstScriptTag);

      window.onYouTubeIframeAPIReady = () => {
        // API Ready
      };
    }
  }, []);

  // Initialize Player when song is selected
  useEffect(() => {
    if (activeSong && window.YT && !playerRef.current) {
      // Small timeout to ensure container is rendered
      setTimeout(() => {
        playerRef.current = new window.YT.Player('demo-player', {
          height: '360',
          width: '640',
          videoId: activeSong.videoId,
          playerVars: {
            'playsinline': 1,
            'controls': 1, // Use native controls for Demo simplicity or 0 for custom
          },
          events: {
            'onReady': (event: any) => {
              setIsPlayerReady(true);
            },
            'onStateChange': (event: any) => {
              if (event.data === window.YT.PlayerState.PLAYING) {
                setIsPlaying(true);
                suppressScrollRef.current = Date.now() + 500; // Small suppression on start
              } else {
                setIsPlaying(false);
                // If user paused, suppress for a bit
                if (event.data === window.YT.PlayerState.PAUSED) {
                  suppressScrollRef.current = Date.now() + 2000;
                }
              }
            }
          }
        });
      }, 100);
      return () => {
        if (playerRef.current) {
          try { playerRef.current.destroy(); } catch (e) { }
          playerRef.current = null;
        }
      }
    }
  }, [activeSong]);

  // Sync Loop
  const updateSync = useCallback(() => {
    if (playerRef.current && isPlayerReady && activeSong) {
      // Safe get time
      let t = 0;
      try {
        t = playerRef.current.getCurrentTime();
      } catch (e) {
        // Player might be in a weird state
      }

      const effectiveTime = t + offsetSec;
      setCurrentTime(effectiveTime);

      // Find active index
      // Optimization: Could start search from current index, but array is small (20 items)
      const idx = activeSong.timeline.findIndex(c => effectiveTime >= c.startSec && effectiveTime < c.endSec);

      if (idx !== -1 && idx !== activeIndex) {
        setActiveIndex(idx);
      } else if (idx === -1 && activeIndex !== -1) {
        // Keep last active if nearly done or reset?
        // If time < first, -1. If time > last, maybe keep last?
        // For now, -1 is fine.
        if (effectiveTime < activeSong.timeline[0].startSec) {
          setActiveIndex(-1);
        }
      }

      // Explicitly handle "finished" or post-song state if needed
    }
    requestRef.current = requestAnimationFrame(updateSync);
  }, [isPlayerReady, activeSong, offsetSec, activeIndex]);

  useEffect(() => {
    requestRef.current = requestAnimationFrame(updateSync);
    return () => cancelAnimationFrame(requestRef.current);
  }, [updateSync]);


  // Auto Scroll Effect
  useEffect(() => {
    if (autoScroll && activeIndex >= 0 && rowRefs.current[activeIndex] && Date.now() > suppressScrollRef.current) {
      rowRefs.current[activeIndex]?.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, [activeIndex, autoScroll]);

  // Handlers
  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setActiveSong(null);
    setIsPlayerReady(false);
    playerRef.current = null;

    const videoId = extractVideoId(inputUrl);
    if (!videoId) {
      setError("Invalid YouTube URL");
      return;
    }

    const match = demoSongs.find(s => s.videoId === videoId);
    if (match) {
      setOffsetSec(match.defaultOffsetSec);
      setActiveSong(match);
    } else {
      setError("This demo only supports specific songs.");
    }
  };

  const copySupportedUrl = () => {
    if (demoSongs[0]) {
      navigator.clipboard.writeText(demoSongs[0].supportedUrl).then(() => {
        alert("Copied to clipboard!");
      });
    }
  };

  const hasTimeline = activeSong && activeSong.timeline.length > 0;

  return (
    <main translate="no" className="notranslate min-h-screen bg-gray-50 pb-32">

      {/* Header */}
      <div className="bg-white shadow border-b border-gray-200 py-4 px-4 sticky top-0 z-10">
        <div className="max-w-2xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold text-gray-800">Demo Player</h1>
          <a href="/" className="text-sm text-blue-600 hover:text-blue-800">Back</a>
        </div>
      </div>

      <div className="max-w-2xl mx-auto p-4 space-y-12">
        {/* [First View] */}
        <div className="text-center space-y-4 py-8">
          <h2 className="text-2xl font-bold text-gray-900 leading-snug">
            再生するだけで、<br />コード進行が“譜面として追える”体験
          </h2>
          <p className="text-gray-600 leading-relaxed">
            これは BandScore が目指している体験を伝えるためのデモです。<br />
            現在は、1曲・固定データのみを使用しています。
          </p>
        </div>

        {/* URL Input */}
        {!activeSong && (
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <form onSubmit={handleUrlSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">YouTube URL</label>
                <input
                  type="text"
                  className="w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 p-2 border"
                  placeholder="https://youtube.com/watch?v=..."
                  value={inputUrl}
                  onChange={e => setInputUrl(e.target.value)}
                />
                <p className="mt-2 text-sm text-gray-600">
                  ※ このデモで入力できるURLは以下の1つのみです<br />
                  <span className="font-mono select-all block mt-1 break-all bg-gray-50 p-2 rounded border border-gray-200">
                    https://www.youtube.com/watch?v=JegJ6cSsUgg
                  </span>
                </p>
              </div>
              <button
                type="submit"
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
              >
                Load Demo
              </button>
            </form>

            {error && (
              <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-md text-sm border border-red-100">
                <p className="font-bold mb-2">{error}</p>
                {error.includes("specific songs") && (
                  <button
                    onClick={copySupportedUrl}
                    className="text-blue-600 underline hover:text-blue-800"
                  >
                    Copy supported URL for testing
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        {/* Player & Content */}
        {activeSong && (
          <div className="space-y-6">
            <div className="aspect-video bg-black rounded-xl overflow-hidden shadow-lg">
              <div id="demo-player" className="w-full h-full"></div>
            </div>

            {/* Controls / Info */}
            <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 flex justify-between items-center text-sm">
              <div>
                <span className="text-gray-500">Offset: </span>
                <span className="font-mono font-bold">{offsetSec > 0 ? '+' : ''}{offsetSec.toFixed(1)}s</span>
              </div>
              <input
                type="range"
                min="-5" max="5" step="0.1"
                value={offsetSec}
                onChange={(e) => setOffsetSec(parseFloat(e.target.value))}
                className="w-32 sm:w-48"
              />
            </div>

            {/* Timeline List */}
            <div className="space-y-2 pb-20">
              {activeSong.timeline.map((item, idx) => {
                const isActive = idx === activeIndex;
                return (
                  <div
                    key={idx}
                    ref={el => { rowRefs.current[idx] = el; }}
                    className={`p-3 rounded-lg flex items-center justify-between transition-all duration-300 ${isActive
                      ? 'bg-blue-600 text-white shadow-md scale-105 my-2'
                      : 'bg-white text-gray-600 border border-gray-100 hover:bg-gray-50'
                      }`}
                  >
                    <div className="flex items-center gap-4">
                      <span className={`text-xs font-mono w-12 ${isActive ? 'text-blue-200' : 'text-gray-400'}`}>
                        {Math.floor(item.startSec / 60)}:{(item.startSec % 60).toFixed(1).padStart(4, '0')}
                      </span>
                      <span
                        translate="no"
                        className={`notranslate text-xl font-bold ${isActive ? 'scale-110 origin-left' : ''}`}
                      >
                        {item.chordName}
                      </span>
                    </div>
                    <div translate="no" className="notranslate flex gap-1 font-mono text-xs">
                      {item.frets.map((f, i) => (
                        <span key={i} className={`w-4 text-center ${isActive ? 'text-blue-100' : 'text-gray-400'}`}>
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
              <p className="text-center text-sm text-gray-500 py-4">
                ※ デモではここまでの表示となります
              </p>
            </div>
          </div>
        )}
        {/* [Demo Disclaimer] */}
        <div className="bg-gray-100 p-6 rounded-xl space-y-3">
          <h3 className="font-bold text-gray-800">このデモについて</h3>
          <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
            <li>実際の音源解析やアップロードは行いません</li>
            <li>コード進行は、あらかじめ用意したデータを使用しています</li>
            <li>デモは体験用のため、保存・編集機能はありません</li>
          </ul>
        </div>

        {/* [Value & Future] */}
        <div className="text-center space-y-4 py-8">
          <p className="text-lg font-medium text-gray-800">
            BandScore では、『耳コピに時間がかかる』『譜面を探すのが大変』<br className="hidden sm:inline" />
            そんな状態から、すぐに練習を始められる体験を目指しています。
          </p>
          <p className="text-gray-600 text-sm">
            将来的には、自分の曲・好きな曲を読み込んで、<br className="hidden sm:inline" />
            コード進行を確認しながら再生・練習するといった使い方を想定しています。
          </p>
        </div>

        {/* [Official Version Objectives] */}
        <div className="bg-gradient-to-br from-blue-50 to-white border-2 border-blue-100 rounded-2xl p-6 sm:p-8 shadow-sm space-y-6">
          <div className="space-y-2 text-center sm:text-left">
            <h3 className="text-xl font-bold text-gray-900 flex items-center justify-center sm:justify-start gap-3">
              <span className="text-2xl">🚀</span> 正式版で目指していること
            </h3>
            <p className="text-gray-700 font-medium leading-relaxed">
              BandScore は、聴いた瞬間に“練習に入れる”体験を作るために開発中です。
            </p>
          </div>

          <ul className="grid sm:grid-cols-2 gap-4">
            {[
              "音源を解析してコードを自動生成",
              "曲に合わせてコード音を再生（練習用）",
              "生成結果の PDF / MIDI 出力",
              "生成コードの編集（修正・整形）"
            ].map((item, i) => (
              <li key={i} className="flex items-center gap-3 bg-white p-4 rounded-xl border border-blue-100 shadow-sm text-gray-700 font-medium text-sm">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold">✔</span>
                {item}
              </li>
            ))}
          </ul>

          <p className="text-xs text-gray-500 bg-white/50 p-4 rounded-lg border border-gray-100 leading-relaxed">
            ※ いずれも開発中の構想です。提供時期・仕様は変更になる可能性があります。この /demo は解析や保存を行わない“体験用”デモです。
          </p>
        </div>

        {/* [CTA] */}
        <div className="bg-blue-50 p-8 rounded-2xl text-center space-y-6">
          <a
            href="/waitlist"
            className="inline-block bg-blue-600 text-white font-bold py-3 px-8 rounded-full shadow-lg hover:bg-blue-700 transition-transform active:scale-95"
          >
            正式版へのご案内
          </a>
          <p className="text-xs text-gray-500">
            ※ 現時点では利用登録はできません。公開準備が整い次第、ご案内します。
          </p>
        </div>
      </div>

      {/* Fixed Bottom Controls */}
      {activeSong && (
        <div className="fixed bottom-0 left-0 right-0 bg-white/90 backdrop-blur border-t border-gray-200 p-4 pb-6 sm:p-4 z-50">
          <div className="max-w-md mx-auto flex items-center justify-between gap-4">
            {/* Stop */}
            <button
              onClick={() => {
                if (playerRef.current && typeof playerRef.current.pauseVideo === 'function') {
                  playerRef.current.pauseVideo();
                  playerRef.current.seekTo(0, true);
                }
                suppressScrollRef.current = Date.now() + 1000;
              }}
              className="p-3 rounded-full bg-red-100 text-red-600 hover:bg-red-200 transition-colors"
            >
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M6 6h12v12H6z" /></svg>
            </button>

            {/* Play/Pause */}
            <button
              onClick={() => {
                if (playerRef.current && typeof playerRef.current.getPlayerState === 'function') {
                  const state = playerRef.current.getPlayerState();
                  if (state === 1) { // Playing
                    playerRef.current.pauseVideo();
                  } else {
                    playerRef.current.playVideo();
                  }
                  suppressScrollRef.current = Date.now() + 500;
                }
              }}
              className="flex-1 bg-blue-600 text-white rounded-xl py-3 font-bold shadow-lg hover:bg-blue-700 active:scale-95 transition-all text-center"
            >
              {isPlaying ? 'PAUSE' : 'PLAY'}
            </button>

            {/* Auto Scroll */}
            <button
              onClick={() => {
                setAutoScroll(!autoScroll);
                suppressScrollRef.current = Date.now() + 500;
              }}
              className={`p-3 rounded-full transition-colors ${autoScroll ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-400'}`}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" /></svg>
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
