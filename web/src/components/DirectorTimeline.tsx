import React, { useState, useEffect, useRef } from 'react';
import { Turn, Character, StoryTimeline } from '../types';
import { Play, Pause, RotateCw, ChevronDown, ChevronUp, Music, Sliders, Volume2, Sparkles } from 'lucide-react';

interface DirectorTimelineProps {
  timeline: StoryTimeline | null;
  turns: Turn[];
  characters: Character[];
  onRegenerateTurn: (turnIdx: number, newStyle?: any) => Promise<void>;
  onTurnChange: (updatedTurns: Turn[]) => void;
  onPlayheadUpdate?: (currentTime: number) => void;
}

export const DirectorTimeline: React.FC<DirectorTimelineProps> = ({
  timeline,
  turns,
  characters,
  onRegenerateTurn,
  onTurnChange,
  onPlayheadUpdate,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [expandedCotIdx, setExpandedCotIdx] = useState<number | null>(null);
  const [regeneratingIdx, setRegeneratingIdx] = useState<number | null>(null);
  const [duckingDb, setDuckingDb] = useState(-14);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (timeline?.total_duration) {
      setDuration(timeline.total_duration);
    }
  }, [timeline]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      const cur = audioRef.current.currentTime;
      setCurrentTime(cur);
      if (onPlayheadUpdate) onPlayheadUpdate(cur);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    const target = pos * duration;
    audioRef.current.currentTime = target;
    setCurrentTime(target);
  };

  const emotionColors: Record<string, string> = {
    neutral: 'bg-slate-700 text-slate-200 border-slate-600',
    happy: 'bg-emerald-900/60 text-emerald-300 border-emerald-700',
    excited: 'bg-amber-900/60 text-amber-300 border-amber-700',
    sad: 'bg-blue-900/60 text-blue-300 border-blue-700',
    angry: 'bg-red-900/60 text-red-300 border-red-700',
    fear: 'bg-purple-900/60 text-purple-300 border-purple-700',
    surprise: 'bg-pink-900/60 text-pink-300 border-pink-700',
    whisper: 'bg-indigo-900/60 text-indigo-300 border-indigo-700',
    tender: 'bg-teal-900/60 text-teal-300 border-teal-700',
    tense: 'bg-orange-900/60 text-orange-300 border-orange-700',
  };

  return (
    <div className="space-y-6">
      {/* Multitrack Visualizer & Transport */}
      <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={togglePlay}
              className="w-10 h-10 rounded-full bg-amber-500 hover:bg-amber-400 text-slate-950 flex items-center justify-center font-bold transition-transform active:scale-95"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
            </button>
            <div>
              <div className="text-sm font-semibold text-slate-200">
                {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
              </div>
              <div className="text-[10px] text-slate-400 font-mono">
                Loudness: {timeline?.lufs_level || -16.0} LUFS • Broadcast Standard
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
              <Music className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400">Sidechain Ducking:</span>
              <span className="font-mono text-cyan-300">{duckingDb} dB</span>
              <input
                type="range"
                min="-24"
                max="-6"
                value={duckingDb}
                onChange={(e) => setDuckingDb(Number(e.target.value))}
                className="w-16 h-1 accent-cyan-400"
              />
            </div>
          </div>
        </div>

        {/* Master Audio Element */}
        {timeline?.master_audio_url && (
          <audio
            ref={audioRef}
            src={timeline.master_audio_url}
            onTimeUpdate={handleTimeUpdate}
            onEnded={() => setIsPlaying(false)}
          />
        )}

        {/* Timeline Tracks (Master, Stems, Ambience) */}
        <div
          onClick={handleSeek}
          className="relative h-28 bg-slate-950 rounded-lg border border-slate-800 cursor-pointer overflow-hidden p-2 space-y-1.5"
        >
          {/* Playhead */}
          <div
            style={{ left: `${duration ? (currentTime / duration) * 100 : 0}%` }}
            className="absolute top-0 bottom-0 w-0.5 bg-amber-400 z-30 shadow-[0_0_8px_#f59e0b]"
          >
            <div className="w-2 h-2 -ml-[3px] bg-amber-400 rotate-45" />
          </div>

          {/* Track 1: Master Mix Waveform Simulator */}
          <div className="h-7 w-full bg-slate-900/60 rounded flex items-center px-2 relative">
            <span className="text-[9px] font-mono text-slate-500 absolute left-2 z-10 uppercase">
              Track 1: Master (-16 LUFS)
            </span>
            <div className="w-full flex items-center justify-between gap-[2px] opacity-40">
              {Array.from({ length: 90 }).map((_, i) => (
                <div
                  key={i}
                  style={{ height: `${15 + Math.sin(i * 0.4) * 12 + Math.cos(i * 0.8) * 8}px` }}
                  className="w-1 bg-amber-400/80 rounded-full"
                />
              ))}
            </div>
          </div>

          {/* Track 2: Dialogue & Character Stems */}
          <div className="h-7 w-full bg-slate-900/60 rounded relative overflow-hidden flex items-center">
            <span className="text-[9px] font-mono text-slate-500 absolute left-2 z-10 uppercase">
              Track 2: Speech Stems
            </span>
            {timeline?.stems?.map((stem, i) => {
              const leftPct = duration ? (stem.start_sec / duration) * 100 : 0;
              const widthPct = duration ? (stem.duration_sec / duration) * 100 : 0;
              return (
                <div
                  key={i}
                  style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  className="absolute h-5 top-1 rounded bg-amber-500/30 border border-amber-400/50 flex items-center px-1 text-[9px] text-amber-200 truncate font-devanagari"
                >
                  {stem.speaker}
                </div>
              );
            })}
          </div>

          {/* Track 3: Procedural Ambience / Music */}
          <div className="h-7 w-full bg-slate-900/60 rounded relative flex items-center px-2">
            <span className="text-[9px] font-mono text-slate-500 absolute left-2 z-10 uppercase">
              Track 3: Ambience ({timeline?.ambience_track || 'village_morning'})
            </span>
            <div className="w-full h-2 bg-cyan-950/40 rounded border border-cyan-800/40 opacity-70" />
          </div>
        </div>
      </div>

      {/* Script Turns & Emotion Table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-amber-400" />
            डायरेक्टर टाइमलाइन (Director Cue Sheet & Chain-of-Thought)
          </h3>
          <span className="text-xs text-slate-400">{turns.length} Turn Sequences</span>
        </div>

        <div className="space-y-2">
          {turns.map((turn) => {
            const isPlayingThisTurn =
              timeline?.stems &&
              timeline.stems[turn.idx] &&
              currentTime >= timeline.stems[turn.idx].start_sec &&
              currentTime <= timeline.stems[turn.idx].end_sec;

            const isExpanded = expandedCotIdx === turn.idx;
            const isRegen = regeneratingIdx === turn.idx;

            return (
              <div
                key={turn.idx}
                className={`p-3 rounded-xl border transition-all ${
                  isPlayingThisTurn
                    ? 'border-amber-400 bg-amber-950/20 shadow-[0_0_12px_rgba(245,158,11,0.15)]'
                    : 'border-slate-800/80 bg-slate-900/40 hover:bg-slate-900/70'
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                      #{turn.idx + 1}
                    </span>
                    <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700 text-slate-200 font-devanagari">
                      {turn.speaker}
                    </span>
                    <span
                      className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded border ${
                        emotionColors[turn.style?.emotion || 'neutral'] || emotionColors.neutral
                      }`}
                    >
                      {turn.style?.emotion || 'neutral'} ({(turn.style?.intensity ?? 0.5).toFixed(1)})
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Expand Chain of Thought button */}
                    <button
                      onClick={() => setExpandedCotIdx(isExpanded ? null : turn.idx)}
                      className="flex items-center gap-1 text-[11px] text-cyan-400 hover:text-cyan-300 bg-cyan-950/40 px-2 py-1 rounded border border-cyan-800/50"
                    >
                      <Sparkles className="w-3 h-3" />
                      <span>Why this emotion?</span>
                      {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    </button>

                    {/* Single-turn regenerate button */}
                    <button
                      disabled={isRegen}
                      onClick={async () => {
                        setRegeneratingIdx(turn.idx);
                        try {
                          await onRegenerateTurn(turn.idx);
                        } finally {
                          setRegeneratingIdx(null);
                        }
                      }}
                      className="flex items-center gap-1 text-[11px] bg-slate-800 hover:bg-slate-700 text-slate-200 px-2.5 py-1 rounded border border-slate-700 disabled:opacity-50"
                    >
                      <RotateCw className={`w-3 h-3 ${isRegen ? 'animate-spin text-amber-400' : ''}`} />
                      <span>{isRegen ? 'Synthesizing...' : 'Regen'}</span>
                    </button>
                  </div>
                </div>

                {/* Devanagari Dialogue / Narration Text */}
                <p className="mt-2 text-sm text-slate-200 font-devanagari leading-relaxed">
                  {turn.text}
                </p>

                {/* Expandable Chain of Thought Drawer */}
                {isExpanded && (
                  <div className="mt-3 p-3 rounded-lg bg-slate-950 border border-cyan-900/50 space-y-2 text-xs">
                    <div className="flex items-center gap-1 text-cyan-400 font-semibold text-[11px]">
                      <Sparkles className="w-3.5 h-3.5" />
                      Agent 2 (Narrative Director) Prosody Reasoning:
                    </div>
                    <p className="text-slate-300 font-mono text-[11px] bg-slate-900 p-2 rounded">
                      {turn.style?.reasoning || 'Default emotional pacing based on narrative context and character intent.'}
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px] text-slate-400 pt-1 font-mono">
                      <div>Pitch: <span className="text-slate-200">{turn.style?.pitch || 'medium'}</span></div>
                      <div>Speed: <span className="text-slate-200">{turn.style?.speed || 'medium'}</span></div>
                      <div>Pause Before: <span className="text-slate-200">{turn.style?.pause_before_ms || 300}ms</span></div>
                      <div>Pause After: <span className="text-slate-200">{turn.style?.pause_after_ms || 250}ms</span></div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
