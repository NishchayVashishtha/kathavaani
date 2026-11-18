import React from 'react';
import { Turn, StoryTimeline } from '../types';
import { Sparkles, PlayCircle } from 'lucide-react';

interface KaraokePlayerProps {
  timeline: StoryTimeline | null;
  turns: Turn[];
  currentTime: number;
  onSeekToTurn?: (turnIdx: number) => void;
}

export const KaraokePlayer: React.FC<KaraokePlayerProps> = ({
  timeline,
  turns,
  currentTime,
  onSeekToTurn,
}) => {
  return (
    <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-slate-100">
            कथा वाचक — कैराओके प्लेयर (Synchronized Story Reader)
          </h3>
        </div>
        <span className="text-[11px] text-slate-400">
          Click any sentence to jump audio playback
        </span>
      </div>

      <div className="space-y-3 max-h-80 overflow-y-auto pr-2">
        {turns.map((turn) => {
          const stem = timeline?.stems ? timeline.stems[turn.idx] : null;
          const isActive =
            stem &&
            currentTime >= stem.start_sec &&
            currentTime <= stem.end_sec;

          return (
            <div
              key={turn.idx}
              onClick={() => onSeekToTurn && onSeekToTurn(turn.idx)}
              className={`p-3 rounded-lg cursor-pointer transition-all ${
                isActive
                  ? 'bg-amber-500/15 border-l-4 border-amber-400 text-amber-100 shadow-md translate-x-1'
                  : 'bg-slate-950/40 hover:bg-slate-900/80 border-l-4 border-transparent text-slate-300'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                    #{turn.idx + 1}
                  </span>
                  <span className="text-xs font-bold font-devanagari text-slate-300">
                    {turn.speaker}
                  </span>
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800/80 text-slate-400 uppercase">
                    {turn.style?.emotion}
                  </span>
                </div>
                {stem && (
                  <span className="text-[10px] font-mono text-slate-500 flex items-center gap-1">
                    <PlayCircle className="w-3 h-3 text-slate-500" />
                    {stem.start_sec.toFixed(1)}s - {stem.end_sec.toFixed(1)}s
                  </span>
                )}
              </div>
              <p
                className={`font-devanagari leading-relaxed text-sm ${
                  isActive ? 'font-semibold text-amber-200' : 'text-slate-200'
                }`}
              >
                {turn.text}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
