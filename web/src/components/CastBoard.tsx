import React from 'react';
import { Character, VoiceProfile } from '../types';
import { Volume2, UserCheck, Shield, Sparkles } from 'lucide-react';

interface CastBoardProps {
  characters: Character[];
  voiceBank: VoiceProfile[];
  onVoiceChange?: (charId: string, voiceId: string) => void;
  onPreviewVoice?: (voiceId: string) => void;
}

export const CastBoard: React.FC<CastBoardProps> = ({
  characters,
  voiceBank,
  onVoiceChange,
  onPreviewVoice,
}) => {
  // Color palette for characters
  const charColors = [
    'border-amber-500/60 bg-amber-950/20 text-amber-300',
    'border-emerald-500/60 bg-emerald-950/20 text-emerald-300',
    'border-cyan-500/60 bg-cyan-950/20 text-cyan-300',
    'border-rose-500/60 bg-rose-950/20 text-rose-300',
    'border-purple-500/60 bg-purple-950/20 text-purple-300',
    'border-blue-500/60 bg-blue-950/20 text-blue-300',
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <UserCheck className="w-5 h-5 text-amber-400" />
            कलाकार मंडल (Cast Board & Timbre Lock)
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            ECAPA-TDNN Voice Allocation with Inter-Character Acoustic Distance Maximization
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg text-slate-300">
          <Shield className="w-4 h-4 text-emerald-400" />
          <span>Timbre Lock: <strong className="text-emerald-400">ACTIVE</strong></span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {characters.map((char, idx) => {
          const assignedVoice = voiceBank.find(v => v.id === char.assigned_voice_id);
          const colorClass = charColors[idx % charColors.length];

          return (
            <div
              key={char.id}
              className={`p-4 rounded-xl border ${colorClass} backdrop-blur-sm transition-all hover:scale-[1.01]`}
            >
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[10px] uppercase font-mono tracking-wider opacity-75">
                    {char.id} • {char.gender} ({char.age_group})
                  </span>
                  <h3 className="text-lg font-bold font-devanagari mt-0.5">{char.name}</h3>
                </div>
                <div className="w-8 h-8 rounded-full bg-slate-800/80 flex items-center justify-center font-bold text-xs">
                  {char.name.slice(0, 1)}
                </div>
              </div>

              <p className="text-xs text-slate-300/80 italic mt-2 line-clamp-2">
                "{char.personality || 'No description'}"
              </p>

              {char.aliases && char.aliases.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {char.aliases.map(alias => (
                    <span key={alias} className="text-[10px] bg-slate-800/80 px-1.5 py-0.5 rounded text-slate-300 font-devanagari">
                      {alias}
                    </span>
                  ))}
                </div>
              )}

              <div className="mt-4 pt-3 border-t border-slate-800/60 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 text-[11px]">Assigned Voice:</span>
                  {assignedVoice && (
                    <button
                      onClick={() => onPreviewVoice && onPreviewVoice(assignedVoice.id)}
                      className="flex items-center gap-1 text-[11px] text-amber-400 hover:text-amber-300"
                    >
                      <Volume2 className="w-3.5 h-3.5" />
                      Preview
                    </button>
                  )}
                </div>

                <select
                  value={char.assigned_voice_id || ''}
                  onChange={(e) => onVoiceChange && onVoiceChange(char.id, e.target.value)}
                  className="w-full text-xs bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-amber-500"
                >
                  {voiceBank.map((voice) => (
                    <option key={voice.id} value={voice.id}>
                      {voice.name} ({voice.gender}, {voice.age_group})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          );
        })}
      </div>

      {/* 2-D ECAPA-TDNN Embedding Scatter Plot */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-slate-200">
              Voice Bank 2-D Acoustic Space (ECAPA-TDNN PCA Projection)
            </h3>
          </div>
          <span className="text-xs text-slate-400">Inter-Character Timbre Distance</span>
        </div>

        <div className="relative h-44 w-full bg-slate-950/80 rounded-lg border border-slate-800 overflow-hidden flex items-center justify-center p-4">
          <div className="absolute inset-0 grid grid-cols-6 grid-rows-4 divide-x divide-y divide-slate-800/30 pointer-events-none" />
          
          {voiceBank.map((voice) => {
            // Normalize coords from PCA [-3, 3] to percentage [10%, 90%]
            const left = Math.min(92, Math.max(8, 50 + voice.pca_coords[0] * 14));
            const top = Math.min(88, Math.max(12, 50 - voice.pca_coords[1] * 14));
            const isAssigned = characters.some(c => c.assigned_voice_id === voice.id);

            return (
              <div
                key={voice.id}
                style={{ left: `${left}%`, top: `${top}%` }}
                className={`absolute -translate-x-1/2 -translate-y-1/2 group cursor-pointer`}
                onClick={() => onPreviewVoice && onPreviewVoice(voice.id)}
              >
                <div className={`w-3.5 h-3.5 rounded-full border-2 transition-transform group-hover:scale-150 ${
                  isAssigned
                    ? 'bg-amber-400 border-white ring-4 ring-amber-500/30'
                    : 'bg-slate-600 border-slate-400'
                }`} />
                <div className="absolute left-1/2 bottom-full -translate-x-1/2 mb-1 hidden group-hover:block z-20 whitespace-nowrap bg-slate-800 text-white text-[10px] px-2 py-0.5 rounded shadow-lg border border-slate-700">
                  {voice.name} ({voice.gender})
                </div>
              </div>
            );
          })}

          <div className="absolute bottom-2 right-3 text-[10px] text-slate-500 font-mono">
            Principal Component 1 (F0 & Formant Dispersal) →
          </div>
        </div>
      </div>
    </div>
  );
};
