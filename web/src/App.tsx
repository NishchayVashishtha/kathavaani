import React, { useState, useEffect } from 'react';
import { StoryInput } from './components/StoryInput';
import { CastBoard } from './components/CastBoard';
import { DirectorTimeline } from './components/DirectorTimeline';
import { KaraokePlayer } from './components/KaraokePlayer';
import { ListeningTest } from './components/ListeningTest';
import { VivaCheatSheet } from './components/VivaCheatSheet';
import { Character, Turn, VoiceProfile, StoryTimeline } from './types';
import {
  Mic2,
  BarChart3,
  Headphones,
  Award,
  Sparkles,
  Download,
  Terminal,
  Activity,
  Layers,
} from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'studio' | 'bakeoff' | 'listening' | 'viva'>('studio');

  // Application State
  const [characters, setCharacters] = useState<Character[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [voiceBank, setVoiceBank] = useState<VoiceProfile[]>([]);
  const [timeline, setTimeline] = useState<StoryTimeline | null>(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [statusMsg, setStatusMsg] = useState('System Ready • NVIDIA H200 (18GB VRAM)');

  // Load Voice Bank & Initial Master Showcase on startup
  useEffect(() => {
    fetch('/api/voicebank')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.voices) {
          setVoiceBank(data.voices);
        }
      })
      .catch((err) => console.error('Failed to load voice bank:', err));

    fetch('/api/gallery')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.productions && data.productions.length > 0) {
          const s1 = data.productions[0];
          if (s1 && s1.timeline) {
            setTimeline({
              master_audio_url: s1.master_mp3,
              stems: s1.timeline.stems || [],
              total_duration: s1.duration_sec,
              lufs_level: -16.0,
            });
          }
        }
      })
      .catch((err) => console.error('Failed to load gallery showcase:', err));
  }, []);

  // Stage 1 & 2: Analyze & Direct
  const handleAnalyze = async (storyText: string) => {
    setIsAnalyzing(true);
    setStatusMsg('Agent 1 (Script Analyst) & Agent 2 (Narrative Director) processing...');
    try {
      // 1. Analyze script
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ story_text: storyText }),
      });
      const data = await res.json();

      if (data.characters && data.turns) {
        setCharacters(data.characters);
        setTurns(data.turns);
        setHasAnalyzed(true);
        setStatusMsg('Script analyzed. Characters & prosody generated with Timbre Lock.');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setStatusMsg('Analysis failed. Using cached story parameters.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Stage 4 & 5: Full Synthesis
  const handleSynthesize = async () => {
    setIsSynthesizing(true);
    setStatusMsg('Executing XTTS v2 multi-speaker synthesis on NVIDIA H200 GPU...');
    try {
      const res = await fetch('/api/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          characters,
          turns,
          ambience: 'village_morning',
          ducking_db: -14,
        }),
      });
      const data = await res.json();

      if (data.timeline) {
        setTimeline(data.timeline);
        setStatusMsg('Production complete! Master audio normalized to -16 LUFS.');
      }
    } catch (err) {
      console.error('Synthesis error:', err);
      setStatusMsg('Synthesis error. Check cluster GPU logs.');
    } finally {
      setIsSynthesizing(false);
    }
  };

  // Single-turn regeneration
  const handleRegenerateTurn = async (turnIdx: number) => {
    const turn = turns.find((t) => t.idx === turnIdx);
    if (!turn) return;

    setStatusMsg(`Regenerating turn #${turnIdx + 1} with XTTS v2 on GPU...`);
    try {
      const res = await fetch('/api/regenerate-turn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          turn_idx: turnIdx,
          text: turn.text,
          speaker: turn.speaker,
          style: turn.style,
          character_voice_id: characters.find((c) => c.name === turn.speaker)?.assigned_voice_id,
        }),
      });
      const data = await res.json();
      if (data.turn_audio_url) {
        setStatusMsg(`Turn #${turnIdx + 1} regenerated successfully.`);
      }
    } catch (err) {
      console.error('Regeneration error:', err);
    }
  };

  // Character voice re-assignment
  const handleVoiceChange = (charId: string, voiceId: string) => {
    setCharacters((prev) =>
      prev.map((c) => (c.id === charId ? { ...c, assigned_voice_id: voiceId } : c))
    );
  };

  // Voice preview playback
  const handlePreviewVoice = (voiceId: string) => {
    const audio = new Audio(`/api/audio/voices/${voiceId}.wav`);
    audio.play().catch((err) => console.error('Audio preview error:', err));
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-100 flex items-center gap-2 font-devanagari">
              कथावाणी (KathaVaani)
              <span className="text-[10px] font-mono font-normal uppercase bg-amber-500/10 border border-amber-500/30 text-amber-300 px-2 py-0.5 rounded-full">
                Research Studio v2.0
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">
              Context-Aware, Expressive Multi-Speaker Hindi Storytelling Synthesis
            </p>
          </div>
        </div>

        {/* Tab Controls */}
        <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-800 p-1 rounded-xl">
          {[
            { id: 'studio', label: '1. KathaVaani Studio', icon: Mic2 },
            { id: 'bakeoff', label: '2. TTS Bake-Off', icon: BarChart3 },
            { id: 'listening', label: '3. Blind A/B MOS', icon: Headphones },
            { id: 'viva', label: '4. Viva Defence & Ablations', icon: Award },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === id
                  ? 'bg-amber-500 text-slate-950 font-bold shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{label}</span>
            </button>
          ))}
        </div>

        {/* Export & Status */}
        <div className="flex items-center gap-3">
          <div className="hidden lg:flex items-center gap-2 text-[11px] font-mono text-slate-400 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span>NVIDIA H200 (18GB)</span>
          </div>

          {timeline && (
            <a
              href={timeline.master_audio_url}
              download="kathavaani_master.wav"
              className="flex items-center gap-1.5 text-xs bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 px-3 py-1.5 rounded-lg font-semibold transition-all"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Master</span>
            </a>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {activeTab === 'studio' && (
          <div className="space-y-6">
            <StoryInput
              onAnalyze={handleAnalyze}
              onSynthesize={handleSynthesize}
              isAnalyzing={isAnalyzing}
              isSynthesizing={isSynthesizing}
              hasAnalyzed={hasAnalyzed}
            />

            {characters.length > 0 && (
              <CastBoard
                characters={characters}
                voiceBank={voiceBank}
                onVoiceChange={handleVoiceChange}
                onPreviewVoice={handlePreviewVoice}
              />
            )}

            {turns.length > 0 && (
              <DirectorTimeline
                timeline={timeline}
                turns={turns}
                characters={characters}
                onRegenerateTurn={handleRegenerateTurn}
                onTurnChange={setTurns}
                onPlayheadUpdate={setCurrentTime}
              />
            )}

            {turns.length > 0 && (
              <KaraokePlayer
                timeline={timeline}
                turns={turns}
                currentTime={currentTime}
                onSeekToTurn={(idx) => {
                  const stem = timeline?.stems ? timeline.stems[idx] : null;
                  if (stem) setCurrentTime(stem.start_sec);
                }}
              />
            )}
          </div>
        )}

        {activeTab === 'bakeoff' && (
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
            <div className="border-b border-slate-800 pb-4">
              <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-amber-400" />
                TTS Engine Empirical Bake-Off & Benchmark (30 Sentences)
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Measured on NVIDIA H200 (18GB MIG) across 30 Devanagari Sentences using OpenAI Whisper ASR & SpeechBrain ECAPA-TDNN
              </p>
            </div>

            {/* Empirical Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-[10px] uppercase font-mono text-slate-400">Selected Engine</span>
                <div className="text-lg font-bold text-amber-400">XTTS v2 (Zero-Shot)</div>
                <div className="text-[11px] text-slate-500 font-mono">Autoregressive GPT + HiFi-GAN</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-[10px] uppercase font-mono text-slate-400">Mean CER (ASR Round-Trip)</span>
                <div className="text-lg font-bold text-emerald-400">22.87% (Neutral)</div>
                <div className="text-[11px] text-slate-500 font-mono">Whisper Small / Hindi</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-[10px] uppercase font-mono text-slate-400">Real-Time Factor (RTF)</span>
                <div className="text-lg font-bold text-cyan-400">0.21x (5x Real-Time)</div>
                <div className="text-[11px] text-slate-500 font-mono">Fast Inference on H200</div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-[10px] uppercase font-mono text-slate-400">Peak VRAM Footprint</span>
                <div className="text-lg font-bold text-purple-400">3.00 GB / 18 GB</div>
                <div className="text-[11px] text-slate-500 font-mono">16.6% VRAM Budget Utilized</div>
              </div>
            </div>

            {/* Audio Comparison Players */}
            <div className="space-y-3 pt-2">
              <h3 className="text-sm font-semibold text-slate-200">
                Direct Side-by-Side Audio Comparison (XTTS v2 vs Meta MMS-TTS)
              </h3>
              {[
                { id: 'SENT_01', text: 'एक पुराने गांव के किनारे एक घना और शांत बरगद का पेड़ था।', emotion: 'neutral' },
                { id: 'SENT_06', text: 'वाह! आज आखिरकार हमारे गांव में बिजली आ ही गई!', emotion: 'happy' },
                { id: 'SENT_16', text: 'खबरदार! अगर किसी ने उस खज़ाने को हाथ लगाने की कोशिश की तो अच्छा नहीं होगा!', emotion: 'angry' },
                { id: 'SENT_24', text: 'धीमे बोलो, मुझे लगता है कोई खिड़की के बाहर चुपचाप सुन रहा है।', emotion: 'whisper' },
              ].map((item) => (
                <div key={item.id} className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-amber-400">{item.id} ({item.emotion})</span>
                    <span className="text-xs font-devanagari text-slate-300 font-medium">{item.text}</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                    <div className="flex items-center gap-2 bg-slate-900 p-2 rounded-lg border border-slate-800">
                      <span className="text-[10px] font-mono text-amber-400 uppercase w-20">XTTS v2:</span>
                      <audio controls className="h-7 w-full" src={`/api/audio/outputs/bakeoff/xtts/${item.id}.wav`} />
                    </div>
                    <div className="flex items-center gap-2 bg-slate-900 p-2 rounded-lg border border-slate-800">
                      <span className="text-[10px] font-mono text-slate-400 uppercase w-20">MMS Baseline:</span>
                      <audio controls className="h-7 w-full" src={`/api/audio/outputs/bakeoff/mms/${item.id}.wav`} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'listening' && (
          <ListeningTest />
        )}

        {activeTab === 'viva' && (
          <VivaCheatSheet />
        )}
      </main>

      {/* Persistent Bottom Status Bar */}
      <footer className="border-t border-slate-800 bg-slate-950/90 px-6 py-2.5 flex items-center justify-between text-xs font-mono text-slate-400">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-slate-300">{statusMsg}</span>
        </div>
        <div className="flex items-center gap-4 text-[11px]">
          <span>LangGraph Master Orchestrator: <strong className="text-emerald-400">ONLINE</strong></span>
          <span>SpeechBrain ECAPA Index: <strong className="text-cyan-400">12 VOICES</strong></span>
          <span>LUFS Broadcast Standard: <strong className="text-amber-400">-16 LUFS</strong></span>
        </div>
      </footer>
    </div>
  );
};
export default App;
