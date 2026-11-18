import React, { useState } from 'react';
import { Headphones, CheckCircle2, Play, Pause, Send } from 'lucide-react';

interface ListeningTestProps {
  onSubmitSuccess?: () => void;
}

export const ListeningTest: React.FC<ListeningTestProps> = ({ onSubmitSuccess }) => {
  const [evaluator, setEvaluator] = useState('');
  const [activeClip, setActiveClip] = useState<'A' | 'B' | null>(null);
  const [ratings, setRatings] = useState({
    naturalness: 4,
    expressiveness: 5,
    speaker_consistency: 4,
    audio_quality: 4,
  });
  const [preferredModel, setPreferredModel] = useState<'A' | 'B' | 'equal'>('A');
  const [feedback, setFeedback] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Audio elements for Clip A (KathaVaani XTTS) vs Clip B (MMS baseline)
  const clipAUrl = '/api/audio/outputs/bakeoff/xtts/SENT_01.wav';
  const clipBUrl = '/api/audio/outputs/bakeoff/mms/SENT_01.wav';

  const [audioA] = useState(() => new Audio(clipAUrl));
  const [audioB] = useState(() => new Audio(clipBUrl));

  const togglePlay = (clip: 'A' | 'B') => {
    if (clip === 'A') {
      audioB.pause();
      if (activeClip === 'A') {
        audioA.pause();
        setActiveClip(null);
      } else {
        audioA.play();
        setActiveClip('A');
      }
    } else {
      audioA.pause();
      if (activeClip === 'B') {
        audioB.pause();
        setActiveClip(null);
      } else {
        audioB.play();
        setActiveClip('B');
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const payload = {
        test_id: `blind_ab_${Date.now()}`,
        evaluator_id: evaluator || 'Anonymous Evaluator',
        story_id: 'SENT_01',
        model_a: 'KathaVaani (XTTS v2 + Director)',
        model_b: 'Meta MMS-TTS Baseline',
        naturalness_score: ratings.naturalness,
        expressiveness_score: ratings.expressiveness,
        speaker_consistency_score: ratings.speaker_consistency,
        audio_quality_score: ratings.audio_quality,
        preference: preferredModel,
        feedback: feedback,
      };

      const res = await fetch('/api/listening-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        setSubmitted(true);
        if (onSubmitSuccess) onSubmitSuccess();
      }
    } catch (err) {
      console.error('Failed to submit listening test:', err);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
      <div className="border-b border-slate-800 pb-4 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Headphones className="w-5 h-5 text-amber-400" />
            अंधा तुलनात्मक परीक्षण (Blind A/B Listening Test & MOS Evaluation)
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Standardized Subjective Evaluation Protocol for Deep Learning Speech Synthesis (MUSHRA / MOS Standard)
          </p>
        </div>
        <span className="px-2.5 py-1 rounded-full text-[11px] bg-slate-800 text-slate-300 border border-slate-700 font-mono">
          Human Ratings: n = 0
        </span>
      </div>

      {submitted ? (
        <div className="p-8 text-center space-y-3 bg-emerald-950/20 border border-emerald-800/40 rounded-xl">
          <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto" />
          <h3 className="text-lg font-bold text-emerald-300">मूल्यांकन सफलतापूर्वक दर्ज किया गया!</h3>
          <p className="text-xs text-slate-300">
            Thank you! Your blind ratings have been logged to the research evaluation database (<code>data_storage/listening_test_results.jsonl</code>).
          </p>
          <button
            onClick={() => setSubmitted(false)}
            className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-4 py-2 rounded-lg mt-2"
          >
            Submit Another Evaluation
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Reference Hindi Sentence */}
          <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-500">Benchmark Reference Sentence:</span>
            <p className="text-sm font-devanagari text-amber-200 font-medium">
              "एक पुराने गांव के किनारे एक घना और शांत बरगद का पेड़ था।"
            </p>
          </div>

          {/* Blind Audio Player Blocks */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
              <div>
                <span className="text-xs font-mono uppercase text-slate-400">Audio Sample</span>
                <h4 className="text-base font-bold text-slate-200">System Alpha</h4>
              </div>
              <button
                type="button"
                onClick={() => togglePlay('A')}
                className={`w-11 h-11 rounded-full flex items-center justify-center font-bold transition-all ${
                  activeClip === 'A'
                    ? 'bg-amber-400 text-slate-950 scale-105'
                    : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                }`}
              >
                {activeClip === 'A' ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
              </button>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
              <div>
                <span className="text-xs font-mono uppercase text-slate-400">Audio Sample</span>
                <h4 className="text-base font-bold text-slate-200">System Beta</h4>
              </div>
              <button
                type="button"
                onClick={() => togglePlay('B')}
                className={`w-11 h-11 rounded-full flex items-center justify-center font-bold transition-all ${
                  activeClip === 'B'
                    ? 'bg-amber-400 text-slate-950 scale-105'
                    : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                }`}
              >
                {activeClip === 'B' ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
              </button>
            </div>
          </div>

          {/* 4 MOS Scales (1 to 5) */}
          <div className="space-y-4 pt-2">
            {[
              { key: 'naturalness', label: 'स्वाभाविकता (Naturalness & Prosody)', desc: 'Human-like phrasing vs robotic cadence' },
              { key: 'expressiveness', label: 'भाव प्रकटीकरण (Emotional Expressiveness)', desc: 'Context-appropriate narrative inflection' },
              { key: 'speaker_consistency', label: 'आवाज़ की स्थिरता (Speaker Consistency / Timbre Lock)', desc: 'Distinct character identity without vocal drift' },
              { key: 'audio_quality', label: 'ध्वनि गुणवत्ता (Acoustic Quality & Clarity)', desc: 'Absence of metallic buzz, phasing, or robotic artifacts' },
            ].map(({ key, label, desc }) => (
              <div key={key} className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between">
                  <div>
                    <h5 className="text-xs font-semibold text-slate-200">{label}</h5>
                    <p className="text-[10px] text-slate-400">{desc}</p>
                  </div>
                  <span className="text-xs font-mono font-bold text-amber-400">
                    {ratings[key as keyof typeof ratings]} / 5
                  </span>
                </div>
                <div className="flex gap-2">
                  {[1, 2, 3, 4, 5].map((score) => (
                    <button
                      key={score}
                      type="button"
                      onClick={() => setRatings(prev => ({ ...prev, [key]: score }))}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                        ratings[key as keyof typeof ratings] === score
                          ? 'bg-amber-500 text-slate-950'
                          : 'bg-slate-900 hover:bg-slate-800 text-slate-400'
                      }`}
                    >
                      {score}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Overall Preference */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-200">Overall Preference</label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { val: 'A', label: 'Prefers System Alpha' },
                { val: 'equal', label: 'About Equal' },
                { val: 'B', label: 'Prefers System Beta' },
              ].map(({ val, label }) => (
                <button
                  key={val}
                  type="button"
                  onClick={() => setPreferredModel(val as any)}
                  className={`py-2 px-3 rounded-lg text-xs font-medium border text-center transition-all ${
                    preferredModel === val
                      ? 'bg-amber-500/20 border-amber-500 text-amber-300'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-900'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Evaluator Name & Notes */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="text-[11px] text-slate-400 block mb-1">Evaluator Name / Viva Examiner ID</label>
              <input
                type="text"
                placeholder="Dr. Sharma (Examiner)"
                value={evaluator}
                onChange={(e) => setEvaluator(e.target.value)}
                className="w-full text-xs bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-amber-500"
              />
            </div>
            <div>
              <label className="text-[11px] text-slate-400 block mb-1">Qualitative Observations</label>
              <input
                type="text"
                placeholder="Noticeable pauses and warm timbre..."
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                className="w-full text-xs bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-amber-500"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 transition-all disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
            <span>{submitting ? 'Submitting...' : 'Submit Evaluation (मूल्यांकन सबमिट करें)'}</span>
          </button>
        </form>
      )}
    </div>
  );
};
