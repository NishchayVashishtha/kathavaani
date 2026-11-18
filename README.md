# KathaVaani (कथावाणी)
### Context-Aware, Expressive, Multi-Speaker Text-to-Speech System for Hindi Storytelling

KathaVaani is an experimental speech synthesis framework designed to transform raw, unstructured Devanagari Hindi storytelling literature into multi-character audio dramas featuring character identity persistence, narrative prosody modeling, and broadcast audio mastering.

Built and benchmarked on a dedicated **NVIDIA H200 Tensor Core GPU MIG instance (`1g.18gb`, 18 GB VRAM)** on an academic DGX cluster.

---

## 1. Architecture Overview

```
Hindi Story Text (Devanagari)
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Master Orchestrator (LangGraph 6-Stage State Machine)          │
└─────────────────────────────────────────────────────────────────┘
  │ Stage 1: Script Analyst     ──> Characters & dialogue turns (gender/honorifics)
  │ Stage 2: Narrative Director  ──> Prosodic parameters (emotion, intensity, speed, pauses)
  │ Stage 3: Voice Allocator     ──> 12-voice SpeechBrain ECAPA-TDNN Bank + Timbre Lock
  │ Stage 4: TTS Synthesis Core  ──> XTTS v2 with temperature modulation (0.21x RTF)
  │ Stage 5: Audio Engineer      ──> EBU R128 (-16 LUFS), procedural ambience, -14 dB ducking
  │ Stage 6: Verification Loop   ──> Whisper round-trip CER & ECAPA consistency audit
            ▼
Master Audio (WAV / MP3) + timeline.json + Per-Turn Stems
```

---

## 2. Real Empirical Benchmarks (H200 GPU)

All metrics are measured directly on cluster hardware:

### 30-Sentence TTS Engine Bake-Off
| Engine | Real-Time Factor (RTF) ↓ | ECAPA Speaker Sim ↑ | Whisper CER (%) ↓ | Peak VRAM | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Coqui XTTS v2** | **0.21x (5x real-time)** | **0.644** | 38.14% (12.7% neutral) | 3.00 GB | **Selected Primary Engine** |
| **Meta MMS-TTS Hindi** | **0.01x** | 0.000 (fixed voice) | **23.08%** | 1.31 GB | Non-cloning Baseline |

*Detailed benchmark breakdown available in [docs/TTS_BAKEOFF.md](docs/TTS_BAKEOFF.md).*

### Architectural Ablations
| Configuration Variant | Whisper CER (%) ↓ | Speaker Cosine Sim ↑ | Mean RTF ↓ | Human MOS ($n=0$) |
| :--- | :---: | :---: | :---: | :---: |
| **Full KathaVaani (Ours)** | **40.25%** | **0.243** | 0.36x | *Pending trial* |
| **Ablation 1: W/O Director** | 45.61% | 0.217 | 0.30x | *Pending trial* |
| **Ablation 2: No Distance Maximization** | 54.23% | 0.158 | 0.33x | *Pending trial* |
| **Ablation 3: Meta MMS-TTS Baseline** | 26.21% | 0.000 | 0.04x | *Pending trial* |

*Detailed ablation analysis and metric explanations in [docs/RESULTS.md](docs/RESULTS.md).*

---

## 3. Quick Start

### Prerequisites
- Python 3.10+ (managed via `uv`)
- FFmpeg 7.0+
- NVIDIA GPU with $\ge 12$ GB VRAM (CUDA 12.1+)

### Synthesis via CLI
```bash
# Run complete end-to-end production of a Hindi story
python -m kathavaani.cli stories/01_panchatantra_sher_aur_khargosh.txt --out outputs/my_story/
```

### Launch Web Studio Server
```bash
# Start background server with automatic port-forwarding
./scripts/start_server.sh
# Open in browser: http://localhost:8000
```

---

## 4. System Limitations (Honest Disclosure)

1. **Rule-Based Director (Current State):** Dialogue attribution and emotional prosody directives currently rely on Devanagari keyword dictionaries and syntactic templates. Direct integration with a quantized open LLM (Qwen2.5/Llama-3.1) is under active development.
2. **ASR Evaluation Mismatch:** Whisper ASR yields higher CER on expressive speech (whispering, shouting) compared to monotone speech, penalizing expressive prosody despite improved perceived listening quality.
3. **Procedural Background Noise:** Ambience is currently synthesized using DSP procedural filtered noise beds and LFO modulations, not a generative neural music model.
4. **Data Scale:** The current corpus contains ~6 hours of Hindi storytelling speech. Expanding to $\ge 10$ hours with open licensed Indic datasets is underway.
5. **Human Evaluation:** Subjective MOS evaluation has $n=0$ ratings collected as of 19 Nov 2026. The Blind A/B interface is active in the studio to collect formal examiner and listener ratings.

---

## 5. Academic Documentation
- [docs/AUDIT.md](docs/AUDIT.md) — Comprehensive honesty audit and issue resolutions.
- [docs/VIVA_CHEAT_SHEET.md](docs/VIVA_CHEAT_SHEET.md) — 25 technical viva questions and model answers.
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) — 5-minute timed live viva demonstration protocol.
- [docs/DECISIONS.md](docs/DECISIONS.md) — Architecture decisions log (001–009).
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — Licenses, URLs, and attribution for all models and datasets.
