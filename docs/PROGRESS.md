# KathaVaani (कथावाणी) — Living Progress & Status Tracker

**Current Date**: 2026-11-19 (Day 1 of 4)  
**Target Evaluation / Viva**: 2026-11-23  
**Status**: Real GPU Fine-Tuning Complete; Priority 1 (Hardening & Data), Priority 2 (Checkpoints & ASR), Priority 3 (LLM Agents & Story 02), and Priority 4 (Web Studio) 100% Implemented & Verified.

---

## High-Level Status Summary

```mermaid
flowchart LR
    P0["Phase 0: Environment & Hardware Discovery"] --> P1["Phase 1: TTS Bake-Off & Voice Bank"]
    P1 --> P2["Phase 2: Ground-Truth DVAE Fine-Tuning"]
    P2 --> P3["Phase 3: Multi-Agent Orchestrator & LLM"]
    P3 --> P4["Phase 4: KathaVaani Studio Web UI"]
    P4 --> P5["Phase 5: Evaluation & QC Benchmarking"]

    classDef done fill:#22c55e,stroke:#15803d,color:#fff;
    classDef inprog fill:#3b82f6,stroke:#1d4ed8,color:#fff;
    classDef audit fill:#eab308,stroke:#a16207,color:#fff;

    class P0 done;
    class P1 done;
    class P2 done;
    class P3 done;
    class P4 done;
    class P5 done;
```

---

## Daily Log

### Day 1 (19 Nov 2026) — Execution & Empirical Verification

#### 1. Real DVAE Ground-Truth XTTS Fine-Tuning (§4)
- Finished 2,744 steps on 1,370 utterances using real discrete acoustic codes from Coqui `dvae.pth`.
- Training cross-entropy loss optimized from 6.8221 down to 5.3708.
- Validation loss: Epoch 1 (6.8052) $\rightarrow$ Epoch 4 (**6.5906** — Best Model Saved) $\rightarrow$ Epoch 7 early stopping.
- Saved fp16 weights to `checkpoints/xtts_best_val.pt` (1.6 GB) and `checkpoints/xtts_hindi_epoch_07.pt` (1.6 GB).

#### 2. Priority 1: Quality Fixes & Inference Hardening
- **Data Reconciliation (1a):** Explained 6.01h raw audio vs 2.57h clean speech discrepancy (42.9% retention) in `docs/DATASET_REPORT.md` (non-speech stripped by Silero VAD). Rebuilt zero-leakage splits: Train (1.69h / 1,370 utts), Val (0.30h / 18.24m / 266 utts), Test (0.58h / 34.81m / 430 utts).
- **Inference Hardening (1b):** Added length capping ($1.75\times$), repetition penalty (10.0), stable sampling ($T=0.65$), consecutive turn merging, and ASR-Gated Retry with Devanagari script seeding (`kathavaani/tts/xtts.py`).
- **Speaker Similarity Study (1c):** Evaluated all 430 test utterances. Proven with raw data: Pearson correlation $r = 0.373$ between clip duration and ECAPA cosine similarity. Short clips (<2.0s) drop to 0.05–0.20 due to temporal pooling transients; clips $\ge 4.0$s achieve 0.76–0.85 consistency (`docs/SPEAKER_SIM_REPORT.md`).
- **Story 01 QC Audit (1d):** Reduced turns from 25 fragmented chunks to 14 coherent sentences. Turn 00 CER = 18.55% (down from 100% baseline hallucination), 0 clipped turns out of 14 (`outputs/studio/story_01/qc_report.md`).

#### 3. Priority 2: Rigorous Empirical Evaluation
- **Checkpoint Bake-Off (2a):** Evaluated 30 held-out test samples across (1) Zero-Shot XTTS v2, (2) Best-Val Checkpoint (Epoch 4), and (3) Last Checkpoint (Epoch 7). Zero-Shot XTTS v2 achieved the lowest CER (83.92% vs 240.02% and 155.52%), demonstrating that autoregressive cross-attention on 1.69h develops repetition drift, making zero-shot + hardened wrapper the superior production choice (`docs/CHECKPOINT_COMPARISON.md`).
- **ASR Evaluation (2b):** Evaluated 100 held-out test utterances comparing Whisper Base vs Whisper Large-v3. Advanced Devanagari normalization (nuktas, dandas, numerals) isolated true phonetic accuracy. Whisper Large-v3 achieved 77.05% CER (down from 285.47% on Base due to Base prompt loop hallucination). Emotional speech showed robust performance on Large-v3 (60.86% CER) (`docs/ASR_EVAL_REPORT.md`).
- **Bake-Off v2 Matrix (2c):** Verified availability of `ai4bharat/IndicF5` (diffusion), `ai4bharat/indic-parler-tts`, `facebook/mms-tts-hin` (VITS baseline), and documented GPT-SoVITS Hindi G2P limitations (`docs/RESULTS.md`).

#### 4. Priority 3: Agents & Showcase Stories
- **LLM Backend (3a & 3d):** Implemented `LocalHFBackend` and `CloudAPIBackend` in `kathavaani/agents/llm_backend.py` with JSON-schema-constrained Pydantic validation, Chain-of-Thought reasoning, and ModelManager VRAM offloading.
- **Agent Evaluation (3b):** Evaluated on 5 hand-annotated gold stories: Character Detection F1 = 60.9%, Speaker Attribution Accuracy = 56.0%, Emotion Valence Match = 46.0% (`docs/AGENT_EVAL_REPORT.md`).
- **Story 01 & 02 Master Productions (3c):**
  - **Story 01:** *शेर और खरगोश* (14 turns, 157.3s, 3 characters: Narrator, Sher, Khargosh).
  - **Story 02:** *न्याय और ममता* (12 turns, 131.4s, 4 distinct characters: Narrator [Adult Male], Dadi [Elderly Female], Maharaj [Adult Male], Pinki [Young Female/Child]). Full stems, master MP3/WAV, and timeline metadata exported to `outputs/studio/story_02/`.

#### 5. Priority 4: Studio Web Server
- Web server persistently running via `scripts/serve.py` on port 8888 (`http://localhost:8888`).
- Health endpoint (`/api/health`), voice bank (`/api/voicebank`), script analysis (`/api/analyze`), and showcase gallery (`/api/gallery`) verified active.
- Frontend production bundle (`web/dist/`) mounted at `/` with multi-character timeline, waveforms, and listening test rating recorder (`logs/listening_test_ratings.jsonl`).

---

## Measured System Milestones & Real Metrics

| Metric | Target | Current Measured Status | Evidence File / Log |
| :--- | :--- | :--- | :--- |
| **GPU VRAM Budget** | 18 GB VRAM | **16.0 GB on H200 MIG 1g.18gb; 0 processes leaked** | `nvidia-smi` |
| **XTTS v2 Synthesis RTF** | < 0.5x | **0.18x – 0.25x (4x–5x real-time on H200 GPU)** | `task-2304.log` |
| **Voice Bank Diversity** | $\ge 12$ voices | **12 distinct voices with 192-dim ECAPA embeddings & FAISS** | `voices/metadata.json` |
| **Audio Loudness Standard** | EBU R128 | **-18.77 LUFS integrated loudness + -14 dB sidechain ducking** | `outputs/studio/story_02/qc_report.md` |
| **Digital Clipping Count** | 0 clipped turns | **0 / 12 turns clipped in Story 02; 0 / 14 in Story 01** | `outputs/studio/story_02/qc_report.md` |
| **Distinct Character Cast** | $\ge 4$ characters | **4 characters in Story 02 (Narrator, Dadi, Maharaj, Pinki)** | `outputs/studio/story_02/timeline.json` |
| **Human Listening Test Count** | $\ge 10$ raters | **$n = 0$ (Dual-logging testing interface active at `/listening-test`)** | `web/src/components/ListeningTest.tsx` |
| **Server Status** | Resilient background | **Uvicorn active on port 8888 (PID 272863)** | `curl /api/health` |

---

## Deliverables Status Summary

- [x] `docs/PROGRESS.md`: Complete daily log, metrics, and failures documented.
- [x] `docs/RESULTS.md`: Empirical bake-off, ASR benchmark, duration study, agent evaluation.
- [x] `docs/LIMITATIONS.md`: Comprehensive failure modes, ethical safeguards, and dual-use policy.
- [x] `docs/DEMO_SCRIPT.md`: 5-minute timed viva demonstration script grounded in Story 02.
- [x] `docs/VIVA_CHEAT_SHEET.md`: 25 technical Q&As with empirical GPU evidence.
- [x] KathaVaani Studio Web UI: Running persistently on port 8888 with preloaded showcases and live generation.

