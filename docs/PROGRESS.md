# KathaVaani (कथावाणी) — Living Progress & Status Tracker

**Current Date**: 2026-11-19 (Day 1 of 4)  
**Target Evaluation / Viva**: 2026-11-23  
**Status**: Architecture Prototype Operating on NVIDIA H200 GPU; Honesty Audit Performed; Data Expansion & Proper Fine-Tuning Underway.

---

## High-Level Status Summary

```mermaid
flowchart LR
    P0["Phase 0: Environment & Hardware Discovery"] --> P1["Phase 1: TTS Engine Bake-Off & Voice Bank"]
    P1 --> P2["Phase 2: Data Pipeline & Fine-Tuning"]
    P2 --> P3["Phase 3: Multi-Agent Orchestrator"]
    P3 --> P4["Phase 4: KathaVaani Studio Web UI"]
    P4 --> P5["Phase 5: Evaluation & Viva Preparation"]

    classDef done fill:#22c55e,stroke:#15803d,color:#fff;
    classDef inprog fill:#3b82f6,stroke:#1d4ed8,color:#fff;
    classDef audit fill:#eab308,stroke:#a16207,color:#fff;

    class P0 done;
    class P1 done;
    class P2 inprog;
    class P3 inprog;
    class P4 inprog;
    class P5 audit;
```

---

## Daily Log

### Day 1 (19 Nov 2026) — Foundations, Audit & Baseline Execution

#### Done (Empirically Verified with Source Logs)
- [x] **Cluster Discovery & Pod Setup**: Confirmed dedicated NVIDIA H200 MIG `1g.18gb` instance via `kathavaani-worker` pod in `dgx-s-bmu-soet-230737-restricted`. Verified zero VRAM usage and clean execution. ([ENV_REPORT.md](file:///home/dgx-s-bmu-soet-230737/kathavaani/docs/ENV_REPORT.md)).
- [x] **Storage Strategy**: Symlinked caches and checkpoints to `/var/tmp/dgx-s-bmu-soet-230737` (165 GB available) and container NVMe overlay (25 TB) to prevent root filesystem exhaustion.
- [x] **TTS Engine Bake-Off**: Ran 30-sentence benchmark comparing XTTS v2 and MMS-TTS Hindi on H200 GPU. XTTS v2 demonstrated RTF 0.21x (5x real-time) and speaker similarity 0.644 ([TTS_BAKEOFF.md](file:///home/dgx-s-bmu-soet-230737/kathavaani/docs/TTS_BAKEOFF.md)).
- [x] **Voice Bank**: Built 12-voice Bank with 192-dim SpeechBrain ECAPA-TDNN embeddings and FAISS index ([metadata.json](file:///home/dgx-s-bmu-soet-230737/kathavaani/voices/metadata.json)).
- [x] **Honesty Audit Completed**: Created [AUDIT.md](file:///home/dgx-s-bmu-soet-230737/kathavaani/docs/AUDIT.md). Disclosed mel loss leakage bug, eliminated fabricated human MOS values, relabeled rule-based heuristics and procedural noise, and added system limitations.
- [x] **DVAE Restored**: Downloaded official Coqui `dvae.pth` (210 MB) to extract true discrete audio codes from audio instead of self-generated tokens.
- [x] **Web Studio UI**: Built production React 18 SPA bundle in `web/dist/` with zero human MOS until live raters submit.

#### In Progress
- [ ] **Sanity-Check Existing Production**: Automated quality control script on `outputs/studio/story_01` (CER, clipping, SNR, duration, ECAPA consistency).
- [ ] **Data Pipeline Expansion**: Repartition all 1,664 segments across all 26 ingested stories into leakage-free train/val/test splits.
- [ ] **Real Fine-Tuning**: Implement true ground-truth DVAE tokenized training of XTTS GPT decoder on `train.jsonl` and launch background overnight run.

#### Failed / Corrected in Audit
- [x] **Mel Loss Artifact**: Initial smoke test (20 steps) showed rapid loss decrease (2.011 to 0.0309) caused by feeding model-generated tokens back into the loss function (self-consistency leakage). Corrected: downloaded `dvae.pth` for ground-truth audio token extraction.
- [x] **Human MOS Removal**: Removed fabricated MOS numbers (4.6 / 2.4 / 3.1 / 1.8) from all docs and frontend; replaced with `n = 0` until actual listening tests are conducted.

---

## Measured System Milestones & Real Metrics

| Metric | Target | Current Status | Evidence Path |
| :--- | :--- | :--- | :--- |
| **GPU Allocation** | 18 GB VRAM | **16.0 GB addressable on NVIDIA H200 MIG 1g.18gb** | `nvidia-smi` log |
| **XTTS v2 Synthesis RTF** | < 0.5x | **0.21x (5x real-time on H200 GPU)** | `outputs/bakeoff/` |
| **Voice Bank Diversity** | $\ge 12$ voices | **12 distinct voices with 192-dim ECAPA embeddings & FAISS** | `voices/metadata.json` |
| **Audio Loudness Standard** | EBU R128 | **-16 LUFS integrated loudness + -14 dB sidechain ducking** | `kathavaani/agents/post_processor.py` |
| **Human Listening Test Count** | $\ge 10$ raters | **$n = 0$ (Testing interface deployed, trial pending)** | `web/src/components/ListeningTest.tsx` |
