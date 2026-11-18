# KathaVaani (कथावाणी) — Comprehensive Honesty Audit

**Audit Date:** 2026-11-19  
**Auditor:** ML Engineer / Antigravity Agent  
**Standard:** Strict Academic Honesty (Zero Tolerance for Unsubstantiated or Fabricated Claims)

---

## 1. Audit Summary Table

| Claim | Document(s) | Source Script / Log | Verified? | Root Cause / Issue | Action Taken |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Human Subjective MOS (4.6, 2.4, 3.1, 1.8)** | `docs/RESULTS.md`, `docs/DEMO_SCRIPT.md`, `docs/VIVA_CHEAT_SHEET.md`, `web/src/components/VivaCheatSheet.tsx` | None (No human listening study conducted yet; page was only just created) | **NO (Fabricated)** | Estimated / proposed scores were erroneously entered into tables as measured human data. | **DELETED everywhere.** Replaced with automatic predicted MOS (UTMOS / NISQA) explicitly marked as `predicted (automatic)`. UI set to `n = 0 human ratings`. |
| **"Fine-Tuned XTTS"** | `docs/RESULTS.md`, `docs/PROGRESS.md`, `docs/VIVA_CHEAT_SHEET.md` | `logs/train_loss.jsonl` (20 steps on 181 utterances) | **NO (Misleading label)** | 20 steps is an initial training smoke test, not a convergence fine-tuning run. | **Relabeled to "Smoke Test (20 steps)"** in all documents until the full multi-epoch background training run completes (§4). |
| **Mel Loss Drop: 2.011 → 0.0309 in 20 steps** | `logs/train_loss.jsonl`, `docs/PROGRESS.md` | `logs/train_loss.jsonl`, `kathavaani/tts/finetune/train_xtts.py` | **NO (Artifact of Label Leakage)** | In `train_xtts.py`, `model.gpt.generate` codes were fed back as training targets. The model evaluated its own generated token sequence, collapsing perplexity artificially. | **Root cause documented in `docs/DECISIONS.md`.** Downloaded official `dvae.pth` (210 MB) to extract true ground-truth discrete codes from actual audio before real training. |
| **Speaker Similarity: 0.644 vs. 0.243** | `docs/TTS_BAKEOFF.md` (0.644) vs. `docs/RESULTS.md` (0.243) | `outputs/bakeoff/` vs. `outputs/ablations/` | **PARTIAL (Acoustic Duration Artifact)** | In bake-off, single 10s reference clip was tested on full sentences. In ablations, short dialogue lines (< 1s, e.g. "महाराज!") caused ECAPA temporal pooling degradation (needs $\ge 2$s for stable embedding). | **Unified metric definition:** SpeechBrain ECAPA-TDNN cosine similarity computed strictly on concatenated utterances $\ge 2.0$s with silence trimmed. Documented in `docs/DECISIONS.md`. |
| **"Random Voices" as Ablation 2** | `docs/RESULTS.md`, `docs/VIVA_CHEAT_SHEET.md`, `web/src/components/VivaCheatSheet.tsx` | `kathavaani/eval/run_ablations.py` | **INACCURATE LABEL** | Randomly sampling voices across turns simulates catastrophic drift, but is not the exact ablation of "Voice Allocator distance optimization without random sampling". | **Renamed to "Ablation 2: Static Allocation without Distance Maximization / Timbre Drift"** and re-implementing exact ablation in §9. |
| **Chain-of-Thought "Why this emotion?"** | `docs/PROGRESS.md`, `docs/DEMO_SCRIPT.md`, `web/src/components/DirectorTimeline.tsx` | `kathavaani/agents/narrative_director.py` | **TEMPLATED (Not LLM CoT)** | In current code, `_evaluate_turn_emotion` uses Hindi emotional keyword dictionary matching + templated rationale strings, not a live autoregressive LLM. | **Relabeled as "Rule-Based Heuristic Rationale"** until local Qwen-2.5/Llama-3 LLM integration is connected in §6. |
| **Procedural Ambience as "Soundscape"** | `docs/PROGRESS.md`, `docs/DEMO_SCRIPT.md` | `kathavaani/agents/post_processor.py` | **ACCURATE BUT OVER-CLAIMED** | Ambience is generated via filtered pink noise and synthetic sine LFO modulation, not neural generative audio (MusicGen/AudioGen). | **Explicitly labeled as "DSP Procedural Noise Ambience"** across all documents. |
| **Dataset: 6.01 h (1,664 segments) vs. 291 in splits** | `docs/DATASET_REPORT.md`, `docs/PROGRESS.md` | `data_storage/segments/` (1,664 files) vs `train.jsonl` (181 files) | **UNEXPLAINED DROP** | The segmentation script had processed 13 of 26 stories when the initial split script was run, leaving half the data unpartitioned. | **Rebuilt manifests** on all 1,664 segments across all 26 stories. Updated `docs/DATASET_REPORT.md` with true hours, speaker counts, and durations. |
| **"Emotion Annotation = RMS Energy"** | `data_pipeline/06_annotate.py`, `docs/DATASET_REPORT.md` | `data_pipeline/06_annotate.py` | **INACCURATE LABEL** | RMS energy measures acoustic power/volume, not affective emotion (valence/arousal). | **Relabeled to "Acoustic Energy Profile"**. Real emotion tagging assigned to audio SER model / text LLM in §3. |
| **"100% Implemented / Fully Verified / Research-Grade"** | `docs/PROGRESS.md`, milestone reports | Various docs | **OVER-CLAIMED** | System is a functioning functional prototype with empirical benchmarks, but has key uncompleted steps (real multi-epoch training, live LLM integration, real human listening tests). | **Removed all "100% done / research-grade" marketing phrasing.** Added explicit `LIMITATIONS` sections to `README.md` and `docs/RESULTS.md`. |

---

## 2. Detailed Root-Cause Analyses

### 2.1 Mel Loss Collapse (2.011 → 0.0309)
- **Investigation:** In `kathavaani/tts/finetune/train_xtts.py`, lines 167–189:
  ```python
  with torch.no_grad():
      gpt_codes = model.gpt.generate(
          cond_latents=gpt_cond_latent, text_inputs=text_tokens, ...
      )
  loss_text, loss_mel, logits = model.gpt(
      text_tokens, text_len, gpt_codes, expected_output_len, cond_latents=gpt_cond_latent
  )
  ```
- **Finding:** The model evaluated next-token prediction against acoustic tokens that the *same model* had just generated. Because temperature sampling from the model's own distribution yields tokens that already lie in the highest logit regions of the model, cross-entropy loss immediately collapsed to near zero.
- **Remedy:** In XTTS v2, true speech tokens must be extracted using the Discrete VAE (`dvae.pth`, 210 MB) directly from the ground-truth audio recording:
  `audio_codes = dvae.get_codebook_indices(dvae_mel)`.
  We have downloaded `dvae.pth` to `models_cache/dvae.pth` and will use it in the real fine-tuning pipeline.

### 2.2 Speaker Similarity Discrepancy (0.644 vs 0.243)
- **Investigation:** In `bakeoff.py`, evaluation was performed on 30 full sentences (mean duration 4.5s) comparing against a clean 10s reference clip (`voices/female_adult_queen.wav`). In `run_ablations.py`, turns 0–3 included ultra-short dialogues (e.g. Turn 3 "महाराज!" lasting only 0.8 seconds).
- **Finding:** ECAPA-TDNN's temporal statistics pooling layer requires at least 2.0–3.0 seconds of vocal content to produce a stable 192-dimensional speaker identity representation. For segments < 1.0s, the embedding vector is heavily corrupted by phoneme transients and zero-padding, leading to an artificially depressed cosine similarity of 0.039–0.200.
- **Remedy:** Standardize the speaker similarity evaluation protocol: always evaluate on utterances $\ge 2.0$ seconds, or concatenate a character's dialogue turns within a scene before computing cosine similarity against the reference voice profile.

### 2.3 Dataset Split Discrepancy (1,664 segments vs 291 in splits)
- **Investigation:** When `data_pipeline/07_build_manifest.py` was originally executed, the background segmentation process (`04_transcribe_segment.py`) had only completed stories 1 through 13.
- **Finding:** 1,664 segments exist on disk across all 26 ingested stories in `data_storage/segments/`, but only 291 were indexed in the initial split.
- **Remedy:** Re-run the manifest partitioning on all 1,664 segments, verifying that story boundaries are strictly respected to guarantee zero leakage between train, val, and test splits.
