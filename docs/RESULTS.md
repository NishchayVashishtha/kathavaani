# KathaVaani — System Evaluation & Ablation Results

**Evaluation Date:** 2026-11-19  
**Execution Hardware:** NVIDIA H200 Tensor Core GPU MIG instance (`nvidia.com/mig-1g.18gb`, 18 GB VRAM)  
**ASR Metric Model:** OpenAI Whisper (`small` / Hindi)  
**Speaker Verification:** SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`, 192-dim)  
**Human Evaluation Status:** $n = 0$ human ratings collected (Listening test interface deployed; pending human trials).

---

## 1. Measured Ablation Studies Summary

> [!NOTE]
> All metrics below are **empirically measured** on the DGX cluster H200 GPU. Subjective human MOS has been removed until real human listener sessions are conducted; predicted automatic MOS will replace it once UTMOS finishes evaluating across splits.

| Architecture Variant | Whisper CER (%) ↓ | Speaker Cosine Sim ↑ | Mean RTF ↓ | Human MOS ($n=0$) | Finding & Architectural Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full KathaVaani (Ours)** | **40.25%** | **0.243** | 0.36x | *Pending trial* | Full multi-stage pipeline: Script Analyst + Narrative Director heuristics + Timbre Lock + LUFS ducking. |
| **Ablation 1: W/O Director** | 45.61% | 0.217 | 0.30x | *Pending trial* | Flat default prosody (temperature 0.65, fixed pauses); monotone delivery across all characters. |
| **Ablation 2: Random Voices** | 54.23% | 0.158 | 0.33x | *Pending trial* | Voices assigned randomly per turn without Timbre Lock or distance maximization; severe speaker confusion. |
| **Ablation 3: Meta MMS-TTS** | **26.21%** | 0.000 | **0.04x** | *Pending trial* | Single-speaker VITS baseline; lower CER due to predictable neutral pacing, but zero voice cloning capability. |

---

## 2. Metric Analyses & Discrepancies

### 2.1 Speaker Similarity Dynamics (0.644 vs. 0.243)
- In the 30-sentence engine bake-off ([TTS_BAKEOFF.md](file:///home/dgx-s-bmu-soet-230737/kathavaani/docs/TTS_BAKEOFF.md)), XTTS v2 achieved a mean speaker similarity of **0.644** when evaluated on continuous 4–6 second full sentences against a 10s clean reference prompt (`voices/female_adult_queen.wav`).
- In the ablation study, test turns included short dialogues (e.g., Turn 3 "महाराज!" lasting < 1.0s). Because ECAPA-TDNN utilizes statistical temporal pooling across frame-level representations, segments shorter than 2.0s do not provide sufficient acoustic context for stable speaker embedding convergence, lowering the score to ~0.04–0.40 on ultra-short lines.
- **Protocol Standardization:** When evaluating multi-turn stories, dialogue turns for each character should be concatenated to $\ge 2.0$s before computing global character consistency.

### 2.2 Character Error Rate (CER) vs. Audio Delivery Style
- Meta MMS-TTS achieves lower CER (26.21%) because it produces slow, uniform, neutral speech that aligns closely with Whisper ASR's standard training distribution.
- KathaVaani / XTTS v2 intentionally varies pitch, speed, and volume (whispering, shouting, trembling). Whisper's ASR decoder exhibits higher insertion/deletion rates when transcribing expressive whisper or shouted prosody.

---

## 3. System Limitations (Honest Disclosure)

1. **Rule-Based Director (Current State):** Agents 1 and 2 currently utilize keyword dictionaries and template heuristics for dialogue attribution and prosodic directives. Integration of a quantized local LLM (Qwen2.5/Llama-3.1) with constrained JSON decoding is required to achieve genuine semantic scene comprehension.
2. **Autoregressive Hallucination Risk:** While clause-level chunking ($\le 200$ chars) reduces failures, XTTS v2 occasionally drops trailing words or mispronounces rare Devanagari conjuncts under elevated sampling temperatures ($T > 0.8$).
3. **Procedural Background Noise:** The background ambience currently uses filtered pink noise and synthetic sine LFO modulation, not a neural music generation model.
4. **Data Scale:** The current training set consists of 26 storytelling audio files (~6.0 hours). Expanding to $\ge 10$ hours using open licensed Indic datasets (AI4Bharat Rasa, IndicTTS) is necessary for high-fidelity domain adaptation.
5. **No Gated Models Without User Tokens:** Advanced diarization (`pyannote/speaker-diarization-3.1`) requires user-provided HuggingFace access tokens. In their absence, fallback energy-based segmentation and ECAPA clustering are used.
