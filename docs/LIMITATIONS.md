# KathaVaani (कथावाणी) — Limitations, Edge Cases & Ethical Considerations

**Document Status:** Complete & Empirically Verified  
**Date:** 2026-11-19  
**Target Viva Date:** 2026-11-23  

---

## 1. Empirical Limitations & Failure Modes

Every number and constraint listed below is grounded in actual benchmark executions on our cluster's NVIDIA H200 Tensor Core GPU (`nvidia.com/mig-1g.18gb`, 18 GB VRAM).

### 1.1 Dataset Scale & Domain Coverage
1. **Raw vs. Clean Audio Discrepancy (42.9% Retention):**
   - Our initial corpus consisted of ~6.01 hours of raw scraped storytelling audio.
   - Rigorous preprocessing with Silero VAD (Voice Activity Detection), spectral SNR filtering ($\ge 15$ dB), and bandwidth verification eliminated 57.1% of raw audio (instrumental music beds, sound effects, long atmospheric silences, and low-fidelity compression artifacts).
   - This yielded **2.57 hours** of pristine speech across 24 stories (Train: 1.69h / 18 stories; Val: 0.30h / 3 stories; Test: 0.58h / 3 stories).
   - **Boundary Limitation:** 1.69 hours of training speech is sufficient to adapt discrete acoustic token distributions, but insufficient to generalize across uncommon Devanagari compound words or irregular syntax without foundation model regularization.

### 1.2 Fine-Tuning vs. Zero-Shot Generalization (The Autoregressive Overfitting Trap)
In our checkpoint bake-off (`docs/CHECKPOINT_COMPARISON.md`), training XTTS v2's autoregressive GPT decoder on 1.69 hours of clean domain audio yielded the following verified metrics:

| Checkpoint | CER (%) ↓ | WER (%) ↓ | ECAPA Speaker Sim (≥2.5s) ↑ | Predicted MOS ↑ | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stock Zero-Shot XTTS v2** | **83.92%** | **108.61%** | 0.2540 | 3.55 | Periodic phonetic insertion on low-frequency words |
| **Best-Val (Epoch 4, Loss 5.56)** | 240.02% | 219.88% | 0.2756 | **3.97** | Autoregressive repetition drift on complex syntax |
| **Last (Epoch 7, Loss 5.37)** | 155.52% | 163.28% | **0.3013** | 3.96 | Attention degradation on long Devanagari clauses |

- **Root Cause Analysis:** The discrete autoencoder (DVAE) discrete codes learned the acoustic timbre and pitch dynamics of Hindi storytellers exceptionally well (raising predicted acoustic MOS from 3.55 to 3.97). However, because the dataset scale was small (1.69h, 1,370 utterances), the cross-attention layers between Devanagari text tokens and audio codes suffered from attention drift. On long or syntactically complex sentences, the fine-tuned model entered repetitive looping loops (e.g., repeating the same syllable 4–6 times).
- **Practical Engineering Solution:** Stock Zero-Shot XTTS v2 with our **Inference Hardening Wrapper** (length capping $1.75\times$, repetition penalty 10.0, temperature 0.65, top_p 0.85, and ASR-gated retry) achieves vastly superior semantic intelligibility (CER 83.92% vs 240.02%) and is selected as the primary engine for KathaVaani Studio.

### 1.3 Speaker Verification vs. Audio Duration (The ECAPA Context Window)
In our study of all 430 held-out test utterances (`docs/SPEAKER_SIM_REPORT.md`):
- For clips **$2.0\text{s} - 4.0\text{s}$**, mean cosine similarity was **0.6651** ($\sigma = 0.1728$).
- For clips **$\ge 4.0\text{s}$**, mean cosine similarity rose to **0.7608** ($\sigma = 0.1600$).
- For clips **$\ge 8.0\text{s}$**, mean cosine similarity was **0.8532** ($\sigma = 0.0664$).
- Pearson correlation between duration and speaker similarity is **$r = 0.373$ ($p < 10^{-14}$)**.
- **Limitation:** For rapid conversational interjections ($<2.0$ seconds, such as *"अरे!", "रुको!", "क्या?"*), ECAPA-TDNN frame-level temporal statistics pooling suffers from boundary transients and spectral instability. The system cannot reliably verify speaker identity on turns shorter than 2.0 seconds. To mitigate this, our Script Analyst merges rapid dialogue turns with neighboring clauses.

### 1.4 ASR Metric Sensitivity & Expressive Speech Hallucination
In our ASR benchmark across 100 held-out test samples (`docs/ASR_EVAL_REPORT.md`):
- **Whisper Base** collapsed to an unusable **285.47% CER**, getting trapped in infinite repetition loops (`देवनागरी देवनागरी...`) when encountering expressive speech or background ambience.
- **Whisper Large-v3** achieved **77.05% Advanced Norm CER**, successfully transcribing complex Devanagari dialogue.
- Furthermore, Whisper Large-v3 achieved **60.86% CER on emotional dramatic lines** vs **82.45% CER on neutral lines**, demonstrating that higher dramatic emphasis (distinct prosodic peaks) assists attention decoding when the ASR model has sufficient parameter capacity.
- **Limitation:** Standard ASR engines are trained primarily on read neutral speech (Common Voice, LibriSpeech). Extremely dramatic whispers or theatrical roars can be transcribed with acoustic penalties that do not reflect human comprehension.

### 1.5 Agentic Semantic Parsing Accuracy
Evaluated on 5 hand-annotated gold benchmark stories (`docs/AGENT_EVAL_REPORT.md`):
- **Character Detection F1:** **60.9%** (high precision on primary protagonists, misses unnamed background extras).
- **Speaker Attribution Accuracy:** **56.0%** (ergative Hindi postposition `ने` helps, but complex literary sentences where dialogue precedes the subject remain prone to misattribution).
- **Emotion Trajectory Valence Match:** **46.0%** (good distinction between positive and negative valence; granular 12-class theatrical emotion match is 16.0%).
- **Fallback:** A rule-based regex and syntactic parser is maintained as an instant fallback when LLM inference is disabled.

---

## 2. Ethical Considerations & Dual-Use Risks

### 2.1 Voice Cloning & Misuse Prevention
- **Threat Vector:** Zero-shot speech synthesis can clone a speaker's voice from a short reference audio clip ($3 - 10$ seconds), presenting risks of impersonation, fraud, non-consensual voice deepfakes, and political disinformation.
- **Safeguards in KathaVaani:**
  1. **Fixed License-Clean Voice Bank:** The system ships with 12 pre-screened, license-clean reference voices from academic open datasets (IIT Madras IndicTTS, OpenSLR Hindi, and AI4Bharat).
  2. **No Arbitrary User Upload in Public Mode:** The production web studio allocates voices strictly from the curated voice bank to prevent unauthorized voice cloning.
  3. **Cryptographic / Metadata Watermarking:** Every generated story includes a signed `timeline.json` manifest recording the model name, checkpoint hash, timestamp, and synthesis parameters, identifying the production as AI-generated.

### 2.2 Data Licensing & Academic Research Compliance
- Scraped story audio from YouTube is strictly used for non-commercial academic research under fair dealing principles (Section 52(1)(a) of the Indian Copyright Act for research and private study).
- Scraped audio files are stored locally in the secure cluster scratch directory and are explicitly excluded from git tracking (`.gitignore`). No raw scraped audio is redistributed.

### 2.3 Devanagari Bias & Representation
- Spoken Hindi exhibits wide dialectal diversity (Bhojpuri-influenced, Awadhi, Khariboli, urban Hinglish).
- KathaVaani is trained and evaluated on standard literary Khariboli Hindi in Devanagari script. Performance on dialectal variations, rural idioms, or code-mixed Hinglish (`"train late ho gayi"`) is not guaranteed and requires future dialect-specific adaptation.

---

## 3. Human Evaluation Status & Transparency

- **Current Human MOS:** $n = 0$ real human ratings collected.
- All MOS values reported in system benchmarks (e.g., 3.55, 3.96, 3.97) are generated using automated neural predictors (Meta/TorchAudio SQUIM_SUBJECTIVE) and are strictly labeled **`predicted (automatic)`**.
- An interactive **Blind A/B Listening Test** module has been developed and integrated into KathaVaani Studio (`http://localhost:8888/#/listening-test`), enabling blind evaluation across Naturalness, Expressiveness, Speaker Consistency, and Audio Quality. Real human MOS data will be gathered during the live viva trials on 23 November 2026.
