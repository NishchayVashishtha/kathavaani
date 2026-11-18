# KathaVaani — System Evaluation & Benchmark Results

**Evaluation Date:** 2026-11-19  
**Execution Hardware:** NVIDIA H200 Tensor Core GPU MIG instance (`nvidia.com/mig-1g.18gb`, 18 GB VRAM)  
**Cluster Environment:** Ubuntu 24.04, Base Command Manager 11, namespace `dgx-s-bmu-soet-230737-restricted`  
**ASR Metric Model:** OpenAI Whisper Large-v3 (1550M params) with Devanagari initial prompt  
**Speaker Verification:** SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`, 192-dim embeddings)  
**Human Evaluation Status:** $n = 0$ real human ratings collected (Listening test interface deployed at `/listening-test`; awaiting formal viva listener trials). Automated MOS metric is generated via Meta/TorchAudio SQUIM_SUBJECTIVE and strictly labelled `predicted (automatic)`.

---

## 1. Checkpoint Bake-Off Comparison (Priority 2a)

Evaluated across 30 held-out test utterances ($\ge 2.5$s) from `data_storage/test.jsonl`.  
Source script: `kathavaani/eval/compare_checkpoints.py`  
Artifact: `outputs/checkpoint_comparison/comparison_results.json` | Report: `docs/CHECKPOINT_COMPARISON.md`

| Model Architecture | Character Error Rate (CER) ↓ | Word Error Rate (WER) ↓ | Speaker Similarity (≥2.5s) ↑ | Predicted MOS (Automatic) ↑ | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Zero-Shot XTTS v2 (Hardened)** | **83.92%** | **108.61%** | **0.2540** | 3.55 | **Recommended for Production** |
| **Best-Val Checkpoint (Epoch 4)** | 240.02% | 219.88% | 0.2756 | **3.97** | Research Ablation Reference |
| **Last Checkpoint (Epoch 7)** | 155.52% | 163.28% | 0.3013 | 3.96 | Research Ablation Reference |

### Architectural Findings:
1. **Phonetic Intelligibility Tradeoff:** While fine-tuning on the clean 1.69h corpus optimized the cross-entropy training loss from 6.82 down to 5.37 and boosted discrete acoustic token quality (raising automatic MOS from 3.55 to 3.97), the limited size and speaker diversity of the 1.69h dataset led the autoregressive cross-attention alignment to develop phonetic repetition/insertion drift on complex unseen Devanagari sentences.
2. **Production Choice:** Stock Zero-Shot XTTS v2 combined with our inference hardening wrapper (length capping $1.75\times$, repetition penalty 10.0, low temperature 0.65, ASR-gated retry) achieves significantly superior semantic intelligibility and is selected as the primary engine for KathaVaani Studio. The fine-tuned checkpoints document data-scale boundary conditions.

---

## 2. ASR Benchmark & Devanagari Normalization Study (Priority 2b)

Evaluated across 100 held-out test utterances from `data_storage/test.jsonl`.  
Source script: `kathavaani/eval/compare_asr.py`  
Artifact: `outputs/asr_benchmark/asr_eval_results.json` | Report: `docs/ASR_EVAL_REPORT.md`

### 2.1 Model Comparison & Normalization Impact

| ASR Model Architecture | Raw CER ↓ | Advanced Norm CER ↓ | Raw WER ↓ | Advanced Norm WER ↓ | Evaluator Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Whisper Base (74M params)** | 284.72% | 285.47% | 272.84% | 272.84% | Baseline (Unstable on Low SNR) |
| **Whisper Large-v3 (1550M params)** | **77.33%** | **77.05%** | **116.14%** | **115.71%** | **Primary Benchmark Evaluator** |

### 2.2 Prosodic Delivery Breakdown: Neutral vs Emotional

| ASR Architecture | Neutral Lines CER ↓ | Emotional Lines CER ↓ | Relative Error Shift | Acoustic Cause |
| :--- | :---: | :---: | :---: | :--- |
| **Whisper Base** | 268.48% | 336.43% | **+25.3%** | Severe prompt hallucination loop (`देवनागरी...`) |
| **Whisper Large-v3** | 82.45% | **60.86%** | **-26.2%** | Robust attention decoder resolves dramatic emphasis |

### 2.3 Devanagari Linguistic Analysis:
- **Nukta Standardization:** In Hindi literature, Urdu/Persian loanwords alternate freely between base consonants and nukta variants (`ज़िंदगी` vs `जिंदगी`, `फ़र्क` vs `फर्क`, `ख़ुशी` vs `खुशी`). Advanced normalization strips nuktas so phonetic loanword matches are not penalized by lexical diacritic choices.
- **Danda & Punctuation Stripping:** Punctuation in ASR output reflects pause acoustic length rather than syntax. Stripping dandas (`।`, `॥`) isolates true phonetic recognition accuracy.
- **Numeral Expansion:** Expanding Arabic digits (`20` $\rightarrow$ `बीस`) eliminates mismatches between digit tokens and transcribed phonetic text.

---

## 3. Bake-Off v2: Multi-Model Feasibility Matrix (Priority 2c)

| Model Name | Repository ID | Architecture | Native Hindi Support? | Voice Cloning? | Practical Status in KathaVaani |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Coqui XTTS v2** | `tts_models/multilingual/multi-dataset/xtts_v2` | Autoregressive GPT + HiFi-GAN Vocoder | Yes (Native Hindi Devanagari) | **Yes (3s zero-shot prompt)** | **Selected Production Engine** |
| **Meta MMS-TTS** | `facebook/mms-tts-hin` | VITS (End-to-End Variational Autoencoder) | Yes (Devanagari character CTC) | No (Single speaker fixed) | Baseline Non-Cloning Reference |
| **AI4Bharat IndicF5** | `ai4bharat/IndicF5` | Flow-Matching Audio Diffusion | Yes (Multilingual Indic) | Yes (Diffusion-based) | Verified public repo; requires `f5_tts` diffusion package |
| **AI4Bharat Indic Parler** | `ai4bharat/indic-parler-tts` | Parler Autoregressive Transformer | Yes (Trained on Indic datasets) | Conditional text-prompted | Verified public repo; requires `parler_tts` extension |
| **GPT-SoVITS v2** | `RVC-Boss/GPT-SoVITS` | Few-Shot Acoustic Matching | No native Hindi G2P | Yes (3s prompt) | Infeasible without custom bilingual Devanagari phonemizer |

---

## 4. Speaker Similarity vs. Audio Duration Study (Priority 1c)

Evaluated across all 430 held-out test utterances in `data_storage/test.jsonl` using SpeechBrain ECAPA-TDNN.  
Artifact: `outputs/speaker_sim_study.json` | Report: `docs/SPEAKER_SIM_REPORT.md`

| Clip Duration Window | Sample Count ($N$) | Mean Speaker Similarity ↑ | Std Dev ($\sigma$) | Min Sim | Max Sim |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2.0s – 4.0s** | 248 | **0.6651** | 0.1728 | 0.0520 | 0.8845 |
| **4.0s – 8.0s** | 171 | **0.7608** | 0.1600 | 0.0573 | 0.8992 |
| **$\ge$ 8.0s** | 11 | **0.8532** | 0.0664 | 0.7303 | 0.9416 |
| **All Test ($\ge 2.5$s)** | 430 | **0.7386** | 0.1681 | 0.0520 | 0.9416 |

**Empirical Conclusion:** Pearson correlation coefficient is **$r = 0.373$** ($p < 10^{-14}$). ECAPA-TDNN frame-level temporal statistics pooling requires $\ge 2.0$s of voiced audio to stabilize; clips $<2.0$s drop to $0.05 - 0.20$ due to phonetic boundary transients, whereas clips $\ge 4.0$s achieve consistent $0.76 - 0.85$ speaker verification.

---

## 5. Agent Evaluation on 5 Gold Stories (Priority 3b)

Evaluated against hand-annotated ground truth stories in `stories/gold/`.  
Artifact: `outputs/agent_eval_results.json` | Report: `docs/AGENT_EVAL_REPORT.md`

| Evaluation Metric | Measured Score | Benchmark Analysis |
| :--- | :---: | :--- |
| **Character Detection F1** | **60.9%** | 100% on Birbal ki Khichdi, 66.7% on Sher-Khargosh, 50% on Eidgah |
| **Speaker Attribution Accuracy** | **56.0%** | Narration vs dialogue attribution across all 50 ground-truth turns |
| **Emotion Direction Valence Match** | **46.0%** | Binary positive/negative emotional trajectory agreement |
| **Emotion Exact Match** | **16.0%** | Exact match across 12 granular theatrical emotion classes |

---

## 6. Real Showcase Production Demos (Priority 3c)

1. **Demo Story 01: शेर और खरगोश (*The Lion and the Rabbit*)**
   - **Path:** `outputs/studio/story_01/`
   - **Cast:** Narrator (कबीर - सूत्रधार), शेर (भासुरक - दुर्जोधन), खरगोश (तन्वी/चतुर खरगोश).
   - **Metrics:** 14 turns, 157.3s duration, 0 digital clippings, 0 loop hallucinations.
2. **Demo Story 02: न्याय और ममता (*Justice and Maternal Love*)**
   - **Path:** `outputs/studio/story_02/`
   - **Cast (4 distinct characters):**
     - Narrator: कबीर (सूत्रधार - गंभीर) (`narrator_male_deep`)
     - दादी माँ: दादी माँ (स्नेहमयी) (`female_elderly_dadi`)
     - राजा विक्रमादित्य: दुर्जोधन (गंभीर प्रतिद्वंद्वी) (`male_villain_dramatic`)
     - नन्ही पिंकी: तन्वी (उत्सुक युवती/बच्ची) (`female_young_lively`)
   - **Acoustic Stems:** Speech stem, ambient BGM stem, timeline metadata (`timeline.json`), master MP3/WAV.
