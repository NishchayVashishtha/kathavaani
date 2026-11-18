# KathaVaani — Data Sources & Model Licenses

All datasets, audio recordings, and pretrained model weights used in KathaVaani adhere to ethical AI guidelines, academic research standards, and open-source licenses.

---

## 1. Primary Datasets

| Dataset | Source URL | License | Purpose in KathaVaani |
| :--- | :--- | :--- | :--- |
| **Google Drive Storytelling Corpus (`hindi_tts_data`)** | `https://drive.google.com/drive/folders/17OKMSInQ7-8KFNUn0voAYPhFzViLAFra` | Academic Research Fair Use / Provided by User | Multi-character Hindi storytelling audio for fine-tuning & expressive diarization. |
| **AI4Bharat Rasa (Expressive Indic TTS)** | `https://github.com/AI4Bharat/indic-tts` | CC-BY 4.0 | Expressive Hindi speech corpus with emotional variations (Rasa). Base training & Voice Bank. |
| **AI4Bharat Kathbath** | `https://huggingface.co/datasets/ai4bharat/indic-superb` | CC-BY 4.0 | Hindi speech recognition & speaker verification validation. |
| **Mozilla Common Voice Hindi (v17)** | `https://commonvoice.mozilla.org/hi/datasets` | CC0 (Public Domain) | Neutral & demographic speaker references for Voice Bank candidates. |
| **Premchand & Panchatantra Stories** | Project Gutenberg / Hindwi Public Domain | Public Domain | Test stories and gold-standard evaluation set for character attribution. |

---

## 2. Pretrained Models & Architectures

| Model | Origin / Organization | License | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **XTTS v2** | Coqui / Idiap Research | Coqui Public Model License 1.0.0 (Research) | Primary multi-lingual zero-shot TTS engine with native Hindi support. |
| **AI4Bharat IndicF5 / Parler-TTS** | AI4Bharat / IIT Madras | MIT / Apache 2.0 | Indic description-conditioned expressive TTS bake-off candidate. |
| **MMS-TTS Hindi** | Meta AI | CC-BY-NC 4.0 | Baseline single-speaker Hindi VITS model. |
| **Whisper large-v3** | OpenAI | Apache 2.0 | Round-trip ASR evaluation (CER/WER) and automated ASR transcription. |
| **SpeechBrain ECAPA-TDNN** | SpeechBrain Consortium | Apache 2.0 | Speaker embedding extraction, Voice Bank indexing, and timbre consistency tracking. |
| **Demucs (htdemucs)** | Meta Research | MIT | Audio preprocessing: speech vocal stem isolation and BGM stripping. |
| **MusicGen Small** | Meta AudioCraft | MIT | Scene-conditioned procedural background music generation. |
| **Qwen 2.5 7B-Instruct** | Alibaba Cloud Qwen Team | Apache 2.0 | Script analysis, dialogue attribution, and chain-of-thought emotional prosody. |
