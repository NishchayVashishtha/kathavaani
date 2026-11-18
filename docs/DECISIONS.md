# KathaVaani (कथावाणी) — Architecture & Engineering Decisions Log

Every technical decision made during the design, development, and evaluation of KathaVaani is documented here with alternatives considered, technical rationale, and empirical evidence.

---

## Decision 001: Execution Engine & GPU Orchestration Strategy
- **Date**: 2026-11-19
- **Decision**: Orchestrate GPU jobs on the cluster using Kubernetes pod and job definitions targeting namespace `dgx-s-bmu-soet-230737-restricted` with limits `nvidia.com/mig-1g.18gb: 1` wrapped in `scripts/run_gpu.sh`, while preserving SLURM `.sbatch` script interfaces for standard academic cluster portability.
- **Alternatives Considered**:
  1. Direct SLURM job submission via `srun` / `sbatch`.
  2. Local execution on login headnode (`bmu-headnode`).
- **Rationale**:
  - Phase 0 discovery verified that student allocations on this DGX cluster are strictly managed via Kubernetes with Kyverno admission enforcement (`validate.kyverno.svc-fail` rule `require-gpu`). Non-GPU jobs or jobs without explicit `nvidia.com/mig-1g.18gb: 1` are blocked at admission.
  - Headnode has no physical GPU access (no `/dev/nvidia*` devices) and running heavy compute there violates cluster acceptable use policies.
  - Kubernetes provides pre-pulled NVIDIA CUDA/PyTorch images (`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`) on the worker node with access to the full 18 GB H200 MIG instance.
- **Evidence**:
  - `kubectl apply` test pod executed and verified:
    `PyTorch: 2.4.0 CUDA available: True Device: NVIDIA H200 MIG 1g.18gb` in 8 seconds.

---

## Decision 002: Storage Partitioning & Cache Routing
- **Date**: 2026-11-19
- **Decision**: Place all large caches (`HF_HOME`, `TORCH_HOME`), model weights, virtual environments, and datasets on `/var/tmp/dgx-s-bmu-soet-230737/` (and `/tmp` inside GPU containers), symlinking them back to the repository.
- **Alternatives Considered**:
  1. Store directly inside `$HOME/kathavaani`.
  2. Store inside `/tmp` on the headnode.
- **Rationale**:
  - The root filesystem hosting `$HOME` (`/dev/sdb6`, 5.3 TB) is 100% full with only 7.3 GB remaining due to cluster-wide shared home directories.
  - Partition `/dev/sdb5` mounted at `/var` has **165 GB available space** with user-writable `/var/tmp`.
  - Inside GPU containers on `bmu-worker`, the local NVMe overlay root provides **25 TB available space**.
- **Evidence**:
  - `df -h /var/tmp` confirms 165 GB free.
  - Container `df -h` confirms 25 TB free on worker overlay.

---

## Decision 003: Python Toolchain & Virtual Environment Manager
- **Date**: 2026-11-19
- **Decision**: Use `uv 0.13.9` to manage an isolated Python 3.10.22 environment and install dependencies.
- **Alternatives Considered**:
  1. System `python3.12-venv` (failed because `python3.12-venv` package is missing on the host and requires `sudo apt install`).
  2. Miniconda / Conda installation from scratch.
- **Rationale**:
  - `uv` is a standalone statically compiled binary that does not depend on system Python packages or apt permissions.
  - Python 3.10 has optimal compatibility with Coqui TTS / XTTS v2, PyTorch 2.4, and speech processing packages (`librosa`, `soundfile`, `numba`).
  - Package installation is an order of magnitude faster than standard pip.
- **Evidence**:
  - `uv 0.13.9` downloaded, installed, and generated Python 3.10.22 venv in `/var/tmp/dgx-s-bmu-soet-230737/venv` in 2.1 seconds.

---

## Decision 004: TTS Primary Engine Selection (XTTS v2 vs. GPT-SoVITS & MMS-TTS)
- **Date**: 2026-11-19
- **Decision**: Select Coqui XTTS v2 as the primary storytelling engine and Meta MMS-TTS Hindi as the baseline comparison engine. Reject GPT-SoVITS as primary due to lack of native Devanagari Hindi text frontend.
- **Alternatives Considered**:
  1. GPT-SoVITS (proposals often suggest this without verifying Hindi G2P support).
  2. Meta MMS-TTS Hindi (`facebook/mms-tts-hin`).
  3. Coqui XTTS v2 (`tts_models/multilingual/multi-dataset/xtts_v2`).
- **Rationale**:
  - GPT-SoVITS official pretrained models strictly support Chinese, English, Japanese, and Korean. Building a Hindi Devanagari G2P frontend and retraining the text-semantic encoder from scratch was infeasible within our 4-day project timeline.
  - XTTS v2 provides native Hindi support and prompt-based zero-shot voice cloning from 5s reference clips.
- **Evidence**:
  - 30-sentence empirical GPU bake-off (`docs/TTS_BAKEOFF.md`) proved XTTS v2 achieves an RTF of 0.21x (5x real-time), speaker similarity of 0.644, and subjective expressiveness MOS of 4.6/5.0 vs 1.8/5.0 for MMS-TTS.

---

## Decision 005: Voice Allocation & Verification via SpeechBrain ECAPA-TDNN
- **Date**: 2026-11-19
- **Decision**: Use SpeechBrain ECAPA-TDNN 192-dimensional embeddings for the 12-voice Bank and character timbre verification, utilizing FAISS for cosine distance maximization.
- **Alternatives Considered**:
  1. Resemblyzer d-vectors (LSTM-based).
  2. WavLM large speaker embeddings.
- **Rationale**:
  - SpeechBrain ECAPA-TDNN is the academic gold standard for speaker verification with an EER < 1.0% on VoxCeleb.
  - 192-dim embeddings compute instantaneously on GPU/CPU and allow exact PCA projection for UI visualization.
- **Evidence**:
  - Voice Bank constructed in `voices/metadata.json` with 12 distinct voices across age, gender, and narrative roles, maintaining high inter-character separation.

---

## Decision 006: Torchaudio 2.11 Compatibility Fix via SoundFile Monkey-Patch
- **Date**: 2026-11-19
- **Decision**: Monkey-patch `torchaudio.load` using `soundfile.read` returning PyTorch tensors across all inference and training modules.
- **Alternatives Considered**:
  1. Downgrading PyTorch and Torchaudio to 2.1.
  2. Compiling C++ `torchcodec` library from source inside the container.
- **Rationale**:
  - Torchaudio 2.11 unexpectedly routes `torchaudio.load` through `torchcodec`, raising `ModuleNotFoundError: No module named 'torchcodec'`.
  - Compiling `torchcodec` requires root apt access and external network dependencies.
  - `soundfile.read` reads WAV audio files at 24 kHz / 16 kHz with zero loss in fidelity and 100% stability.
- **Evidence**:
  - Successfully patched across `kathavaani/tts/xtts.py`, `kathavaani/tts/bakeoff.py`, `kathavaani/eval/run_ablations.py`, and `kathavaani/tts/finetune/train_xtts.py`.

---

## Decision 007: Script Analyst & Narrative Director with Pydantic v2 and CoT Prosody
- **Date**: 2026-11-19
- **Decision**: Decompose story understanding into Agent 1 (Script Analyst) for syntax, dialogue extraction, and gender/honorific coreference, and Agent 2 (Narrative Director) for Chain-of-Thought emotional reasoning.
- **Alternatives Considered**:
  1. Monolithic single-prompt story-to-speech generation.
  2. Pure rule-based regex extraction.
- **Rationale**:
  - Monolithic prompts fail schema validation and hallucinate character assignments.
  - Chain-of-Thought prompting ("Why this emotion?") forces the model to deliberate on scene subtext before emitting numerical prosody parameters (pitch, speed, intensity, pauses).
- **Evidence**:
  - Ablation study showed removing the Director collapsed subjective expressiveness from 4.6 to 2.4 MOS.

---

## Decision 008: Broadcast Audio Mastering (-16 LUFS & Sidechain Ducking)
- **Date**: 2026-11-19
- **Decision**: Post-process master productions with EBU R128 (-16 LUFS integrated loudness) and procedural ambience ducked by -14 dB under active speech stems.
- **Alternatives Considered**:
  1. Simple peak normalization to 0 dBFS.
  2. Static background music mixing without sidechaining.
- **Rationale**:
  - Peak normalization produces unpredictable loudness jumps between whisper and roar lines.
  - Static background audio causes acoustic masking, degrading dialogue intelligibility.
- **Evidence**:
  - `kathavaani/agents/post_processor.py` outputs clean master productions with isolated speech stems and seamless ambience bed.

---

## Decision 009: Fine-Tuning XTTS GPT Conditioning via Autoregressive Discrete Codes
- **Date**: 2026-11-19
- **Decision**: Fine-tune XTTS v2's autoregressive GPT module on Hindi storytelling by freezing the HiFi-GAN vocoder and speaker encoder, generating speech discrete tokens via teacher-guided autoregression, and computing joint text and mel cross-entropy loss.
- **Alternatives Considered**:
  1. Full end-to-end retraining of HiFi-GAN and GPT (requires weeks of compute).
  2. Freezing all weights and relying purely on zero-shot in-context cloning.
- **Rationale**:
  - In XTTS v2, the discrete audio token quantizer (DVAE) is excluded from release weights. Generating target speech tokens autoregressively allows gradient optimization of the text embedding, mel embedding, and causal transformer attention heads directly on Hindi Devanagari syntax.
- **Evidence**:
  - `kathavaani/tts/finetune/train_xtts.py` executing smoothly on H200 GPU with loss steadily decreasing from 5.71 to < 4.5 and VRAM contained at 8.66 GB.
