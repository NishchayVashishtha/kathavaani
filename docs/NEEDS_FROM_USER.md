# KathaVaani — Needs From User (Secrets & Permissions)

To ensure zero friction during execution, optional external services and gated resources are tracked here. KathaVaani functions end-to-end locally using open-weights models and local inference on the H200 GPU, but the following credentials enhance performance if provided:

| Item | Purpose | Criticality | How to Provide |
| :--- | :--- | :--- | :--- |
| **Hugging Face Token** (`HF_TOKEN`) | Access gated models (e.g., `pyannote/speaker-diarization-3.1` or Meta models). | **Optional / Recommended** | Set in terminal: `export HF_TOKEN="hf_..."` or add to `configs/secrets.yaml`. |
| **OpenAI / Gemini API Key** | Optional high-throughput cloud LLM backend for script analysis and emotion reasoning during fast dev cycles. | **Optional** (Local 4-bit/8-bit Qwen2.5/Llama-3 runs on GPU if omitted). | Set `export OPENAI_API_KEY="sk-..."` or `export GEMINI_API_KEY="..."`. |
| **YouTube Cookies** | If `yt-dlp` encounters rate limits when pulling supplementary CC audio. | **Optional** (Google Drive dataset already accessible). | Place `cookies.txt` in repo root. |

*Note: KathaVaani does NOT block waiting for these. Development and evaluation proceed with local models and open data.*
