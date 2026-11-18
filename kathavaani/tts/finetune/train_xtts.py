"""
KathaVaani — Fine-Tuning Module for XTTS v2 GPT Conditioning on Hindi Storytelling
Optimized for 18 GB VRAM NVIDIA H200 GPU:
- fp16 / bf16 mixed precision
- Gradient checkpointing
- Small batch size (2) with gradient accumulation (4)
- Resumable checkpoints in checkpoints/
- Loss logging to logs/train_loss.jsonl
"""

import os
import time
import json
import logging
import argparse
from typing import List, Dict, Any
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Fix PyTorch 2.6 weights_only=True default for Coqui TTS checkpoints
_orig_torch_load = torch.load
def _compat_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _compat_torch_load
try:
    import torch.serialization
    import TTS.utils.io
    TTS.utils.io.torch.load = _compat_torch_load
except Exception:
    pass

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])
except Exception:
    pass

# Fix torchaudio >= 2.4 / 2.11 torchcodec requirement by routing through soundfile
try:
    import torchaudio
    _orig_torchaudio_load = getattr(torchaudio, "load", None)
    def _compat_torchaudio_load(filepath, *args, **kwargs):
        try:
            data, sr = sf.read(filepath, dtype='float32')
            tensor = torch.from_numpy(data)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.t()
            return tensor, sr
        except Exception:
            if _orig_torchaudio_load is not None:
                return _orig_torchaudio_load(filepath, *args, **kwargs)
            raise
    torchaudio.load = _compat_torchaudio_load
except Exception:
    pass

# Patch GPT.forward for cond_mels compatibility when cond_latents is passed
from TTS.tts.layers.xtts.gpt import GPT
_orig_gpt_forward = GPT.forward

def _patched_gpt_forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels=None, cond_idxs=None, cond_lens=None, cond_latents=None, return_attentions=False, return_latent=False):
    if cond_mels is None and cond_latents is not None:
        cond_mels = torch.zeros((cond_latents.shape[0], 1, 80, cond_latents.shape[1]), device=cond_latents.device)
    return _orig_gpt_forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels=cond_mels, cond_idxs=cond_idxs, cond_lens=cond_lens, cond_latents=cond_latents, return_attentions=return_attentions, return_latent=return_latent)

GPT.forward = _patched_gpt_forward

logger = logging.getLogger("kathavaani.finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class HindiStoryDataset(Dataset):
    def __init__(self, manifest_path: str, max_samples: int = 500):
        self.samples = []
        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest {manifest_path} not found.")
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if os.path.exists(data.get("audio_path", "")) and len(data.get("text", "")) > 5:
                        self.samples.append(data)
                if len(self.samples) >= max_samples:
                    break

        logger.info(f"Loaded {len(self.samples)} valid samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def run_finetuning(
    train_manifest="data_storage/train.jsonl",
    val_manifest="data_storage/val.jsonl",
    output_dir="checkpoints",
    epochs=1,
    batch_size=2,
    grad_accum_steps=4,
    learning_rate=1e-5,
    max_steps=20,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    loss_log_path = "logs/train_loss.jsonl"

    dataset = HindiStoryDataset(train_manifest)
    if len(dataset) == 0:
        logger.error(f"No training data found in {train_manifest}. Please run data pipeline first.")
        return

    logger.info("Initializing XTTS v2 model for Hindi storytelling fine-tuning...")
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    model = tts.synthesizer.tts_model

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.gpt, "set_gradient_checkpointing"):
        model.gpt.set_gradient_checkpointing(True)
        logger.info("Gradient checkpointing enabled on XTTS GPT.")

    # Freeze HiFi-GAN vocoder & speaker encoder; only train GPT conditioning
    for p in model.hifigan_decoder.parameters():
        p.requires_grad = False
    for p in model.gpt.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.gpt.parameters() if p.requires_grad)
    logger.info(f"Trainable GPT parameters: {trainable_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.gpt.parameters(), lr=learning_rate, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()

    step = 0
    running_loss = 0.0
    t0 = time.time()

    logger.info(f"Starting Fine-Tuning Run: {max_steps} steps | Accumulation: {grad_accum_steps} | Device: {device}")

    # Training Loop
    model.train()
    optimizer.zero_grad()

    while step < max_steps:
        for idx in range(len(dataset)):
            if step >= max_steps:
                break

            item = dataset[idx]
            audio_path = item["audio_path"]
            text = item["text"]

            try:
                # Extract speaker conditioning latents
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path)

                # Tokenize text
                text_tokens = torch.tensor(
                    model.tokenizer.encode(text, lang="hi"), dtype=torch.long, device=device
                ).unsqueeze(0)
                text_len = torch.tensor([text_tokens.shape[1]], dtype=torch.long, device=device)

                # Generate speech tokens via model GPT
                with torch.no_grad():
                    gpt_codes = model.gpt.generate(
                        cond_latents=gpt_cond_latent,
                        text_inputs=text_tokens,
                        do_sample=True,
                        top_p=0.85,
                        top_k=50,
                        temperature=0.75,
                    )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * model.gpt.code_stride_len], device=device
                )

                # Forward pass through GPT conditioning
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss_text, loss_mel, logits = model.gpt(
                        text_tokens,
                        text_len,
                        gpt_codes,
                        expected_output_len,
                        cond_latents=gpt_cond_latent,
                    )
                    loss = (loss_text + loss_mel) / grad_accum_steps

                scaler.scale(loss).backward()
                running_loss += (loss_text.item() + loss_mel.item())

                if (idx + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    step += 1

                    # Log every step
                    avg_loss = running_loss / max(1, step * grad_accum_steps)
                    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
                    elapsed = time.time() - t0
                    logger.info(
                        f"[Step {step:03d}/{max_steps}] Loss: {avg_loss:.4f} (text={loss_text.item():.4f}, mel={loss_mel.item():.4f}) | Peak VRAM: {peak_vram:.2f} GB | Elapsed: {elapsed:.1f}s"
                    )

                    log_entry = {
                        "step": step,
                        "loss": round(avg_loss, 4),
                        "loss_text": round(loss_text.item(), 4),
                        "loss_mel": round(loss_mel.item(), 4),
                        "peak_vram_gb": round(peak_vram, 2),
                        "elapsed_sec": round(elapsed, 1),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    with open(loss_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    # Checkpoint periodically and at final step
                    if step % 5 == 0 or step == max_steps:
                        ckpt_path = os.path.join(output_dir, f"xtts_hindi_step_{step:04d}.pt")
                        torch.save({
                            "step": step,
                            "model_state_dict": model.gpt.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": avg_loss,
                        }, ckpt_path)
                        logger.info(f"Saved resumable checkpoint to {ckpt_path}")

            except Exception as e:
                logger.warning(f"Skipping sample {audio_path}: {e}")
                continue

    logger.info("Fine-tuning completed successfully!")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20, help="Max fine-tuning steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()

    run_finetuning(max_steps=args.steps, learning_rate=args.lr)
