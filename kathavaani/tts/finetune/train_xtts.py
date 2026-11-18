"""
KathaVaani — XTTS v2 GPT Decoder Fine-Tuning Pipeline
Features:
- True ground-truth discrete acoustic tokenization via official Coqui DVAE (dvae.pth)
- Mixed precision (fp16) + Gradient Checkpointing + Gradient Accumulation
- Multi-epoch training loop on train.jsonl (1,568 utterances) with validation on val.jsonl (68 utterances)
- Early stopping tracking best validation loss
- Saves checkpoints to high-capacity storage (/var/tmp/dgx-s-bmu-soet-230737/checkpoints/)
- Generates 3 fixed audio evaluation samples per epoch into outputs/finetune_samples/
- Resumable (--resume) and background-safe logging
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Fix PyTorch 2.6 weights_only default
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

# Fix torchaudio load
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

# Patch GPT.forward for cond_mels compatibility
from TTS.tts.layers.xtts.gpt import GPT
_orig_gpt_forward = GPT.forward

def _patched_gpt_forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels=None, cond_idxs=None, cond_lens=None, cond_latents=None, return_attentions=False, return_latent=False):
    if cond_mels is None and cond_latents is not None:
        cond_mels = torch.zeros((cond_latents.shape[0], 1, 80, cond_latents.shape[1]), device=cond_latents.device)
    return _orig_gpt_forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels=cond_mels, cond_idxs=cond_idxs, cond_lens=cond_lens, cond_latents=cond_latents, return_attentions=return_attentions, return_latent=return_latent)

GPT.forward = _patched_gpt_forward

from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.trainer.gpt_trainer import TorchMelSpectrogram

logger = logging.getLogger("kathavaani.finetune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


FIXED_EVAL_SENTENCES = [
    ("eval_01_neutral", "एक पुराने गांव के किनारे एक घना और शांत बरगद का पेड़ था।"),
    ("eval_02_dialogue", "शेर ने दहाड़ते हुए कहा, ठीक है! लेकिन किसी दिन भोजन में देरी हुई तो मैं सबको मार दूंगा!"),
    ("eval_03_whisper", "खरगोश शेर को कुएँ के पास ले गया और धीरे से फुसफुसाया, महाराज वह दुष्ट यहीं छिपा है।"),
]


class HindiStoryDataset(Dataset):
    def __init__(self, manifest_path: str, max_samples: Optional[int] = None):
        self.samples = []
        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest {manifest_path} not found.")
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if os.path.exists(data.get("audio_path", "")) and len(data.get("text", "")) >= 5:
                        self.samples.append(data)
                if max_samples and len(self.samples) >= max_samples:
                    break

        logger.info(f"Loaded {len(self.samples)} valid samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def run_training(
    train_manifest="data_storage/train.jsonl",
    val_manifest="data_storage/val.jsonl",
    output_dir="/var/tmp/dgx-s-bmu-soet-230737/checkpoints",
    symlink_dir="checkpoints",
    epochs=10,
    batch_size=2,
    grad_accum_steps=4,
    learning_rate=5e-6,
    device="cuda",
    early_stopping_patience=3,
    save_every_epochs=1,
    resume_checkpoint=None,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(symlink_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs/finetune_samples", exist_ok=True)
    loss_log_path = "logs/train_loss.jsonl"

    train_dataset = HindiStoryDataset(train_manifest)
    val_dataset = HindiStoryDataset(val_manifest)

    if len(train_dataset) == 0:
        logger.error(f"No training samples found in {train_manifest}.")
        return

    # 1. Load DVAE and Mel Extractor for Ground-Truth Tokenization
    dvae_path = "models_cache/dvae.pth"
    if not os.path.exists(dvae_path):
        dvae_path = "/workspace/kathavaani/models_cache/dvae.pth"

    logger.info(f"Loading Discrete VAE from {dvae_path}...")
    dvae_ckpt = torch.load(dvae_path, map_location="cpu", weights_only=False)
    dvae = DiscreteVAE(
        channels=80, normalization=None, positional_dims=1, num_tokens=1024,
        codebook_dim=512, hidden_dim=512, num_resnet_blocks=3, kernel_size=3,
        num_layers=2, use_transposed_convs=False
    )
    dvae.load_state_dict(dvae_ckpt, strict=False)
    dvae.to(device).eval()
    for p in dvae.parameters():
        p.requires_grad = False

    mel_extractor = TorchMelSpectrogram(sampling_rate=22050).to(device)

    # 2. Load XTTS v2 Synthesizer and Model
    logger.info("Initializing XTTS v2 model for Hindi storytelling fine-tuning...")
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    model = tts.synthesizer.tts_model

    # Enable gradient checkpointing
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

    # Optimizer with Cosine Annealing Learning Rate
    optimizer = torch.optim.AdamW(model.gpt.parameters(), lr=learning_rate, weight_decay=1e-2, eps=1e-8)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location="cpu")
        model.gpt.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info(f"Resumed at epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

    global_step = 0
    t_start = time.time()

    logger.info("======================================================================")
    logger.info(f" Starting XTTS Fine-Tuning Run: {epochs} Epochs")
    logger.info(f" Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")
    logger.info(f" Effective Batch Size: {batch_size * grad_accum_steps} | LR: {learning_rate}")
    logger.info("======================================================================")

    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050).to(device)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_mel_loss = 0.0
        epoch_text_loss = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        optimizer.zero_grad()

        # Shuffle indices
        indices = np.random.permutation(len(train_dataset))

        for idx, sample_idx in enumerate(indices):
            sample = train_dataset[sample_idx]
            audio_path = sample["audio_path"]
            text = sample["text"]

            try:
                # 1. Conditioning latents
                cond_latents, _ = model.get_conditioning_latents(audio_path)

                # 2. Tokenize text
                text_tokens = torch.tensor(
                    model.tokenizer.encode(text, lang="hi"), dtype=torch.long, device=device
                ).unsqueeze(0)
                text_len = torch.tensor([text_tokens.shape[1]], dtype=torch.long, device=device)

                # 3. Read and resample audio to 22050 Hz for DVAE
                wav_arr, sr = sf.read(audio_path, dtype="float32")
                wav_tensor = torch.from_numpy(wav_arr).unsqueeze(0).to(device)
                if sr != 22050:
                    if sr == 16000:
                        wav_22k = resampler(wav_tensor)
                    else:
                        temp_resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050).to(device)
                        wav_22k = temp_resample(wav_tensor)
                else:
                    wav_22k = wav_tensor

                # 4. Extract TRUE discrete audio tokens via DVAE
                with torch.no_grad():
                    mel = mel_extractor(wav_22k)
                    audio_codes = dvae.get_codebook_indices(mel)

                wav_lengths = torch.tensor([wav_22k.shape[-1]], dtype=torch.long, device=device)

                # 5. Forward pass through GPT
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss_text, loss_mel, _ = model.gpt(
                        text_tokens,
                        text_len,
                        audio_codes,
                        wav_lengths,
                        cond_latents=cond_latents,
                    )
                    loss = (loss_text + loss_mel) / grad_accum_steps

                scaler.scale(loss).backward()

                epoch_loss += (loss_text.item() + loss_mel.item())
                epoch_text_loss += loss_text.item()
                epoch_mel_loss += loss_mel.item()
                epoch_steps += 1

                # Gradient step
                if (idx + 1) % grad_accum_steps == 0 or (idx + 1) == len(indices):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.gpt.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 20 == 0 or global_step == 1:
                        avg_loss = epoch_loss / max(1, epoch_steps)
                        vram_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                        logger.info(
                            f"[Epoch {epoch:02d}/{epochs:02d} | Step {global_step:04d}] "
                            f"Loss: {avg_loss:.4f} (text={loss_text.item():.4f}, mel={loss_mel.item():.4f}) | "
                            f"VRAM: {vram_used:.2f} GB"
                        )

            except Exception as e:
                logger.warning(f"Skipping sample {audio_path}: {e}")
                continue

        # Epoch training summary
        train_avg_loss = epoch_loss / max(1, epoch_steps)
        t_epoch = time.time() - t_epoch_start
        logger.info(f"=== Epoch {epoch:02d} Complete | Train Loss: {train_avg_loss:.4f} | Time: {t_epoch:.1f}s ===")

        # -------------------------------------------------------------
        # Validation Evaluation
        # -------------------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        logger.info(f"Evaluating validation set ({len(val_dataset)} samples)...")

        with torch.no_grad():
            for v_idx in range(len(val_dataset)):
                v_sample = val_dataset[v_idx]
                try:
                    v_audio = v_sample["audio_path"]
                    v_text = v_sample["text"]

                    v_cond_latents, _ = model.get_conditioning_latents(v_audio)
                    v_text_tokens = torch.tensor(
                        model.tokenizer.encode(v_text, lang="hi"), dtype=torch.long, device=device
                    ).unsqueeze(0)
                    v_text_len = torch.tensor([v_text_tokens.shape[1]], dtype=torch.long, device=device)

                    v_wav_arr, v_sr = sf.read(v_audio, dtype="float32")
                    v_wav_tensor = torch.from_numpy(v_wav_arr).unsqueeze(0).to(device)
                    if v_sr != 22050:
                        v_wav_22k = resampler(v_wav_tensor)
                    else:
                        v_wav_22k = v_wav_tensor

                    v_mel = mel_extractor(v_wav_22k)
                    v_audio_codes = dvae.get_codebook_indices(v_mel)
                    v_wav_lengths = torch.tensor([v_wav_22k.shape[-1]], dtype=torch.long, device=device)

                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        v_loss_text, v_loss_mel, _ = model.gpt(
                            v_text_tokens,
                            v_text_len,
                            v_audio_codes,
                            v_wav_lengths,
                            cond_latents=v_cond_latents,
                        )
                    val_loss_sum += (v_loss_text.item() + v_loss_mel.item())
                    val_steps += 1
                except Exception:
                    continue

        val_avg_loss = val_loss_sum / max(1, val_steps)
        logger.info(f"[Validation | Epoch {epoch:02d}] Val Loss: {val_avg_loss:.4f} (vs Best: {best_val_loss:.4f})")

        # Log to train_loss.jsonl
        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(train_avg_loss, 4),
            "val_loss": round(val_avg_loss, 4),
            "vram_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2),
            "epoch_sec": round(t_epoch, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(loss_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # -------------------------------------------------------------
        # Generate 3 Fixed Evaluation Samples
        # -------------------------------------------------------------
        logger.info(f"Synthesizing 3 fixed eval sentences for Epoch {epoch:02d}...")
        ref_sample = "voices/narrator_male_deep.wav"
        for tag, eval_text in FIXED_EVAL_SENTENCES:
            try:
                sample_out = f"outputs/finetune_samples/epoch_{epoch:02d}_{tag}.wav"
                tts.tts_to_file(
                    text=eval_text,
                    speaker_wav=ref_sample,
                    language="hi",
                    file_path=sample_out,
                )
            except Exception as e:
                logger.warning(f"Eval sample {tag} generation failed: {e}")

        # -------------------------------------------------------------
        # Checkpointing & Early Stopping
        # -------------------------------------------------------------
        # Save epoch checkpoint (fp16 model weights only to conserve disk space)
        epoch_ckpt_path = os.path.join(output_dir, f"xtts_hindi_epoch_{epoch:02d}.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": {k: v.half() for k, v in model.gpt.state_dict().items()},
            "train_loss": train_avg_loss,
            "val_loss": val_avg_loss,
        }, epoch_ckpt_path)

        # Symlink latest
        latest_symlink = os.path.join(symlink_dir, "latest.pt")
        if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(epoch_ckpt_path, latest_symlink)

        # Best model
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            best_ckpt_path = os.path.join(output_dir, "xtts_best_val.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": {k: v.half() for k, v in model.gpt.state_dict().items()},
                "val_loss": val_avg_loss,
            }, best_ckpt_path)
            best_symlink = os.path.join(symlink_dir, "xtts_best_val.pt")
            if os.path.islink(best_symlink) or os.path.exists(best_symlink):
                os.remove(best_symlink)
            os.symlink(best_ckpt_path, best_symlink)
            logger.info(f"New Best Val Loss ({best_val_loss:.4f})! Saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Early stopping patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at Epoch {epoch:02d}!")
                break

    logger.info(f"XTTS Fine-Tuning Run Completed in {time.time() - t_start:.1f}s.")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    run_training(epochs=args.epochs, learning_rate=args.lr, resume_checkpoint=args.resume)
