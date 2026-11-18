"""
XTTS v2 Engine Adapter for KathaVaani
Supports Zero-Shot Cross-Lingual & Native Hindi Voice Cloning.
"""

import os
import re
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import soundfile as sf
import torch

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

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs])
except Exception:
    pass

from .engine import BaseTTSEngine

logger = logging.getLogger("kathavaani.tts.xtts")


def normalize_hindi_eval_text(text: str) -> str:
    """Normalizes Hindi text for phonetic/character accuracy comparison."""
    text = re.sub(r"[।,?!;:\"'—\-\(\)\[\]\{\}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_dist(s1: str, s2: str) -> int:
    """Fast iterative Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dels = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]


def compute_hindi_cer(ref: str, hyp: str) -> float:
    """Computes Character Error Rate (CER) on normalized Devanagari text."""
    ref_norm = normalize_hindi_eval_text(ref)
    hyp_norm = normalize_hindi_eval_text(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    dist = levenshtein_dist(ref_norm, hyp_norm)
    return dist / max(1, len(ref_norm))


class XTTSEngine(BaseTTSEngine):
    def __init__(
        self,
        model_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        enable_asr_gate: bool = True,
    ):
        super().__init__(name="XTTS-v2", device=device)
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.enable_asr_gate = enable_asr_gate
        self.model = None
        self.asr_model = None
        self.last_synthesis_meta: Dict[str, Any] = {}

    def load(self):
        """Loads XTTS v2 model checkpoint onto GPU, optionally loading fine-tuned GPT weights."""
        if self.model is not None:
            return

        logger.info(f"Loading XTTS v2 on device: {self.device}")
        try:
            from TTS.api import TTS
            model_name = self.model_path or "tts_models/multilingual/multi-dataset/xtts_v2"
            self.model = TTS(model_name).to(self.device)
            logger.info("Base XTTS v2 successfully loaded.")

            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                logger.info(f"Loading fine-tuned GPT weights from {self.checkpoint_path}...")
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.model.synthesizer.tts_model.gpt.load_state_dict(state_dict)
                logger.info(f"Fine-tuned weights ({len(state_dict)} params) loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            raise e

    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        style: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        language: str = "hi",
    ) -> Tuple[np.ndarray, int]:
        """
        Hardened XTTS v2 synthesis:
        - Repetition penalty = 10.0-12.0
        - Temperature = 0.55-0.70
        - Top-p = 0.85
        - Proportional duration bounds (max 1.7x expected speech duration)
        - ASR-gated multi-candidate reranking (picks candidate with lowest Whisper CER)
        """
        if self.model is None:
            self.load()

        style = style or {}
        speed = float(style.get("speed_factor", 1.0))
        emotion = style.get("emotion", "neutral")
        intensity = float(style.get("intensity", 0.5))

        # Base hyperparameter configuration
        base_temp = 0.65
        base_rep = 10.0
        top_p = 0.85

        if emotion in ["angry", "excited", "surprise"]:
            base_temp = 0.70 + 0.05 * intensity
        elif emotion in ["sad", "whisper", "tender"]:
            base_temp = 0.55 - 0.05 * intensity

        if "speed" in style and isinstance(style["speed"], str):
            sp_map = {"slow": 0.85, "medium": 1.0, "fast": 1.15}
            speed = sp_map.get(style["speed"], 1.0)

        # Proportional duration bound: Hindi speech rate is ~11 chars/sec
        expected_sec = max(1.8, len(text) / 11.0)
        max_allowed_sec = max(3.5, 1.75 * expected_sec)

        max_attempts = 3 if self.enable_asr_gate else 1
        candidates = []

        logger.info(f"[XTTS-Hardened] Synthesizing '{text[:30]}...' ({len(text)} chars) | Expected dur: {expected_sec:.1f}s | Gate: {self.enable_asr_gate}")

        try:
            for attempt in range(1, max_attempts + 1):
                cur_temp = base_temp
                cur_rep = base_rep
                if attempt == 2:
                    cur_temp = max(0.50, base_temp - 0.08)
                    cur_rep = 12.0
                elif attempt == 3:
                    cur_temp = min(0.72, base_temp + 0.06)
                    cur_rep = 10.0

                wav = self.model.tts(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language,
                    speed=speed,
                    temperature=cur_temp,
                    repetition_penalty=cur_rep,
                    top_p=top_p,
                )
                wav_arr = np.array(wav, dtype=np.float32)
                dur_sec = len(wav_arr) / 24000.0

                dur_anomaly = (dur_sec > max_allowed_sec) or (dur_sec < 0.6)
                cer = 1.0

                if self.enable_asr_gate:
                    if self.asr_model is None:
                        import whisper
                        self.asr_model = whisper.load_model("base", device=self.device)

                    import scipy.signal
                    wav_16k = scipy.signal.resample(wav_arr, int(len(wav_arr) * 16000 / 24000))
                    asr_res = self.asr_model.transcribe(wav_16k.astype(np.float32), language="hi")
                    hyp_text = asr_res.get("text", "")
                    cer = compute_hindi_cer(text, hyp_text)

                score = cer + (0.5 if dur_anomaly else 0.0)
                candidates.append((score, cer, dur_sec, wav_arr, attempt))

                if not dur_anomaly and cer <= 0.18:
                    break

            # Select best candidate
            candidates.sort(key=lambda c: c[0])
            best_score, best_cer, best_dur, best_wav, attempts_used = candidates[0]

            # Truncate if audio exceeded duration bound
            if best_dur > max_allowed_sec:
                best_wav = best_wav[:int(max_allowed_sec * 24000)]

            sr = 24000
            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                sf.write(output_path, best_wav, sr)

            self.last_synthesis_meta = {
                "attempts": attempts_used,
                "retries": attempts_used - 1,
                "cer": round(best_cer, 4),
                "duration_sec": round(len(best_wav) / float(sr), 2),
                "asr_gated": self.enable_asr_gate,
            }

            logger.info(
                f"[XTTS-Hardened] Saved: attempts={attempts_used}, CER={best_cer:.1%}, "
                f"dur={len(best_wav)/sr:.2f}s -> {output_path or 'memory'}"
            )

            return best_wav, sr

        except Exception as e:
            logger.error(f"XTTS synthesis error: {e}")
            raise e

    def unload(self):
        """Frees XTTS v2 model from GPU memory."""
        if self.model is not None:
            logger.info("Unloading XTTS v2...")
            del self.model
            self.model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
