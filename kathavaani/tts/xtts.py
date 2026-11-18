"""
XTTS v2 Engine Adapter for KathaVaani
Supports Zero-Shot Cross-Lingual & Native Hindi Voice Cloning.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
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


class XTTSEngine(BaseTTSEngine):
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(name="XTTS-v2", device=device)
        self.model_path = model_path
        self.model = None

    def load(self):
        """Loads XTTS v2 model checkpoint onto GPU."""
        if self.model is not None:
            return

        logger.info(f"Loading XTTS v2 on device: {self.device}")
        try:
            from TTS.api import TTS
            model_name = self.model_path or "tts_models/multilingual/multi-dataset/xtts_v2"
            self.model = TTS(model_name).to(self.device)
            logger.info("XTTS v2 successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load official XTTS model: {e}")
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
        Synthesizes Devanagari text using XTTS v2 conditioned on reference speaker audio.
        """
        if self.model is None:
            self.load()

        style = style or {}
        speed = float(style.get("speed_factor", 1.0))
        emotion = style.get("emotion", "neutral")
        intensity = float(style.get("intensity", 0.5))

        # Emotion-aware hyperparameter modulation
        temperature = 0.65
        if emotion in ["angry", "excited", "surprise"]:
            temperature = 0.75 + 0.1 * intensity
        elif emotion in ["sad", "whisper", "tender"]:
            temperature = 0.55 - 0.1 * intensity

        # Map speed string if provided
        if "speed" in style and isinstance(style["speed"], str):
            sp_map = {"slow": 0.85, "medium": 1.0, "fast": 1.15}
            speed = sp_map.get(style["speed"], 1.0)

        logger.info(f"[XTTS-v2] Synthesizing ({len(text)} chars) | Emotion: {emotion} | Speed: {speed}")

        try:
            # XTTS supports language="hi"
            wav = self.model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                speed=speed,
                temperature=temperature,
            )
            wav_arr = np.array(wav, dtype=np.float32)
            sr = 24000  # Standard XTTS sample rate

            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                sf.write(output_path, wav_arr, sr)
                logger.info(f"[XTTS-v2] Saved synthesis to {output_path}")

            return wav_arr, sr

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
