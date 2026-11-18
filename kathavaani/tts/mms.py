"""
MMS-TTS Hindi Baseline Engine Adapter for KathaVaani
Meta MMS VITS Hindi (facebook/mms-tts-hin) single-speaker baseline.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import soundfile as sf
import torch
from .engine import BaseTTSEngine

logger = logging.getLogger("kathavaani.tts.mms")


class MMSEngine(BaseTTSEngine):
    def __init__(self, model_name: str = "facebook/mms-tts-hin", device: str = "cuda"):
        super().__init__(name="MMS-TTS-Hindi", device=device)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return

        logger.info(f"Loading MMS-TTS Hindi ({self.model_name}) on {self.device}")
        try:
            from transformers import VitsModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = VitsModel.from_pretrained(self.model_name).to(self.device)
            logger.info("MMS-TTS Hindi loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load MMS-TTS: {e}")
            raise e

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        style: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        language: str = "hi",
    ) -> Tuple[np.ndarray, int]:
        if self.model is None:
            self.load()

        logger.info(f"[MMS-TTS] Synthesizing Devanagari text ({len(text)} chars)")
        try:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model(**inputs).waveform

            wav_arr = output.squeeze().cpu().numpy().astype(np.float32)
            sr = self.model.config.sampling_rate

            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                sf.write(output_path, wav_arr, sr)
                logger.info(f"[MMS-TTS] Saved synthesis to {output_path}")

            return wav_arr, sr
        except Exception as e:
            logger.error(f"MMS synthesis error: {e}")
            raise e

    def unload(self):
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
