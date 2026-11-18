"""
Unified TTS Engine Interface for KathaVaani
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import os
import soundfile as sf
import numpy as np


class BaseTTSEngine(ABC):
    """Abstract base class for all TTS engine adapters."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device

    @abstractmethod
    def load(self):
        """Loads model weights onto device."""
        pass

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        style: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        language: str = "hi",
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from Devanagari Hindi text conditioned on speaker reference
        and stylistic parameters.
        Returns:
            (waveform_numpy, sample_rate)
        """
        pass

    @abstractmethod
    def unload(self):
        """Unloads weights from GPU memory."""
        pass
