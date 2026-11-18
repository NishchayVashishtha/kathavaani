"""
ModelManager for KathaVaani
Strict VRAM Management & Dynamic Model Swapping for 18 GB GPU Budget.
"""

import os
import gc
import json
import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("kathavaani.models")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ModelManager:
    """
    Manages loading, caching, offloading, and memory profiling of heavy neural models
    to stay strictly within the 18 GB VRAM budget.
    """

    def __init__(self, log_path: str = "logs/vram_profile.json"):
        self.loaded_models: Dict[str, Any] = {}
        self.log_path = log_path
        self.profiles: Dict[str, Dict[str, Any]] = {}
        os.makedirs(os.path.dirname(self.log_path) if os.path.dirname(self.log_path) else ".", exist_ok=True)
        self._load_existing_profiles()

    def _load_existing_profiles(self):
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
            except Exception:
                self.profiles = {}

    def _save_profiles(self):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist VRAM profile: {e}")

    def get_vram_gb(self) -> float:
        if torch is not None and torch.cuda.is_available():
            return round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
        return 0.0

    def get_peak_vram_gb(self) -> float:
        if torch is not None and torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        return 0.0

    def unload(self, name: str):
        """Explicitly offloads a model and clears CUDA cache."""
        if name in self.loaded_models:
            logger.info(f"Offloading model: {name}")
            del self.loaded_models[name]
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info(f"Offloaded {name}. Current VRAM: {self.get_vram_gb()} GB")

    def unload_all(self):
        """Unload all active models."""
        for name in list(self.loaded_models.keys()):
            self.unload(name)

    def register(self, name: str, model: Any):
        """Registers a loaded model."""
        self.loaded_models[name] = model

    def get(self, name: str) -> Optional[Any]:
        return self.loaded_models.get(name)

    @contextmanager
    def profile_vram(self, stage_name: str):
        """Context manager to measure execution time and peak VRAM consumption."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start_time = time.time()
        start_vram = self.get_vram_gb()
        logger.info(f"[{stage_name}] Starting. Initial VRAM: {start_vram} GB")

        try:
            yield
        finally:
            elapsed = round(time.time() - start_time, 2)
            peak_vram = self.get_peak_vram_gb()
            end_vram = self.get_vram_gb()

            profile_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": elapsed,
                "initial_vram_gb": start_vram,
                "peak_vram_gb": peak_vram,
                "final_vram_gb": end_vram,
            }
            self.profiles[stage_name] = profile_entry
            self._save_profiles()
            logger.info(
                f"[{stage_name}] Completed in {elapsed}s | Peak VRAM: {peak_vram} GB | Final VRAM: {end_vram} GB"
            )


# Global singleton manager
model_manager = ModelManager()
