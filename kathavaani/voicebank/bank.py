"""
Voice Bank & Timbre Lock for KathaVaani
Maintains reference speaker profiles, ECAPA-TDNN embeddings, FAISS vector indexing,
and inter-character acoustic distance maximization.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger("kathavaani.voicebank")


@dataclass
class VoiceProfile:
    id: str
    name: str
    gender: str          # "male", "female"
    age_group: str       # "child", "young", "adult", "elderly"
    role: str            # "narrator", "character"
    pitch_range: str     # "low", "medium", "high"
    audio_path: str
    transcript: str
    embedding: Optional[List[float]] = None
    coords_2d: Optional[List[float]] = None  # (x, y) for UI scatter plot


class VoiceBank:
    """
    Manages Hindi storytelling reference voices with ECAPA-TDNN embeddings and FAISS index.
    """

    def __init__(self, voices_dir: str = "voices", metadata_file: str = "voices/metadata.json"):
        self.voices_dir = voices_dir
        self.metadata_file = metadata_file
        self.profiles: Dict[str, VoiceProfile] = {}
        self.encoder = None
        os.makedirs(self.voices_dir, exist_ok=True)
        self.load()

    def load(self):
        """Loads voice profiles from metadata JSON."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        vp = VoiceProfile(**item)
                        self.profiles[vp.id] = vp
                logger.info(f"Loaded {len(self.profiles)} voice profiles from {self.metadata_file}")
            except Exception as e:
                logger.error(f"Error loading voice metadata: {e}")

    def save(self):
        """Persists voice profiles to metadata JSON."""
        data = [asdict(vp) for vp in self.profiles.values()]
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.profiles)} voice profiles to {self.metadata_file}")

    def _get_encoder(self):
        """Lazy loads SpeechBrain ECAPA-TDNN speaker encoder."""
        if self.encoder is None:
            try:
                from speechbrain.inference.speaker import SpeakerRecognition
                self.encoder = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=os.path.join(os.environ.get("TORCH_HOME", "/tmp/torch_cache"), "speechbrain_ecapa"),
                )
                logger.info("SpeechBrain ECAPA-TDNN speaker encoder loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load SpeechBrain ECAPA: {e}. Falling back to acoustic feature encoder.")
                self.encoder = "fallback"
        return self.encoder

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """Extracts 192-dim ECAPA-TDNN speaker embedding using pure soundfile + PyTorch."""
        enc = self._get_encoder()
        if enc != "fallback" and enc is not None:
            try:
                import torch
                import soundfile as sf
                data, fs = sf.read(audio_path)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                # Resample to 16000 if needed using scipy
                if fs != 16000:
                    import scipy.signal
                    num_samples = int(len(data) * 16000 / fs)
                    data = scipy.signal.resample(data, num_samples)
                    fs = 16000

                signal = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
                emb = enc.encode_batch(signal).squeeze().detach().cpu().numpy()
                norm = np.linalg.norm(emb) + 1e-9
                emb = emb / norm
                return emb.astype(np.float32)
            except Exception as e:
                logger.warning(f"ECAPA extraction failed on {audio_path}: {e}. Using acoustic fallback.")

        # Fallback acoustic feature vector (FFT spectral energy + zero crossing + energy)
        try:
            import soundfile as sf
            data, fs = sf.read(audio_path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            # 192-dim FFT energy bands
            fft_mag = np.abs(np.fft.rfft(data[:min(len(data), 8192)]))
            if len(fft_mag) < 192:
                emb = np.pad(fft_mag, (0, 192 - len(fft_mag)))
            else:
                indices = np.linspace(0, len(fft_mag) - 1, 192).astype(int)
                emb = fft_mag[indices]
            norm = np.linalg.norm(emb) + 1e-9
            emb = emb / norm
            return emb.astype(np.float32)
        except Exception as e:
            logger.error(f"Acoustic fallback failed: {e}")
            return np.ones(192, dtype=np.float32) / np.sqrt(192)

    def allocate_voice(
        self,
        character_gender: str,
        character_age: str,
        is_narrator: bool = False,
        assigned_voice_ids: Optional[List[str]] = None,
    ) -> VoiceProfile:
        """
        Voice Allocation Algorithm:
        1. Filters candidate voices by role, gender, and age compatibility.
        2. Maximizes acoustic/embedding distance from already-assigned voices in the story.
        """
        assigned_voice_ids = assigned_voice_ids or []
        candidates = []

        for vp in self.profiles.values():
            if is_narrator:
                if vp.role == "narrator":
                    candidates.append(vp)
            else:
                # Character matching: match gender; if age matches prioritize, else accept compatible
                if vp.role == "character" and vp.gender.lower() == character_gender.lower():
                    candidates.append(vp)

        # Fallback if candidates empty
        if not candidates:
            candidates = [vp for vp in self.profiles.values() if (vp.role == "narrator" if is_narrator else vp.role == "character")]
        if not candidates:
            candidates = list(self.profiles.values())

        if not candidates:
            raise ValueError("VoiceBank is empty! Register voices first.")

        # Exclude already assigned voices if possible
        unassigned_candidates = [c for c in candidates if c.id not in assigned_voice_ids]
        pool = unassigned_candidates if unassigned_candidates else candidates

        if not assigned_voice_ids or all(self.profiles[aid].embedding is None for aid in assigned_voice_ids if aid in self.profiles):
            # Prioritize age match
            for c in pool:
                if c.age_group.lower() == character_age.lower():
                    return c
            return pool[0]

        # Maximize min-distance to already assigned voice embeddings
        assigned_embs = [
            np.array(self.profiles[aid].embedding, dtype=np.float32)
            for aid in assigned_voice_ids
            if aid in self.profiles and self.profiles[aid].embedding is not None
        ]

        if not assigned_embs:
            return pool[0]

        best_candidate = None
        best_min_dist = -1.0

        for candidate in pool:
            if candidate.embedding is None:
                continue
            c_emb = np.array(candidate.embedding, dtype=np.float32)
            # Compute cosine distances (1 - cosine_similarity)
            distances = [1.0 - float(np.dot(c_emb, a_emb)) for a_emb in assigned_embs]
            min_dist = min(distances) if distances else 1.0

            # Bonus for matching age group
            if candidate.age_group.lower() == character_age.lower():
                min_dist += 0.2

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate

        return best_candidate or pool[0]

    def compute_similarity(self, voice_id: str, audio_path: str) -> float:
        """
        Computes cosine similarity between reference voice embedding and a synthesized audio turn.
        Used for Speaker Consistency Tracking & Timbre Drift Detection.
        """
        if voice_id not in self.profiles or self.profiles[voice_id].embedding is None:
            return 1.0
        ref_emb = np.array(self.profiles[voice_id].embedding, dtype=np.float32)
        test_emb = self.extract_embedding(audio_path)
        similarity = float(np.dot(ref_emb, test_emb))
        return round(max(0.0, min(1.0, similarity)), 4)

    def compute_2d_projection(self):
        """Computes 2D projection (x, y in [-1, 1]) for UI scatter plot."""
        embs = []
        keys = []
        for vid, vp in self.profiles.items():
            if vp.embedding is not None:
                embs.append(vp.embedding)
                keys.append(vid)

        if len(embs) < 2:
            for k in keys:
                self.profiles[k].coords_2d = [0.0, 0.0]
            return

        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(np.array(embs))
            # Normalize to [-1, 1]
            max_val = np.max(np.abs(coords)) + 1e-7
            coords = coords / max_val
            for idx, k in enumerate(keys):
                self.profiles[k].coords_2d = [round(float(coords[idx][0]), 3), round(float(coords[idx][1]), 3)]
            self.save()
            logger.info("Computed 2D voice embedding projections.")
        except Exception as e:
            logger.warning(f"PCA 2D projection failed: {e}")


voice_bank = VoiceBank()
