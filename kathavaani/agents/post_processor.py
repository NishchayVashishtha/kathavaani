"""
Agent 4: Audio Post-Processor for KathaVaani
Broadcast Loudness Normalization (-16 LUFS), Silence Trimming, Crossfades,
Scene Ambience & Sidechain Ducking, Stem Export, and timeline.json generation.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

logger = logging.getLogger("kathavaani.agents.post_processor")


class AudioPostProcessor:
    """
    Assembles synthesized audio turns into a cinema-grade audio theatre production.
    """

    TARGET_LUFS = -16.0
    SAMPLE_RATE = 24000

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(self.sample_rate)

    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Normalizes audio track to ITU-R BS.1770-4 -16 LUFS standard."""
        if len(audio) == 0:
            return audio
        try:
            loudness = self.meter.integrated_loudness(audio)
            if np.isneginf(loudness) or np.isnan(loudness):
                return audio
            normalized = pyln.normalize.loudness(audio, loudness, self.TARGET_LUFS)
            # Clip protection
            max_peak = np.max(np.abs(normalized))
            if max_peak > 0.98:
                normalized = normalized / max_peak * 0.98
            return normalized.astype(np.float32)
        except Exception as e:
            logger.warning(f"Loudness normalization failed: {e}. Peak normalizing.")
            peak = np.max(np.abs(audio)) + 1e-9
            return (audio / peak * 0.9).astype(np.float32)

    def trim_silence(self, audio: np.ndarray, threshold_db: float = -45.0) -> np.ndarray:
        """Trims leading and trailing silence below energy threshold."""
        if len(audio) == 0:
            return audio
        threshold = 10 ** (threshold_db / 20.0)
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) == 0:
            return audio
        start_idx = max(0, non_silent[0] - int(self.sample_rate * 0.02))  # Keep 20ms margin
        end_idx = min(len(audio), non_silent[-1] + int(self.sample_rate * 0.02))
        return audio[start_idx:end_idx]

    def generate_ambience_bed(self, mood: str, duration_sec: float) -> np.ndarray:
        """
        Generates or synthesizes subtle scene-aware ambient background bed.
        """
        total_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, total_samples)

        # Procedural acoustic ambience based on scene mood
        if "mystery" in mood.lower() or "suspense" in mood.lower() or "dark" in mood.lower():
            # Deep atmospheric sub-bass drone with subtle low frequency pulse
            f0 = 55.0  # A1
            drone = 0.05 * np.sin(2 * np.pi * f0 * t) + 0.03 * np.sin(2 * np.pi * (f0 * 1.5) * t)
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t)
            ambience = drone * mod
        elif "festival" in mood.lower() or "joyful" in mood.lower():
            # Warm melodic shimmer
            f0 = 130.81  # C3
            drone = 0.04 * np.sin(2 * np.pi * f0 * t) + 0.03 * np.sin(2 * np.pi * (f0 * 1.25) * t)
            ambience = drone
        else:
            # Gentle pastoral nature drone
            f0 = 110.0  # A2
            drone = 0.03 * np.sin(2 * np.pi * f0 * t) + 0.02 * np.sin(2 * np.pi * (f0 * 1.33) * t)
            ambience = drone

        return ambience.astype(np.float32)

    def apply_sidechain_ducking(
        self,
        bgm: np.ndarray,
        speech: np.ndarray,
        ducking_db: float = -14.0,
        attack_ms: int = 50,
        release_ms: int = 300,
    ) -> np.ndarray:
        """
        Ducks background music level when speech energy is present.
        """
        length = min(len(bgm), len(speech))
        bgm = bgm[:length]
        speech = speech[:length]

        # Calculate smooth speech envelope
        envelope = np.abs(speech)
        window = int(self.sample_rate * 0.05)  # 50ms window
        if window > 0:
            kernel = np.ones(window) / window
            envelope = np.convolve(envelope, kernel, mode="same")

        # Create ducking gain curve (1.0 = normal, duck_factor = during speech)
        duck_factor = 10 ** (ducking_db / 20.0)  # e.g. -14 dB ~ 0.20
        speech_active = (envelope > 0.02).astype(float)

        # Smooth speech activity
        smooth_active = np.convolve(speech_active, np.hanning(int(self.sample_rate * 0.1)), mode="same")
        smooth_active = np.clip(smooth_active, 0.0, 1.0)

        gain_curve = 1.0 - smooth_active * (1.0 - duck_factor)
        return (bgm * gain_curve).astype(np.float32)

    def assemble_production(
        self,
        synthesized_turns: List[Dict],
        director_styles: List[Dict],
        output_dir: str = "outputs/master",
        enable_bgm: bool = True,
        bgm_volume: float = 0.15,
    ) -> Dict:
        """
        Assembles all per-turn audio stems into a final master track and timeline.json.
        """
        os.makedirs(output_dir, exist_ok=True)
        timeline_entries = []
        master_buffer = []

        current_time_ms = 0

        # Track per-character stems
        character_buffers: Dict[str, List[Tuple[int, int, np.ndarray]]] = {}

        for idx, turn in enumerate(synthesized_turns):
            wav_path = turn["audio_path"]
            speaker_id = turn["speaker_id"]
            speaker_name = turn["speaker_name"]
            style = director_styles[idx] if idx < len(director_styles) else {}

            pause_before_ms = style.get("pause_before_ms", 300)
            pause_after_ms = style.get("pause_after_ms", 200)

            # Insert pause before
            if pause_before_ms > 0:
                silence_samples = int((pause_before_ms / 1000.0) * self.sample_rate)
                master_buffer.append(np.zeros(silence_samples, dtype=np.float32))
                current_time_ms += pause_before_ms

            # Load and process speech turn
            data, sr = sf.read(wav_path)
            if sr != self.sample_rate:
                import scipy.signal
                num_s = int(len(data) * self.sample_rate / sr)
                data = scipy.signal.resample(data, num_s)

            data = self.trim_silence(data)
            data = self.normalize_loudness(data)

            # Crossfade 20ms envelope to avoid edge pop
            fade_len = int(self.sample_rate * 0.015)
            if len(data) > 2 * fade_len:
                data[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
                data[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

            start_turn_ms = current_time_ms
            turn_duration_ms = int((len(data) / self.sample_rate) * 1000.0)
            end_turn_ms = start_turn_ms + turn_duration_ms

            master_buffer.append(data)
            current_time_ms = end_turn_ms

            # Log to timeline
            timeline_entries.append({
                "idx": idx,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "text": turn["text"],
                "start_ms": start_turn_ms,
                "end_ms": end_turn_ms,
                "duration_ms": turn_duration_ms,
                "emotion": style.get("emotion", "neutral"),
                "intensity": style.get("intensity", 0.5),
                "pitch": style.get("pitch", "medium"),
                "speed": style.get("speed", "medium"),
                "reasoning": style.get("reasoning", ""),
            })

            # Record in character stems
            if speaker_id not in character_buffers:
                character_buffers[speaker_id] = []
            character_buffers[speaker_id].append((start_turn_ms, end_turn_ms, data))

            # Insert pause after
            if pause_after_ms > 0:
                silence_samples = int((pause_after_ms / 1000.0) * self.sample_rate)
                master_buffer.append(np.zeros(silence_samples, dtype=np.float32))
                current_time_ms += pause_after_ms

        # Concatenate speech master
        speech_master = np.concatenate(master_buffer) if master_buffer else np.zeros(self.sample_rate)
        total_duration_sec = len(speech_master) / self.sample_rate

        # Mix Background Music & Sidechain Ducking
        final_master = speech_master.copy()
        bgm_track = None

        if enable_bgm and total_duration_sec > 0:
            predominant_mood = director_styles[0].get("scene_mood", "peaceful") if director_styles else "peaceful"
            raw_bgm = self.generate_ambience_bed(predominant_mood, total_duration_sec)
            ducked_bgm = self.apply_sidechain_ducking(raw_bgm, speech_master, ducking_db=-14.0)
            ducked_bgm = ducked_bgm * bgm_volume
            final_master = final_master + ducked_bgm
            bgm_track = ducked_bgm

        # Normalize final master
        final_master = self.normalize_loudness(final_master)

        # Export Master WAV & MP3
        master_wav_path = os.path.join(output_dir, "master.wav")
        master_mp3_path = os.path.join(output_dir, "master.mp3")
        sf.write(master_wav_path, final_master, self.sample_rate)

        # Use static ffmpeg to export clean MP3
        os.system(f"ffmpeg -y -i '{master_wav_path}' -codec:a libmp3lame -qscale:a 2 '{master_mp3_path}' >/dev/null 2>&1")

        # Export Character Stems
        stems_dir = os.path.join(output_dir, "stems")
        os.makedirs(stems_dir, exist_ok=True)
        total_samples = len(final_master)

        for spk_id, entries in character_buffers.items():
            stem_track = np.zeros(total_samples, dtype=np.float32)
            for s_ms, e_ms, audio_data in entries:
                s_idx = int((s_ms / 1000.0) * self.sample_rate)
                e_idx = min(s_idx + len(audio_data), total_samples)
                stem_track[s_idx:e_idx] = audio_data[: e_idx - s_idx]
            stem_wav = os.path.join(stems_dir, f"{spk_id}.wav")
            sf.write(stem_wav, stem_track, self.sample_rate)

        if bgm_track is not None:
            sf.write(os.path.join(stems_dir, "bgm.wav"), bgm_track, self.sample_rate)

        # Export timeline.json
        timeline_meta = {
            "total_duration_ms": current_time_ms,
            "total_turns": len(timeline_entries),
            "sample_rate": self.sample_rate,
            "master_wav": master_wav_path,
            "master_mp3": master_mp3_path,
            "turns": timeline_entries,
        }
        timeline_path = os.path.join(output_dir, "timeline.json")
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(timeline_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Production successfully exported to {output_dir} (Total: {current_time_ms / 1000:.1f}s)")
        return timeline_meta


audio_post_processor = AudioPostProcessor()
