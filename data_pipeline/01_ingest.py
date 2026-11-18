"""
Stage 1 of KathaVaani Data Pipeline:
Scans downloaded storytelling audio, converts to standard 16 kHz mono WAV,
extracts duration, and creates raw audio manifest.
"""

import os
import glob
import json
import logging
import soundfile as sf
import numpy as np

logger = logging.getLogger("kathavaani.pipeline.ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def ingest_raw_audio(
    input_dir="/var/tmp/dgx-s-bmu-soet-230737/datasets/raw_audio/hindi_tts_data",
    output_dir="data_storage/processed_16k",
    manifest_file="data_storage/raw_manifest.json",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)

    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    logger.info(f"Discovered {len(wav_files)} audio files in {input_dir}")

    manifest = []
    total_duration_sec = 0.0

    for idx, fpath in enumerate(wav_files):
        fname = os.path.basename(fpath)
        base_id = f"story_{idx:03d}"
        target_wav = os.path.join(output_dir, f"{base_id}.wav")

        try:
            data, sr = sf.read(fpath)
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Resample to 16 kHz if necessary
            if sr != 16000:
                import scipy.signal
                num_samples = int(len(data) * 16000 / sr)
                data = scipy.signal.resample(data, num_samples)
                sr = 16000

            # Normalize amplitude
            peak = np.max(np.abs(data)) + 1e-9
            if peak > 0.01:
                data = data / peak * 0.9

            sf.write(target_wav, data.astype(np.float32), 16000)
            duration_sec = len(data) / 16000.0
            total_duration_sec += duration_sec

            manifest.append({
                "story_id": base_id,
                "original_filename": fname,
                "processed_wav": target_wav,
                "duration_seconds": round(duration_sec, 2),
                "sample_rate": 16000,
            })
            logger.info(f"Ingested [{idx+1}/{len(wav_files)}] {fname[:40]}... ({duration_sec/60:.1f} mins)")

        except Exception as e:
            logger.error(f"Failed to ingest {fpath}: {e}")

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Raw Ingestion Complete: {len(manifest)} files | Total Audio: {total_duration_sec/3600:.2f} hours."
    )
    return manifest


if __name__ == "__main__":
    ingest_raw_audio()
