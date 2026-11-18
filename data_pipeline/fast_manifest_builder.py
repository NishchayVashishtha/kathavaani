"""
Fast Segment Manifest Builder for KathaVaani
Generates segmented_manifest.json from existing segmented WAVs using Whisper on GPU.
"""

import os
import re
import json
import time
import logging
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kathavaani.fast_manifest")


def build_manifest(segments_dir="data_storage/segments", output_manifest="data_storage/segmented_manifest.json", max_files=250):
    if not os.path.exists(segments_dir):
        logger.error(f"Segments directory {segments_dir} does not exist.")
        return

    wav_files = sorted([f for f in os.listdir(segments_dir) if f.endswith(".wav")])
    if not wav_files:
        logger.error("No wav files found in segments directory.")
        return

    wav_files = wav_files[:max_files]
    logger.info(f"Building manifest for {len(wav_files)} segmented Hindi utterances...")

    import whisper
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = whisper.load_model("small", device=device)
    logger.info(f"Whisper loaded on {device}.")

    manifest_records = []
    t0 = time.time()

    for idx, fname in enumerate(wav_files):
        fpath = os.path.join(segments_dir, fname)
        try:
            # Parse story_id and utt_id from filename e.g. utt_00002_story_000.wav
            parts = fname.replace(".wav", "").split("_")
            utt_id = f"{parts[0]}_{parts[1]}"
            story_id = f"{parts[2]}_{parts[3]}" if len(parts) >= 4 else "story_000"

            data, sr = sf.read(fpath)
            dur = round(len(data) / sr, 2)

            res = asr.transcribe(fpath, language="hi")
            text = res.get("text", "").strip()

            # Clean Devanagari text
            text = re.sub(r"[a-zA-Z]", "", text)
            text = re.sub(r"[\t\r\f\v]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) > 5:
                manifest_records.append({
                    "id": utt_id,
                    "story_id": story_id,
                    "audio_path": fpath,
                    "duration": dur,
                    "text": text,
                    "start_time": 0.0,
                    "end_time": dur,
                })

            if (idx + 1) % 25 == 0 or idx == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / max(1, elapsed)
                logger.info(f"Processed [{idx+1}/{len(wav_files)}] segments ({rate:.1f} seg/sec) | Kept: {len(manifest_records)}")

        except Exception as e:
            logger.warning(f"Error processing {fname}: {e}")

    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(manifest_records)} utterances to {output_manifest} in {time.time()-t0:.1f}s.")
    return manifest_records


if __name__ == "__main__":
    build_manifest()
