"""
Stage 4 of KathaVaani Data Pipeline:
Transcribes Hindi storytelling audio with Whisper, extracts word-level timestamps,
segments into 3-15s utterances, and applies Devanagari linguistic normalization.
"""

import os
import json
import logging
import soundfile as sf
import numpy as np

logger = logging.getLogger("kathavaani.pipeline.transcribe")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def normalize_devanagari_transcript(text: str) -> str:
    """Devanagari normalization for Hindi TTS training."""
    import re
    # Remove unwanted Latin artifacts or weird symbols
    text = re.sub(r"[a-zA-Z]", "", text)
    text = re.sub(r"[\t\r\f\v]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_and_transcribe(
    manifest_file="data_storage/raw_manifest.json",
    segments_dir="data_storage/segments",
    output_manifest="data_storage/segmented_manifest.json",
    model_size="small",
    max_stories=26,
):
    os.makedirs(segments_dir, exist_ok=True)

    if not os.path.exists(manifest_file):
        logger.error(f"Manifest {manifest_file} not found. Run 01_ingest.py first.")
        return

    with open(manifest_file, "r", encoding="utf-8") as f:
        stories = json.load(f)

    logger.info(f"Loading Whisper model '{model_size}' on GPU...")
    import whisper
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = whisper.load_model(model_size, device=device)
    logger.info(f"Whisper loaded on {device}.")

    all_segments = []
    seg_counter = 0

    for story_idx, item in enumerate(stories[:max_stories]):
        wav_path = item["processed_wav"]
        story_id = item["story_id"]
        logger.info(f"Transcribing Story [{story_idx+1}/{min(len(stories), max_stories)}]: {story_id}")

        try:
            # Transcribe with segment timestamps
            res = asr.transcribe(wav_path, language="hi", task="transcribe")
            raw_segments = res.get("segments", [])

            # Load full audio
            full_audio, sr = sf.read(wav_path)

            for seg in raw_segments:
                start_sec = seg["start"]
                end_sec = seg["end"]
                dur = end_sec - start_sec
                text = normalize_devanagari_transcript(seg["text"])

                # Filter: keep 2.5s to 15s with non-empty Devanagari text
                if 2.5 <= dur <= 15.0 and len(text) > 8:
                    start_sample = int(start_sec * sr)
                    end_sample = min(int(end_sec * sr), len(full_audio))
                    seg_audio = full_audio[start_sample:end_sample]

                    seg_filename = f"utt_{seg_counter:05d}_{story_id}.wav"
                    seg_path = os.path.join(segments_dir, seg_filename)
                    sf.write(seg_path, seg_audio.astype(np.float32), sr)

                    all_segments.append({
                        "id": f"utt_{seg_counter:05d}",
                        "story_id": story_id,
                        "audio_path": seg_path,
                        "duration": round(dur, 2),
                        "text": text,
                        "start_time": round(start_sec, 2),
                        "end_time": round(end_sec, 2),
                    })
                    seg_counter += 1

            logger.info(f"Story {story_id} generated {len(raw_segments)} segments (total kept: {seg_counter})")

            # Checkpoint manifest after each story
            with open(output_manifest, "w", encoding="utf-8") as f:
                json.dump(all_segments, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error transcribing story {story_id}: {e}")

    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)

    logger.info(f"Segmentation Complete: {len(all_segments)} clean Hindi utterances generated.")
    return all_segments


if __name__ == "__main__":
    segment_and_transcribe()
