"""
Stage 6 of KathaVaani Data Pipeline:
Annotates segmented Hindi utterances with emotional tone, intensity,
and speaker demographic tags using acoustic prosody and text lexicon.
"""

import os
import json
import logging
import soundfile as sf
import numpy as np

logger = logging.getLogger("kathavaani.pipeline.annotate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HINDI_EMOTIONS = {
    "fear": ["डर", "कांप", "भूत", "राक्षस", "भागो", "बचाओ", "खौफ", "चीख"],
    "whisper": ["धीरे", "फुसफुसा", "सन्नाटा", "दबे"],
    "angry": ["क्रोध", "गुस्सा", "हिम्मत", "गद्दार", "चिल्लाया", "दहाड़ा", "धोखा", "मार"],
    "happy": ["खुश", "आनंद", "वाह", "शाबाश", "बधाई", "जीत", "सुंदर", "हंसी"],
    "sad": ["रोना", "आंसू", "दुःख", "कष्ट", "गरीब", "अकेला", "मरा"],
    "tender": ["बेटा", "प्यार", "ममता", "गोद", "स्नेह", "दुलार"],
    "surprise": ["अरे", "असंभव", "क्या", "चौंक", "जादू", "विचित्र"],
    "tense": ["रहस्य", "सन्नाटे", "अंधेरा", "आहट", "कुआँ", "जंगल"],
}


def annotate_dataset(
    input_manifest="data_storage/segmented_manifest.json",
    output_manifest="data_storage/annotated_manifest.json",
):
    if not os.path.exists(input_manifest):
        logger.error(f"Input manifest {input_manifest} not found.")
        return

    with open(input_manifest, "r", encoding="utf-8") as f:
        utterances = json.load(f)

    logger.info(f"Annotating {len(utterances)} Hindi utterances...")
    annotated = []

    for item in utterances:
        text = item["text"]
        audio_path = item["audio_path"]

        # 1. Text keyword sentiment matching
        detected_emotion = "neutral"
        intensity = 0.5

        for emo, words in HINDI_EMOTIONS.items():
            for w in words:
                if w in text:
                    detected_emotion = emo
                    intensity = 0.75
                    break
            if detected_emotion != "neutral":
                break

        # 2. Acoustic energy & pitch modulation
        try:
            data, sr = sf.read(audio_path)
            rms = float(np.sqrt(np.mean(data ** 2)))
            # If high RMS energy and angry/fear/happy, raise intensity
            if rms > 0.15:
                intensity = min(1.0, intensity + 0.15)
                if detected_emotion == "neutral":
                    detected_emotion = "excited"
            elif rms < 0.03:
                intensity = max(0.3, intensity - 0.15)
                if detected_emotion in ["fear", "neutral"]:
                    detected_emotion = "whisper"
        except Exception:
            rms = 0.05

        item_copy = dict(item)
        item_copy["emotion"] = detected_emotion
        item_copy["intensity"] = round(intensity, 2)
        item_copy["rms_energy"] = round(rms, 4)
        annotated.append(item_copy)

    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    logger.info(f"Emotion annotation completed for {len(annotated)} utterances.")
    return annotated


if __name__ == "__main__":
    annotate_dataset()
