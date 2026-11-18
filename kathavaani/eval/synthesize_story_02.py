"""
KathaVaani — Story 02 Master Production Synthesis (Priority 3c)
Synthesizes stories/02_nyay_aur_mamta.txt with >= 4 distinct characters:
- Narrator (Adult Male / Storyteller)
- दादी माँ (Elderly Female / Distressed & Pleading)
- राजा विक्रमादित्य (Adult Male / Powerful & Righteous)
- नन्ही पिंकी (Young Female / Child / Innocent & Gentle)
Exports to outputs/studio/story_02/
"""

import os
import json
import logging
from kathavaani.orchestrator.graph import orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kathavaani.story_02")

def main():
    story_path = "stories/02_nyay_aur_mamta.txt"
    output_dir = "outputs/studio/story_02"
    os.makedirs(output_dir, exist_ok=True)

    with open(story_path, "r", encoding="utf-8") as f:
        story_text = f.read().strip()

    logger.info(f"Loaded story from {story_path} ({len(story_text)} chars). Starting master synthesis...")
    timeline_meta = orchestrator.run(story_text=story_text, output_dir=output_dir)

    logger.info(f"Story 02 synthesis complete! Output saved to {output_dir}")
    logger.info(f"Total Duration: {timeline_meta.get('total_duration_sec', 0.0):.2f}s")
    logger.info(f"Total Turns: {len(timeline_meta.get('turns', []))}")
    logger.info(f"Master files: {timeline_meta.get('master_wav')}, {timeline_meta.get('master_mp3')}")

if __name__ == "__main__":
    main()
