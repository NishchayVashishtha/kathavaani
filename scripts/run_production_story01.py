"""
Runner script for regenerating Story 01 with hardened inference and turn merging.
"""

import os
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kathavaani.prod_story01")

from kathavaani.orchestrator.graph import orchestrator
from kathavaani.eval.qc_story import analyze_story_qc

def main():
    story_path = "stories/01_panchatantra_sher_aur_khargosh.txt"
    output_dir = "outputs/studio/story_01"

    with open(story_path, "r", encoding="utf-8") as f:
        story_text = f.read()

    logger.info(f"=== Starting Hardened Production for Story 01 ({len(story_text)} chars) ===")
    t0 = time.time()
    timeline = orchestrator.run(story_text=story_text, output_dir=output_dir)
    elapsed = time.time() - t0
    logger.info(f"=== Production finished in {elapsed:.1f}s ===")

    logger.info("=== Running Quality Control Audit (qc_story.py) ===")
    analyze_story_qc(
        story_dir=output_dir,
        output_json=os.path.join(output_dir, "qc_report.json"),
        output_md=os.path.join(output_dir, "qc_report.md"),
    )
    logger.info("=== QC Audit Complete! ===")

if __name__ == "__main__":
    main()
