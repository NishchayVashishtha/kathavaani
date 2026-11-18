"""
KathaVaani CLI Entrypoint
Usage:
  python -m kathavaani.cli story.txt --out outputs/my_story/
"""

import os
import sys
import argparse
import logging
from kathavaani.orchestrator.graph import KathaVaaniOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kathavaani.cli")


def main():
    parser = argparse.ArgumentParser(
        description="KathaVaani (कथावाणी): Context-Aware Expressive Multi-Speaker TTS for Hindi Storytelling"
    )
    parser.add_argument("story_path", type=str, help="Path to Hindi story text file (Devanagari, UTF-8)")
    parser.add_argument("--out", "-o", type=str, default="outputs/production", help="Output directory for audio and timeline")
    parser.add_argument("--engine", choices=["xtts", "mms"], default="xtts", help="TTS Engine (default: xtts)")
    parser.add_argument("--no-bgm", action="store_true", help="Disable background music bed")
    parser.add_argument("--bgm-volume", type=float, default=0.15, help="BGM volume multiplier (0.0 to 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.story_path):
        print(f"Error: Story file '{args.story_path}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.story_path, "r", encoding="utf-8") as f:
        story_text = f.read().strip()

    if not story_text:
        print("Error: Story file is empty.", file=sys.stderr)
        sys.exit(1)

    print("======================================================================")
    print(" KathaVaani (कथावाणी) — Hindi Storytelling Audio Theatre Producer")
    print(f" Input File: {args.story_path} ({len(story_text)} chars)")
    print(f" Engine:     {args.engine.upper()} | Output: {args.out}")
    print("======================================================================")

    engine = None
    if args.engine == "mms":
        from kathavaani.tts.mms import MMSEngine
        engine = MMSEngine()
    else:
        from kathavaani.tts.xtts import XTTSEngine
        engine = XTTSEngine()

    orchestrator = KathaVaaniOrchestrator(tts_engine=engine)
    res = orchestrator.run(story_text, output_dir=args.out)

    print("\n----------------------------------------------------------------------")
    print(f" Production Succeeded!")
    print(f" Master Audio:   {res.get('master_wav')}")
    print(f" MP3 Audio:      {res.get('master_mp3')}")
    print(f" Timeline:       {os.path.join(args.out, 'timeline.json')}")
    print(f" Total Duration: {res.get('total_duration_ms', 0) / 1000.0:.2f} seconds")
    print("----------------------------------------------------------------------")


if __name__ == "__main__":
    main()
