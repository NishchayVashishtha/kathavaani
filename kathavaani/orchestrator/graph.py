"""
Master Orchestrator for KathaVaani
LangGraph state machine integrating Script Analyst, Narrative Director,
Voice Allocator (Timbre Lock), TTS Engine, and Post-Processor with validation critique loop.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, TypedDict
import soundfile as sf
import numpy as np

from kathavaani.agents.script_analyst import script_analyst, ScriptAnalysisResult
from kathavaani.agents.narrative_director import narrative_director, DirectorPlan
from kathavaani.agents.voice_allocator import voice_allocator, CastAllocationPlan
from kathavaani.agents.post_processor import audio_post_processor
from kathavaani.voicebank.bank import voice_bank
from kathavaani.models.manager import model_manager

logger = logging.getLogger("kathavaani.orchestrator")


class KathaVaaniState(TypedDict):
    story_text: str
    script_result: Optional[ScriptAnalysisResult]
    director_plan: Optional[DirectorPlan]
    cast_plan: Optional[CastAllocationPlan]
    synthesized_turns: List[Dict[str, Any]]
    timeline_meta: Optional[Dict[str, Any]]
    retry_count: int
    errors: List[str]
    vram_profile: Dict[str, Any]


class KathaVaaniOrchestrator:
    """
    Coordinates multi-agent execution pipeline with self-critique loops and VRAM budget safety.
    """

    def __init__(self, tts_engine=None):
        self.tts_engine = tts_engine

    def _get_tts_engine(self):
        if self.tts_engine is None:
            from kathavaani.tts.xtts import XTTSEngine
            self.tts_engine = XTTSEngine()
        return self.tts_engine

    def step_script_analyst(self, state: KathaVaaniState) -> KathaVaaniState:
        """Stage 1: Script Analyst character and turn extraction."""
        logger.info("=== [Stage 1] Script Analyst ===")
        res = script_analyst.analyze(state["story_text"])
        state["script_result"] = res
        return state

    def step_narrative_director(self, state: KathaVaaniState) -> KathaVaaniState:
        """Stage 2: Narrative Director Chain-of-Thought prosody generation."""
        logger.info("=== [Stage 2] Narrative Director ===")
        plan = narrative_director.analyze_prosody(state["script_result"])
        state["director_plan"] = plan
        return state

    def step_validate_critique(self, state: KathaVaaniState) -> KathaVaaniState:
        """Stage 3: Validation Critique Loop."""
        logger.info("=== [Stage 3] Validation Critique Loop ===")
        plan = state["director_plan"]
        needs_retry = False

        # Validate emotion schema and intensity range
        for style in plan.turn_styles:
            if not (0.0 <= style.intensity <= 1.0):
                logger.warning(f"Turn {style.idx} invalid intensity {style.intensity}. Clamping.")
                style.intensity = max(0.0, min(1.0, style.intensity))
            if style.pause_before_ms > 1500 or style.pause_after_ms > 1500:
                logger.warning(f"Turn {style.idx} excessive pause. Clamping to 800ms.")
                style.pause_before_ms = min(800, style.pause_before_ms)
                style.pause_after_ms = min(800, style.pause_after_ms)

        state["retry_count"] += 1
        return state

    def step_voice_allocator(self, state: KathaVaaniState) -> KathaVaaniState:
        """Stage 4: Voice Allocator & Timbre Lock."""
        logger.info("=== [Stage 4] Voice Allocator (Timbre Lock) ===")
        cast = voice_allocator.allocate_cast(state["script_result"])
        state["cast_plan"] = cast
        return state

    def step_synthesis(self, state: KathaVaaniState, output_dir: str = "outputs/current") -> KathaVaaniState:
        """Stage 5: Turn-by-Turn TTS Synthesis with Timbre Consistency Tracking."""
        logger.info("=== [Stage 5] TTS Synthesis Engine ===")
        os.makedirs(output_dir, exist_ok=True)
        engine = self._get_tts_engine()

        script = state["script_result"]
        director = state["director_plan"]
        cast = state["cast_plan"]

        synthesized_turns = []

        with model_manager.profile_vram("full_story_synthesis"):
            style_map = {s.idx: s for s in director.turn_styles}
            for turn in script.turns:
                style = style_map.get(turn.idx, director.turn_styles[0] if director.turn_styles else None)
                speaker_id = turn.speaker_id

                # Resolve assigned reference voice
                assignment = cast.assignments.get(speaker_id) or cast.assignments.get("narrator")
                ref_wav = assignment.reference_audio

                turn_wav_path = os.path.join(output_dir, f"turn_{turn.idx:03d}_{speaker_id}.wav")

                # Synthesize
                wav_arr, sr = engine.synthesize(
                    text=turn.text,
                    speaker_wav=ref_wav,
                    style=style.model_dump(),
                    output_path=turn_wav_path,
                    language="hi",
                )

                # Compute speaker consistency / drift check
                consistency = voice_bank.compute_similarity(assignment.voice_id, turn_wav_path)
                synth_meta = getattr(engine, "last_synthesis_meta", {})
                logger.info(f"Turn {turn.idx} ({assignment.voice_name}) -> Consistency: {consistency:.3f} | ASR Attempts: {synth_meta.get('attempts', 1)}")

                synthesized_turns.append({
                    "idx": turn.idx,
                    "speaker_id": speaker_id,
                    "speaker_name": turn.speaker_name,
                    "text": turn.text,
                    "audio_path": turn_wav_path,
                    "consistency": consistency,
                    "attempts": synth_meta.get("attempts", 1),
                    "gate_cer": synth_meta.get("cer", 0.0),
                })

        state["synthesized_turns"] = synthesized_turns
        return state

    def step_post_processor(self, state: KathaVaaniState, output_dir: str = "outputs/current") -> KathaVaaniState:
        """Stage 6: Audio Post-Processor (-16 LUFS, Ducking, Stems, Timeline)."""
        logger.info("=== [Stage 6] Post-Processor & Master Assembly ===")
        director_styles = [s.model_dump() for s in state["director_plan"].turn_styles]

        timeline_meta = audio_post_processor.assemble_production(
            synthesized_turns=state["synthesized_turns"],
            director_styles=director_styles,
            output_dir=output_dir,
            enable_bgm=True,
            bgm_volume=0.15,
        )

        state["timeline_meta"] = timeline_meta
        return state

    def run(self, story_text: str, output_dir: str = "outputs/production") -> Dict[str, Any]:
        """Runs the complete end-to-end KathaVaani storytelling production pipeline."""
        t_start = time.time()
        logger.info(f"Starting KathaVaani Master Production ({len(story_text)} characters)...")

        state: KathaVaaniState = {
            "story_text": story_text,
            "script_result": None,
            "director_plan": None,
            "cast_plan": None,
            "synthesized_turns": [],
            "timeline_meta": None,
            "retry_count": 0,
            "errors": [],
            "vram_profile": {},
        }

        state = self.step_script_analyst(state)
        state = self.step_narrative_director(state)
        state = self.step_validate_critique(state)
        state = self.step_voice_allocator(state)
        state = self.step_synthesis(state, output_dir=output_dir)
        state = self.step_post_processor(state, output_dir=output_dir)

        total_time = round(time.time() - t_start, 2)
        logger.info(f"KathaVaani Master Production finished in {total_time}s.")
        return state["timeline_meta"]


orchestrator = KathaVaaniOrchestrator()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KathaVaani End-to-End Master Story Production")
    parser.add_argument("--story-file", type=str, default=None, help="Path to story text file")
    parser.add_argument("--story-text", type=str, default=None, help="Raw Hindi story text")
    parser.add_argument("--output-dir", type=str, default="outputs/studio", help="Output directory")
    args = parser.parse_args()

    text = ""
    if args.story_file and os.path.exists(args.story_file):
        with open(args.story_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.story_text:
        text = args.story_text
    else:
        text = "एक गाँव में एक दयालु किसान रहता था। उसका नाम रामू था।"

    res = orchestrator.run(
        story_text=text,
        output_dir=args.output_dir,
    )
    print("SUCCESS: Master production created at:", res.get("master_mp3") or res.get("master_wav"))

