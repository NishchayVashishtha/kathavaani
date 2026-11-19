"""
FastAPI Server for KathaVaani Studio
Provides REST & WebSocket APIs for Script Analysis, Narrative Direction,
Voice Allocation, Synthesis, Turn-Level Regeneration, and Blind Listening Tests.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from kathavaani.agents.script_analyst import script_analyst, ScriptAnalysisResult
from kathavaani.agents.narrative_director import narrative_director, DirectorPlan
from kathavaani.agents.voice_allocator import voice_allocator, CastAllocationPlan
from kathavaani.agents.post_processor import audio_post_processor
from kathavaani.voicebank.bank import voice_bank
from kathavaani.models.manager import model_manager
from kathavaani.orchestrator.graph import orchestrator

logger = logging.getLogger("kathavaani.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="KathaVaani Studio API",
    description="Context-Aware, Expressive Multi-Speaker Hindi Storytelling TTS",
    version="2.0.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class StoryInput(BaseModel):
    story_text: str
    title: Optional[str] = "मेरी हिंदी कहानी"


class SynthesizeRequest(BaseModel):
    characters: Optional[List[Dict[str, Any]]] = None
    turns: Optional[List[Dict[str, Any]]] = None
    story_text: Optional[str] = None
    ambience: Optional[str] = "village_morning"
    ducking_db: Optional[float] = -14.0


class RegenerateTurnInput(BaseModel):
    turn_idx: int
    text: str
    speaker: Optional[str] = None
    speaker_id: Optional[str] = None
    character_voice_id: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    emotion: Optional[str] = "neutral"
    intensity: Optional[float] = 0.5
    pitch: Optional[str] = "medium"
    speed: Optional[str] = "medium"
    pause_before_ms: Optional[int] = 300
    pause_after_ms: Optional[int] = 200


class ListeningTestSubmission(BaseModel):
    test_id: Optional[str] = None
    evaluator_id: Optional[str] = "Anonymous"
    story_id: Optional[str] = "SENT_01"
    model_a: Optional[str] = "KathaVaani (XTTS v2 + Director)"
    model_b: Optional[str] = "Meta MMS-TTS Baseline"
    user_id: Optional[str] = None
    sample_a_id: Optional[str] = None
    sample_b_id: Optional[str] = None
    naturalness_score: Optional[int] = 4
    expressiveness_score: Optional[int] = 5
    speaker_consistency_score: Optional[int] = 4
    audio_quality_score: Optional[int] = 4
    mos_rating_a: Optional[int] = None
    mos_rating_b: Optional[int] = None
    preference: Optional[str] = "A"
    preferred_sample: Optional[str] = None
    feedback: Optional[str] = ""
    comments: Optional[str] = None


# In-memory session state
SESSION = {
    "script_result": None,
    "director_plan": None,
    "cast_plan": None,
    "synthesized_turns": [],
    "timeline_meta": None,
}


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "gpu": model_manager.get_peak_vram_gb(),
        "loaded_models": list(model_manager.loaded_models.keys()),
    }


@app.get("/api/sample_stories")
def get_sample_stories():
    """Lists available preloaded Hindi stories."""
    stories_dir = "stories"
    samples = []
    if os.path.exists(stories_dir):
        for fname in sorted(os.listdir(stories_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(stories_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                samples.append({
                    "id": fname.replace(".txt", ""),
                    "filename": fname,
                    "title": fname.replace(".txt", "").replace("_", " ").title(),
                    "preview": content[:120] + "...",
                    "full_text": content,
                })
    return {"stories": samples}


@app.post("/api/analyze")
def analyze_script(payload: StoryInput):
    """
    Combined Agent 1 (Script Analyst) + Agent 2 (Director) + Agent 3 (Voice Allocator).
    Returns characters with assigned voices and turns with CoT emotion reasoning.
    """
    try:
        # 1. Script Analysis
        script_res = script_analyst.analyze(payload.story_text)
        SESSION["script_result"] = script_res

        # 2. Narrative Direction
        director_plan = narrative_director.analyze_prosody(script_res)
        SESSION["director_plan"] = director_plan

        # 3. Voice Allocation
        cast_plan = voice_allocator.allocate_cast(script_res)
        SESSION["cast_plan"] = cast_plan

        # Merge results for frontend
        char_list = []
        for c in script_res.characters:
            voice_id = cast_plan.assignments.get(c.id, None)
            assigned_id = voice_id.voice_id if voice_id else "narrator_male_deep"
            vp = voice_bank.profiles.get(assigned_id)
            coords = vp.coords_2d if vp and vp.coords_2d else [0.0, 0.0]

            char_list.append({
                "id": c.id,
                "name": c.name,
                "gender": c.gender,
                "age_group": c.age_group,
                "personality": c.personality,
                "aliases": c.aliases,
                "assigned_voice_id": assigned_id,
                "pca_coords": coords,
            })

        turn_list = []
        for t in script_res.turns:
            dir_turn = next((dt for dt in director_plan.turn_styles if dt.idx == t.idx), None)
            if dir_turn:
                style = {
                    "emotion": dir_turn.emotion,
                    "intensity": dir_turn.intensity,
                    "pitch": dir_turn.pitch,
                    "speed": dir_turn.speed,
                    "volume": dir_turn.volume,
                    "pause_before_ms": dir_turn.pause_before_ms,
                    "pause_after_ms": dir_turn.pause_after_ms,
                    "reasoning": dir_turn.reasoning,
                    "style_prompt": dir_turn.style_prompt,
                }
            else:
                style = {
                    "emotion": "neutral",
                    "intensity": 0.5,
                    "pitch": "medium",
                    "speed": "medium",
                    "volume": "medium",
                    "pause_before_ms": 300,
                    "pause_after_ms": 250,
                    "reasoning": "Standard neutral storytelling cadence.",
                }

            turn_list.append({
                "idx": t.idx,
                "type": t.type,
                "speaker": getattr(t, "speaker_name", getattr(t, "speaker", "Narrator")),
                "speaker_id": getattr(t, "speaker_id", "narrator"),
                "text": t.text,
                "addressed_to": t.addressed_to,
                "style": style,
            })

        return {
            "characters": char_list,
            "turns": turn_list,
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voices")
@app.get("/api/voicebank")
def get_voice_bank():
    """Returns all 12 Voice Bank profiles including 2D coordinates for Cast Board."""
    profiles = []
    for vp in voice_bank.profiles.values():
        profiles.append({
            "id": vp.id,
            "name": vp.name,
            "gender": vp.gender,
            "age_group": vp.age_group,
            "role": vp.role,
            "pitch_range": vp.pitch_range,
            "audio_path": vp.audio_path,
            "transcript": vp.transcript,
            "pca_coords": vp.coords_2d or [0.0, 0.0],
        })
    return {"voices": profiles}


@app.post("/api/synthesize")
@app.post("/api/synthesize_full")
def synthesize_full(payload: SynthesizeRequest):
    """Executes the complete multi-agent production and returns timeline metadata."""
    try:
        story_text = payload.story_text or (
            "\n".join(t["text"] for t in payload.turns) if payload.turns else ""
        )
        if not story_text:
            raise HTTPException(status_code=400, detail="Story text or turns required.")

        import torch
        if not torch.cuda.is_available():
            logger.info("Server running on headnode without direct CUDA. Dispatching synthesis to H200 GPU worker pod...")
            os.makedirs("outputs/studio", exist_ok=True)
            tmp_story_path = "outputs/studio/live_input.txt"
            with open(tmp_story_path, "w", encoding="utf-8") as f:
                f.write(story_text)

            import subprocess
            cmd = [
                "bash", "scripts/run_gpu.sh",
                "python3", "-m", "kathavaani.orchestrator.graph",
                "--story-file", "/workspace/kathavaani/outputs/studio/live_input.txt",
                "--output-dir", "/workspace/kathavaani/outputs/studio",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                logger.error(f"GPU Worker synthesis failed: {proc.stderr}")
                raise RuntimeError(f"GPU synthesis error: {proc.stderr[-300:]}")
        else:
            res = orchestrator.run(
                story_text,
                output_dir="outputs/studio",
                ambience_name=payload.ambience or "village_morning",
                ducking_db=payload.ducking_db or -14.0,
            )
            SESSION["timeline_meta"] = res

        # Build timeline object for frontend
        timeline_path = "outputs/studio/timeline.json"
        timeline_data = {
            "master_audio_url": "/api/audio/outputs/studio/master.mp3",
            "stems": [],
            "total_duration": 0.0,
        }
        if os.path.exists(timeline_path):
            with open(timeline_path, "r", encoding="utf-8") as f:
                raw_tl = json.load(f)
                master_file = raw_tl.get("master_mp3") or raw_tl.get("master_wav") or "outputs/studio/master.mp3"
                clean_mf = master_file.replace("/workspace/kathavaani/", "").lstrip("/")
                timeline_data["master_audio_url"] = f"/api/audio/{clean_mf}"
                timeline_data["total_duration"] = round(raw_tl.get("total_duration_ms", 0) / 1000.0, 1)
                timeline_data["turns"] = raw_tl.get("turns", [])

        return {
            "status": "success",
            "timeline": timeline_data,
        }
    except Exception as e:
        logger.error(f"Full synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/regenerate-turn")
@app.post("/api/regenerate_turn")
def regenerate_single_turn(payload: RegenerateTurnInput):
    """
    Core Wow Feature: Regenerates a single line of dialogue/narration
    when the user edits text, emotion, or timing in the Director's Timeline.
    """
    idx = payload.turn_idx
    output_dir = "outputs/studio"
    speaker = payload.speaker or payload.speaker_id or "Narrator"
    turn_wav_path = os.path.join(output_dir, f"turn_{idx:03d}_{speaker}.wav")

    try:
        ref_voice_id = payload.character_voice_id or "narrator_male_deep"
        ref_wav = f"voices/{ref_voice_id}.wav"
        if not os.path.exists(ref_wav):
            ref_wav = "voices/narrator_male_deep.wav"

        engine = orchestrator._get_tts_engine()
        style = payload.style or {
            "emotion": payload.emotion or "neutral",
            "intensity": payload.intensity or 0.5,
            "speed": payload.speed or "medium",
            "pitch": payload.pitch or "medium",
        }

        wav_arr, sr = engine.synthesize(
            text=payload.text,
            speaker_wav=ref_wav,
            style=style,
            output_path=turn_wav_path,
            language="hi",
        )

        return {
            "status": "success",
            "idx": idx,
            "turn_audio_url": f"/api/audio/{turn_wav_path}",
            "duration_s": round(len(wav_arr) / sr, 2),
        }
    except Exception as e:
        logger.error(f"Turn regeneration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timeline")
def get_timeline():
    """Returns the current story timeline."""
    timeline_path = "outputs/studio/timeline.json"
    if os.path.exists(timeline_path):
        with open(timeline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"stems": []}


@app.get("/api/audio/{filename:path}")
@app.head("/api/audio/{filename:path}")
def serve_audio(filename: str):
    """Serves synthesized audio wav/mp3 files."""
    clean_fn = filename.lstrip("/")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        clean_fn,
        os.path.join(base_dir, clean_fn),
        os.path.join("/workspace/kathavaani", clean_fn),
        os.path.join("/home/dgx-s-bmu-soet-230737/kathavaani", clean_fn),
        os.path.join(base_dir, "outputs", clean_fn),
        os.path.join("/workspace/kathavaani/outputs", clean_fn),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isfile(path):
            media_type = "audio/mpeg" if clean_fn.endswith(".mp3") else "audio/wav"
            return FileResponse(path, media_type=media_type)
    raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found")


@app.get("/api/gallery")
def get_gallery():
    """Returns pre-synthesized master showcase productions (Story 01 & Story 02)."""
    productions = []
    
    # Story 01
    s1_timeline = "outputs/studio/story_01/timeline.json"
    if os.path.exists(s1_timeline):
        with open(s1_timeline, "r", encoding="utf-8") as f:
            t1 = json.load(f)
        productions.append({
            "id": "story_01",
            "title": "पंचतंत्र: शेर और चतुर खरगोश",
            "characters": ["कबीर (सूत्रधार)", "भासुरक (शेर)", "चतुर खरगोश"],
            "duration_sec": round(t1.get("total_duration_ms", 157300) / 1000, 1),
            "master_mp3": "/api/audio/outputs/studio/story_01/master.mp3",
            "master_wav": "/api/audio/outputs/studio/story_01/master.wav",
            "timeline": t1
        })
        
    # Story 02
    s2_timeline = "outputs/studio/story_02/timeline.json"
    if os.path.exists(s2_timeline):
        with open(s2_timeline, "r", encoding="utf-8") as f:
            t2 = json.load(f)
        productions.append({
            "id": "story_02",
            "title": "न्याय और ममता (विक्रमादित्य, दादी माँ, नन्ही पिंकी)",
            "characters": ["कबीर (सूत्रधार)", "दादी माँ", "राजा विक्रमादित्य", "नन्ही पिंकी"],
            "duration_sec": round(t2.get("total_duration_ms", 131438) / 1000, 1),
            "master_mp3": "/api/audio/outputs/studio/story_02/master.mp3",
            "master_wav": "/api/audio/outputs/studio/story_02/master.wav",
            "timeline": t2
        })
        
    return {"productions": productions}


@app.post("/api/listening_test")
@app.post("/api/listening-test")
def submit_listening_test(submission: ListeningTestSubmission):
    """Records blind human evaluation MOS and preference rating to JSONL & CSV."""
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator_id": submission.evaluator_id or submission.user_id or "Anonymous",
        "story_id": submission.story_id or "SENT_01",
        "model_a": submission.model_a or submission.sample_a_id or "KathaVaani",
        "model_b": submission.model_b or submission.sample_b_id or "MMS Baseline",
        "naturalness_score": submission.naturalness_score or submission.mos_rating_a or 4,
        "expressiveness_score": submission.expressiveness_score or submission.mos_rating_a or 4,
        "speaker_consistency_score": submission.speaker_consistency_score or 4,
        "audio_quality_score": submission.audio_quality_score or submission.mos_rating_a or 4,
        "preference": submission.preference or submission.preferred_sample or "A",
        "feedback": submission.feedback or submission.comments or "",
    }

    log_files = ["logs/listening_test_ratings.jsonl", "data_storage/listening_test_results.jsonl"]
    for lf in log_files:
        os.makedirs(os.path.dirname(lf), exist_ok=True)
        with open(lf, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "recorded", "message": "Rating saved successfully. Thank you!"}


# Mount Vite Frontend Production Build at /
dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
elif os.path.exists("web/dist"):
    app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
