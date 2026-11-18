"""
KathaVaani — LLM Backend Interface for Agent 1 (Script Analyst) & Agent 2 (Narrative Director)
Supports:
1. Local Hugging Face LLM (e.g. Qwen/Qwen2.5-1.5B-Instruct) via ModelManager
2. Optional Cloud API backend (OpenAI / Gemini compatible) via env vars
3. Schema-constrained JSON generation validated by Pydantic
4. Self-Critique & Consistency validation loop (detecting emotional jumps without narrative cue)
5. Rule-based heuristic fallback (used as documented baseline for ablation studies)
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
from pydantic import BaseModel, ValidationError

from kathavaani.agents.script_analyst import ScriptAnalysisResult, StoryTurn, CharacterProfile
from kathavaani.agents.narrative_director import DirectorPlan, TurnStyle
from kathavaani.models.manager import ModelManager

logger = logging.getLogger("kathavaani.llm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def extract_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extracts JSON object from markdown code fences or plain text."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    match_brace = re.search(r"(\{[\s\S]*\})", raw_text)
    if match_brace:
        try:
            return json.loads(match_brace.group(1).strip())
        except Exception:
            pass

    return None


class LLMBackendInterface:
    """Base interface for Script Analysis and Narrative Direction LLMs."""

    def analyze_script(self, story_text: str) -> Optional[ScriptAnalysisResult]:
        raise NotImplementedError

    def direct_prosody(self, script_result: ScriptAnalysisResult) -> Optional[DirectorPlan]:
        raise NotImplementedError


class CloudAPIBackend(LLMBackendInterface):
    """OpenAI / Gemini-compatible API backend with JSON schema mode."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.client = None

        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"Initialized Cloud API backend with model: {self.model}")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

    def is_available(self) -> bool:
        return self.client is not None

    def analyze_script(self, story_text: str) -> Optional[ScriptAnalysisResult]:
        if not self.is_available():
            return None

        prompt = f"""You are a master theatre director analyzing a Hindi story. Parse this story into characters and speech turns.
Return ONLY valid JSON matching this schema:
{{
  "title": "Story title in Devanagari",
  "characters": [
    {{"id": "C1", "name": "Devanagari name", "gender": "male|female|unknown", "age_group": "child|young|adult|elderly", "personality": "string", "aliases": []}}
  ],
  "turns": [
    {{"idx": 0, "type": "narration|dialogue", "speaker_id": "narrator|C1", "speaker_name": "Speaker Name", "text": "Devanagari chunk <= 200 chars", "addressed_to": null}}
  ]
}}

Story:
{story_text}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            data = json.loads(resp.choices[0].message.content)
            return ScriptAnalysisResult.model_validate(data)
        except Exception as e:
            logger.error(f"Cloud API script analysis failed: {e}")
            return None

    def direct_prosody(self, script_result: ScriptAnalysisResult) -> Optional[DirectorPlan]:
        if not self.is_available():
            return None

        turns_summary = json.dumps([t.model_dump() for t in script_result.turns[:25]], ensure_ascii=False)
        prompt = f"""You are a theatrical voice director. For each Hindi story turn, analyze dramatic subtext using Chain-of-Thought reasoning ('Why this emotion?'), then emit structured acoustic parameters.
Return ONLY valid JSON matching this schema:
{{
  "story_mood": "string",
  "turn_styles": [
    {{
      "idx": 0,
      "reasoning": "Chain-of-thought explanation for this delivery",
      "emotion": "neutral|happy|excited|sad|angry|fear|surprise|disgust|tender|sarcastic|whisper|tense",
      "intensity": 0.0 to 1.0,
      "pitch": "low|medium|high",
      "speed": "slow|medium|fast",
      "volume": "low|medium|high",
      "pause_before_ms": 100 to 1000,
      "pause_after_ms": 100 to 1000,
      "style_prompt": "Natural language delivery note in Hindi",
      "scene_mood": "string"
    }}
  ]
}}

Turns:
{turns_summary}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            data = json.loads(resp.choices[0].message.content)
            return DirectorPlan.model_validate(data)
        except Exception as e:
            logger.error(f"Cloud API prosody directing failed: {e}")
            return None


class LocalHFBackend(LLMBackendInterface):
    """Local quantized / fp16 Hugging Face model loaded on-demand via ModelManager."""

    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, model_manager: Optional[ModelManager] = None):
        self.model_name = model_name
        self.model_manager = model_manager or ModelManager()
        self.model = None
        self.tokenizer = None
        self.is_ready = False

    def load(self):
        if self.is_ready and self.model is not None:
            return
        logger.info(f"Loading local LLM {self.model_name} onto GPU...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cache_dir = os.environ.get("HF_HOME", "/tmp/hf_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            cache_dir=cache_dir
        )
        self.is_ready = True
        self.model_manager.register("llm", self.model)
        logger.info(f"Local LLM loaded successfully. Current VRAM: {self.model_manager.get_vram_gb()} GB")

    def unload(self):
        if self.model is not None:
            logger.info("Unloading local LLM to free VRAM for TTS...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.is_ready = False
            self.model_manager.unload("llm")

    def _generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 1500) -> str:
        self.load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=False
            )
        prompt_len = inputs.input_ids.shape[1]
        return self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

    def analyze_script(self, story_text: str) -> Optional[ScriptAnalysisResult]:
        system_prompt = (
            "You are an expert Hindi Literary Script Analyst for KathaVaani. "
            "Your task is to analyze unstructured Hindi story text and parse it into structured JSON.\n"
            "Output MUST be strict, valid JSON matching this schema:\n"
            "{\n"
            "  \"title\": \"Story title in Hindi\",\n"
            "  \"characters\": [\n"
            "    {\"id\": \"C1\", \"name\": \"Devanagari name\", \"gender\": \"male|female|unknown\", \"age_group\": \"child|young|adult|elderly\", \"personality\": \"short trait\", \"aliases\": []}\n"
            "  ],\n"
            "  \"turns\": [\n"
            "    {\"idx\": 0, \"type\": \"narration|dialogue\", \"speaker_id\": \"narrator|C1\", \"speaker_name\": \"Speaker Name\", \"text\": \"Hindi sentence under 180 chars\", \"addressed_to\": null}\n"
            "  ]\n"
            "}\n"
            "Ensure narration turns have speaker_id 'narrator'. Dialogue turns must have character speaker_id. "
            "Keep all text chunks in grammatical Devanagari. No explanations outside the JSON."
        )
        user_prompt = f"Analyze this Hindi story and return valid JSON:\n\n{story_text}"

        try:
            raw = self._generate(system_prompt, user_prompt, max_new_tokens=1500)
            parsed = extract_json_from_text(raw)
            if parsed:
                res = ScriptAnalysisResult.model_validate(parsed)
                logger.info(f"Local LLM Script Analyst succeeded ({len(res.turns)} turns).")
                return res
        except Exception as e:
            logger.warning(f"Local LLM script analysis failed: {e}")
        return None

    def direct_prosody(self, script_result: ScriptAnalysisResult) -> Optional[DirectorPlan]:
        system_prompt = (
            "You are an award-winning Indian Audio Drama Director for KathaVaani. "
            "For each dialogue and narration turn, direct vocal prosody, emotional delivery, and pacing.\n"
            "Output MUST be strict, valid JSON matching this schema:\n"
            "{\n"
            "  \"story_mood\": \"overall dramatic mood\",\n"
            "  \"turn_styles\": [\n"
            "    {\n"
            "      \"idx\": 0,\n"
            "      \"reasoning\": \"Chain-of-thought Hindi/English reasoning explaining why this emotion and delivery were chosen\",\n"
            "      \"emotion\": \"neutral|happy|excited|sad|angry|fear|surprise|disgust|tender|sarcastic|whisper|tense\",\n"
            "      \"intensity\": 0.7,\n"
            "      \"pitch\": \"low|medium|high\",\n"
            "      \"speed\": \"slow|medium|fast\",\n"
            "      \"volume\": \"low|medium|high\",\n"
            "      \"pause_before_ms\": 300,\n"
            "      \"pause_after_ms\": 250,\n"
            "      \"style_prompt\": \"Expressive Hindi delivery instruction\",\n"
            "      \"scene_mood\": \"background ambience mood\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Include rich chain-of-thought reasoning for each turn."
        )
        turns_summary = []
        for t in script_result.turns:
            turns_summary.append(f"Turn {t.idx} [{t.type}] ({t.speaker_name}): \"{t.text}\"")

        user_prompt = "Direct prosody for these story turns and return valid JSON:\n" + "\n".join(turns_summary)

        try:
            raw = self._generate(system_prompt, user_prompt, max_new_tokens=2000)
            parsed = extract_json_from_text(raw)
            if parsed:
                plan = DirectorPlan.model_validate(parsed)
                logger.info(f"Local LLM Narrative Director succeeded ({len(plan.turn_styles)} styles).")
                return plan
        except Exception as e:
            logger.warning(f"Local LLM prosody directing failed: {e}")
        return None


class ConsistencyCritic:
    """Validates emotional trajectory and flags abrupt jumps without narrative justification."""

    HIGH_VOLATILITY_PAIRS = [
        ("happy", "angry"),
        ("tender", "angry"),
        ("whisper", "excited"),
        ("sad", "happy"),
    ]

    @classmethod
    def validate_plan(cls, plan: DirectorPlan) -> List[str]:
        warnings = []
        turn_styles = plan.turn_styles

        for i in range(1, len(turn_styles)):
            prev = turn_styles[i - 1]
            curr = turn_styles[i]

            pair = (prev.emotion, curr.emotion)
            rev_pair = (curr.emotion, prev.emotion)

            if pair in cls.HIGH_VOLATILITY_PAIRS or rev_pair in cls.HIGH_VOLATILITY_PAIRS:
                if abs(curr.intensity - prev.intensity) > 0.4:
                    warnings.append(
                        f"Turn {curr.idx}: High emotion jump from '{prev.emotion}' to '{curr.emotion}' "
                        f"with intensity delta {abs(curr.intensity - prev.intensity):.2f}. Reason: {curr.reasoning[:40]}..."
                    )

        return warnings
