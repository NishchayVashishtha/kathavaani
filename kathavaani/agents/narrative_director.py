"""
Agent 2: Narrative Director for KathaVaani
Generates Chain-of-Thought (CoT) emotional prosody reasoning and structured acoustic styles per turn.
Includes validation loop, sentiment heuristics, and critique pass.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

from kathavaani.agents.script_analyst import StoryTurn, ScriptAnalysisResult

logger = logging.getLogger("kathavaani.agents.director")

EMOTION_VOCAB = Literal[
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
    "tender",
    "sarcastic",
    "whisper",
    "tense",
]


class TurnStyle(BaseModel):
    idx: int = Field(description="Turn index corresponding to StoryTurn")
    reasoning: str = Field(description="Chain-of-thought prosody reasoning ('Why this emotion?')")
    emotion: EMOTION_VOCAB = Field(default="neutral", description="Primary emotional tone")
    intensity: float = Field(ge=0.0, le=1.0, default=0.5, description="Emotional intensity scalar (0.0 to 1.0)")
    pitch: Literal["low", "medium", "high"] = Field(default="medium")
    speed: Literal["slow", "medium", "fast"] = Field(default="medium")
    volume: Literal["low", "medium", "high"] = Field(default="medium")
    pause_before_ms: int = Field(ge=0, le=2000, default=300, description="Pause duration before turn in ms")
    pause_after_ms: int = Field(ge=0, le=2000, default=200, description="Pause duration after turn in ms")
    style_prompt: str = Field(default="", description="Natural language Hindi acoustic delivery description")
    scene_mood: str = Field(default="neutral", description="Overarching scene mood for background music")


class DirectorPlan(BaseModel):
    story_mood: str = Field(default="dramatic storytelling", description="Global mood of the story")
    turn_styles: List[TurnStyle] = Field(description="List of structured styles per turn")


class NarrativeDirector:
    """
    Directs vocal prosody, timing, and emotional delivery for multi-speaker storytelling.
    """

    HINDI_LEXICON = {
        "fear": ["डर", "कांप", "भूत", "राक्षस", "परछाई", "खौफ", "बचाओ", "भागो", "आदमखोर", "लाश"],
        "whisper": ["धीरे", "फुसफुसा", "चुप", "सन्नाटा", "दबे पांव", "आहट"],
        "angry": ["क्रोध", "गुस्सा", "हिम्मत", "गद्दार", "बंदी", "लज्जा", "झूठ", "धोखा", "कालकोठरी", "चिल्लाया"],
        "happy": ["खुश", "आनंद", "मिठाई", "वाह", "शाबाश", "बधाई", "सोने", "जीत", "हर्ष", "सफलता"],
        "excited": ["जल्दी", "मेला", "दौड़", "झूला", "उत्साह", "चलो"],
        "sad": ["रोना", "आंसू", "दुःख", "बिछड़", "बाढ़", "अनाथ", "तकलीफ", "कष्ट", "दर्द"],
        "tender": ["बेटा", "प्यार", "ममता", "गोद", "आशीर्वाद", "स्नेह", "संयम"],
        "surprise": ["असंभव", "अरे", "क्या सच", "चौंक", "अचानक", "जादुई"],
        "tense": ["रहस्य", "तहखाना", "घुंघरू", "अजनबी", "सन्नाटे", "अंधेरा", "साया"],
    }

    def __init__(self, llm_backend=None):
        self.llm_backend = llm_backend

    def analyze_prosody(self, script_result: ScriptAnalysisResult) -> DirectorPlan:
        """
        Directs prosody and style for each turn using context window and emotional reasoning.
        """
        logger.info(f"Directing prosody for {len(script_result.turns)} turns...")
        styles: List[TurnStyle] = []

        prev_emotion = "neutral"
        prev_intensity = 0.5

        for turn in script_result.turns:
            text = turn.text
            speaker = turn.speaker_name

            # 1. Detect emotion & intensity
            emotion, intensity, CoT = self._evaluate_turn_emotion(turn, prev_emotion)

            # 2. Derive acoustic modifiers
            pitch = "medium"
            speed = "medium"
            volume = "medium"
            pause_before = 300
            pause_after = 250
            scene_mood = "peaceful village"

            if emotion in ["fear", "whisper"]:
                speed = "slow"
                volume = "low"
                pitch = "high" if emotion == "fear" else "low"
                pause_before = 450
                pause_after = 350
                style_prompt = "फुसफुसाते हुए, डरी हुई धीमी आवाज़ में"
                scene_mood = "tense mystery"
            elif emotion in ["angry", "disgust"]:
                speed = "fast"
                volume = "high"
                pitch = "high"
                pause_before = 200
                pause_after = 400
                style_prompt = "क्रोध में चिल्लाते हुए, भारी और तीव्र आवाज़"
                scene_mood = "dramatic battle"
            elif emotion in ["happy", "excited"]:
                speed = "fast"
                volume = "high"
                pitch = "high"
                pause_before = 250
                pause_after = 200
                style_prompt = "हर्ष और उत्साह से भरी खिलखिलाती आवाज़"
                scene_mood = "joyful festival"
            elif emotion in ["sad", "tender"]:
                speed = "slow"
                volume = "low"
                pitch = "low"
                pause_before = 400
                pause_after = 500
                style_prompt = "गंभीर, रुआंसी और धीमी आवाज़"
                scene_mood = "melancholic emotional"
            elif emotion == "tense":
                speed = "slow"
                volume = "medium"
                pitch = "low"
                pause_before = 500
                pause_after = 400
                style_prompt = "रहस्यमयी और तनावपूर्ण सन्नाटे के साथ"
                scene_mood = "dark suspense"
            else:
                style_prompt = "स्वाभाविक और स्पष्ट वाचन"

            # 3. Critique pass & consistency check
            # Flag sudden emotion jump without narrative cue
            if abs(intensity - prev_intensity) > 0.6 and prev_emotion != "neutral" and emotion != "neutral":
                logger.info(f"Critique: Smoothing emotion jump from {prev_emotion} to {emotion}")
                intensity = round((intensity + prev_intensity) / 2.0, 2)

            prev_emotion = emotion
            prev_intensity = intensity

            ts = TurnStyle(
                idx=turn.idx,
                reasoning=CoT,
                emotion=emotion,
                intensity=intensity,
                pitch=pitch,
                speed=speed,
                volume=volume,
                pause_before_ms=pause_before,
                pause_after_ms=pause_after,
                style_prompt=style_prompt,
                scene_mood=scene_mood,
            )
            styles.append(ts)

        return DirectorPlan(
            story_mood="भारतीय लोककथा एवं नाटकीय प्रस्तुति",
            turn_styles=styles,
        )

    def _evaluate_turn_emotion(self, turn: StoryTurn, prev_emotion: str) -> tuple:
        """Rule-assisted Chain-of-Thought emotion reasoning."""
        text = turn.text
        speaker = turn.speaker_name

        matched_emotions = []
        for emo, keywords in self.HINDI_LEXICON.items():
            for kw in keywords:
                if kw in text:
                    matched_emotions.append((emo, kw))

        if matched_emotions:
            primary_emo, trigger_word = matched_emotions[0]
            intensity = 0.8 if len(matched_emotions) > 1 else 0.65
            CoT = (
                f"पात्र '{speaker}' के कथन में शब्द '{trigger_word}' आने से यहाँ {primary_emo.upper()} "
                f"का भाव उत्पन्न होता है। कथा के इस मोड़ पर तनाव/उत्तेजना बढ़ रही है।"
            )
            return primary_emo, intensity, CoT

        # Narration baseline vs character speech
        if turn.type == "narration":
            CoT = "कथा का सामान्य घटनाक्रम सूत्रधार द्वारा शांत और संतुलित गति से सुनाया जा रहा है।"
            return "neutral", 0.4, CoT
        else:
            # Carry over moderate emotion from scene context
            emo = prev_emotion if prev_emotion != "neutral" else "neutral"
            CoT = f"पात्र '{speaker}' का संवाद पूर्व संदर्भ ({emo}) के अनुरूप स्वाभाविक भाव से बोला जा रहा है।"
            return emo, 0.5, CoT


narrative_director = NarrativeDirector()
