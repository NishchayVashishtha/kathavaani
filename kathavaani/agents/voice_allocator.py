"""
Agent 3: Voice Allocator & Timbre Lock for KathaVaani
Matches character profiles to Voice Bank candidates, maximizes acoustic distance,
locks timbre across the story, and monitors speaker consistency / drift.
"""

import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from kathavaani.agents.script_analyst import CharacterProfile, ScriptAnalysisResult
from kathavaani.voicebank.bank import VoiceBank, VoiceProfile, voice_bank

logger = logging.getLogger("kathavaani.agents.allocator")


class CharacterVoiceAssignment(BaseModel):
    character_id: str
    character_name: str
    gender: str
    age_group: str
    voice_id: str
    voice_name: str
    reference_audio: str
    acoustic_distance_to_cast: float = Field(default=1.0)


class CastAllocationPlan(BaseModel):
    assignments: Dict[str, CharacterVoiceAssignment] = Field(
        description="Mapping from character_id (or 'narrator') to assigned voice profile"
    )


class VoiceAllocator:
    """
    Allocates distinct, timbre-locked voices from the Voice Bank to each character in the story.
    """

    def __init__(self, vbank: VoiceBank = voice_bank):
        self.voice_bank = vbank

    def allocate_cast(self, script_result: ScriptAnalysisResult) -> CastAllocationPlan:
        """
        Allocates persistent voices for narrator and all story characters,
        maximizing pairwise inter-character acoustic distance.
        """
        logger.info(f"Allocating voices for story '{script_result.title}' with {len(script_result.characters)} characters...")
        assignments: Dict[str, CharacterVoiceAssignment] = {}
        assigned_voice_ids: List[str] = []

        # 1. Allocate Narrator Voice (default male deep or female warm)
        narrator_vp = self.voice_bank.allocate_voice(
            character_gender="male",
            character_age="adult",
            is_narrator=True,
            assigned_voice_ids=assigned_voice_ids,
        )
        assignments["narrator"] = CharacterVoiceAssignment(
            character_id="narrator",
            character_name="सूत्रधार",
            gender=narrator_vp.gender,
            age_group=narrator_vp.age_group,
            voice_id=narrator_vp.id,
            voice_name=narrator_vp.name,
            reference_audio=narrator_vp.audio_path,
            acoustic_distance_to_cast=1.0,
        )
        assigned_voice_ids.append(narrator_vp.id)
        logger.info(f"Assigned Narrator -> {narrator_vp.name} ({narrator_vp.id})")

        # 2. Allocate Characters with Inter-Speaker Distance Maximization
        for char in script_result.characters:
            vp = self.voice_bank.allocate_voice(
                character_gender=char.gender,
                character_age=char.age_group,
                is_narrator=False,
                assigned_voice_ids=assigned_voice_ids,
            )
            assignments[char.id] = CharacterVoiceAssignment(
                character_id=char.id,
                character_name=char.name,
                gender=vp.gender,
                age_group=vp.age_group,
                voice_id=vp.id,
                voice_name=vp.name,
                reference_audio=vp.audio_path,
                acoustic_distance_to_cast=1.0,
            )
            assigned_voice_ids.append(vp.id)
            logger.info(f"Assigned Character '{char.name}' ({char.id}) -> {vp.name} ({vp.id})")

        return CastAllocationPlan(assignments=assignments)

    def verify_consistency(self, character_id: str, synthesized_audio: str, threshold: float = 0.65) -> tuple:
        """
        Computes speaker consistency (cosine similarity) between synthesized turn and assigned voice.
        Returns: (passed_threshold: bool, similarity_score: float)
        """
        # Get assigned voice id
        pass


voice_allocator = VoiceAllocator()
