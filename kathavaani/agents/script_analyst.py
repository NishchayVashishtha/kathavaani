"""
Agent 1: Script Analyst for KathaVaani
Extracts character cast, gender/age demographics, dialogue attribution,
verb agreement heuristics, and clause-aware chunking (<= 200 chars).
Validated via Pydantic v2.
"""

import re
import json
import logging
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger("kathavaani.agents.analyst")


class CharacterProfile(BaseModel):
    id: str = Field(description="Unique character ID (e.g. 'C1', 'C2')")
    name: str = Field(description="Canonical character name in Devanagari")
    gender: Literal["male", "female", "unknown"] = Field(description="Inferred or stated gender")
    age_group: Literal["child", "young", "adult", "elderly"] = Field(description="Demographic age category")
    personality: str = Field(default="neutral", description="Short personality/demeanor description")
    aliases: List[str] = Field(default_factory=list, description="Common aliases, titles or honorifics")


class StoryTurn(BaseModel):
    idx: int = Field(description="0-indexed sequence position of the turn")
    type: Literal["narration", "dialogue"] = Field(description="Whether line is story narration or character speech")
    speaker_id: str = Field(description="'narrator' or character ID (e.g. 'C1')")
    speaker_name: str = Field(description="Display name of the speaker")
    text: str = Field(description="Devanagari text chunk for TTS synthesis (<= 200 chars)")
    addressed_to: Optional[str] = Field(default=None, description="Addressee ID if identifiable dialogue")


class ScriptAnalysisResult(BaseModel):
    title: str = Field(default="हिंदी कहानी", description="Story title")
    characters: List[CharacterProfile] = Field(description="List of detected story characters")
    turns: List[StoryTurn] = Field(description="Ordered sequence of synthesizable narration and dialogue turns")


class ScriptAnalyst:
    """
    Parses unstructured Hindi storytelling text into structured, speaker-attributed turns.
    Combines Devanagari linguistic heuristics with LLM structuring.
    """

    REPORTING_VERBS_MALE = ["बोला", "कहा", "पूछा", "चिल्लाया", "दहाड़ा", "गरजा", "फुसफुसाया", "हंसा", "रोया"]
    REPORTING_VERBS_FEMALE = ["बोली", "कही", "पूछी", "चिल्लाई", "दहाड़ी", "गरजी", "फुसफुसाई", "हंसी", "रोई"]
    HONORIFIC_VERBS = ["बोले", "कहे", "पूछे", "चिल्लाए", "फरमाया"]

    def __init__(self, llm_backend=None):
        self.llm_backend = llm_backend

    def chunk_narration(self, text: str, max_chars: int = 180) -> List[str]:
        """
        Splits long Hindi narration into natural, sentence/clause-aware TTS chunks (<= max_chars).
        Never cuts mid-word; respects Devanagari dandas (।), question marks (?), and conjunctions.
        """
        text = text.strip()
        if not text:
            return []

        # Split on sentence boundaries first
        raw_sentences = re.split(r"([।?!;\n]+)", text)
        sentences = []
        for i in range(0, len(raw_sentences) - 1, 2):
            s = raw_sentences[i].strip() + (raw_sentences[i + 1].strip() if i + 1 < len(raw_sentences) else "")
            if s:
                sentences.append(s)
        if len(raw_sentences) % 2 == 1 and raw_sentences[-1].strip():
            sentences.append(raw_sentences[-1].strip())

        chunks = []
        for sent in sentences:
            if len(sent) <= max_chars:
                chunks.append(sent)
            else:
                # Sub-split long sentence by clauses / conjunctions
                clause_pattern = r"(,| और | लेकिन | परंतु | क्योंकि | जब | तब | जिससे | कि )"
                parts = re.split(clause_pattern, sent)
                current = ""
                for p in parts:
                    if len(current) + len(p) <= max_chars:
                        current += p
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        current = p
                if current.strip():
                    chunks.append(current.strip())

        return chunks if chunks else [text]

    def parse_rule_based(self, story_text: str) -> ScriptAnalysisResult:
        """
        Robust deterministic Devanagari parser extracting dialogue quotes,
        speaker gender from verb endings, and splitting turns.
        """
        # Normalize quotes
        norm_text = story_text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # Pattern to capture quoted dialogue and surrounding context
        # e.g.: दादी ने कहा, "तुम कहाँ जा रहे हो?"
        quote_pattern = r'([^\n"]*?)(?:"([^"]+)"|—\s*([^।\n]+))(.*?)(?=[।\n]|$)'

        characters: Dict[str, CharacterProfile] = {}
        turns: List[StoryTurn] = []
        turn_idx = 0

        # Split text into lines/paragraphs
        lines = [line.strip() for line in norm_text.split("\n") if line.strip()]

        for line in lines:
            # Check for dialogue quotes in this line
            quotes = list(re.finditer(r'"([^"]+)"', line))

            if not quotes:
                # Pure narration line
                for chunk in self.chunk_narration(line):
                    turns.append(StoryTurn(
                        idx=turn_idx,
                        type="narration",
                        speaker_id="narrator",
                        speaker_name="सूत्रधार",
                        text=chunk,
                    ))
                    turn_idx += 1
            else:
                last_end = 0
                for match in quotes:
                    start, end = match.span()
                    prefix = line[last_end:start].strip()
                    dialogue_text = match.group(1).strip()
                    last_end = end

                    # If there's narrative prefix, emit it
                    if prefix:
                        for chunk in self.chunk_narration(prefix):
                            turns.append(StoryTurn(
                                idx=turn_idx,
                                type="narration",
                                speaker_id="narrator",
                                speaker_name="सूत्रधार",
                                text=chunk,
                            ))
                            turn_idx += 1

                    # Infer speaker from prefix
                    speaker_id, speaker_name, gender, age = self._infer_speaker(prefix, characters)
                    if speaker_id not in characters:
                        characters[speaker_id] = CharacterProfile(
                            id=speaker_id,
                            name=speaker_name,
                            gender=gender,
                            age_group=age,
                            personality="expressive",
                        )

                    # Emit dialogue turn(s)
                    for chunk in self.chunk_narration(dialogue_text):
                        turns.append(StoryTurn(
                            idx=turn_idx,
                            type="dialogue",
                            speaker_id=speaker_id,
                            speaker_name=speaker_name,
                            text=chunk,
                        ))
                        turn_idx += 1

                # Trailing narration after last quote
                suffix = line[last_end:].strip()
                if suffix:
                    for chunk in self.chunk_narration(suffix):
                        turns.append(StoryTurn(
                            idx=turn_idx,
                            type="narration",
                            speaker_id="narrator",
                            speaker_name="सूत्रधार",
                            text=chunk,
                        ))
                        turn_idx += 1

        # Fallback if no characters extracted
        if not characters:
            characters["C1"] = CharacterProfile(
                id="C1",
                name="पात्र",
                gender="male",
                age_group="adult",
                personality="dramatic",
            )

        return ScriptAnalysisResult(
            title="हिंदी लोककथा",
            characters=list(characters.values()),
            turns=turns,
        )

    def _infer_speaker(self, prefix: str, existing_chars: dict) -> tuple:
        """Heuristic gender & speaker extraction from Hindi verb morphology."""
        gender = "male"
        age = "adult"
        name = "पात्र"

        # Check female verbs
        for v in self.REPORTING_VERBS_FEMALE:
            if v in prefix:
                gender = "female"
                break

        # Check demographic titles
        if any(w in prefix for w in ["दादी", "नानी", "अम्मा", "बुढ़िया"]):
            gender = "female"
            age = "elderly"
            name = "दादी माँ"
        elif any(w in prefix for w in ["दादा", "बाबा", "बूढ़ा", "साधु"]):
            gender = "male"
            age = "elderly"
            name = "बाबा"
        elif any(w in prefix for w in ["बच्चा", "लड़का", "बेटा", "पिंटू", "रोहन"]):
            gender = "male"
            age = "young"
            name = "पिंटू"
        elif any(w in prefix for w in ["बच्ची", "लड़की", "बेटी", "पिंकी", "तन्वी"]):
            gender = "female"
            age = "young"
            name = "तन्वी"
        elif any(w in prefix for w in ["राजा", "महाराज", "सम्राट"]):
            gender = "male"
            age = "adult"
            name = "महाराज"
        elif any(w in prefix for w in ["रानी", "महारानी"]):
            gender = "female"
            age = "adult"
            name = "महारानी"

        # Find matching existing character or assign new
        for cid, char in existing_chars.items():
            if char.name == name:
                return cid, char.name, char.gender, char.age_group

        cid = f"C{len(existing_chars) + 1}"
        return cid, name, gender, age

    def analyze(self, story_text: str) -> ScriptAnalysisResult:
        """Main analysis entrypoint with schema validation."""
        logger.info(f"Analyzing story script ({len(story_text)} chars)...")
        res = self.parse_rule_based(story_text)
        logger.info(f"Script Analyst produced {len(res.characters)} characters and {len(res.turns)} turns.")
        return res


script_analyst = ScriptAnalyst()
