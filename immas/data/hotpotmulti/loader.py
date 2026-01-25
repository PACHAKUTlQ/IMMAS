"""
immas.data.hotpotqa.loader

Utilities for loading and indexing the HotpotQA dataset.
Refactored to synthesize multi-turn dialogues by grouping questions sharing the same topic.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, cast

from datasets import load_dataset


HOTPOTQA_DATASET_NAME = "hotpot_qa"
HOTPOTQA_CONFIG_NAME = "distractor"

# Default max turns per synthetic dialogue to avoid overly huge contexts
DEFAULT_SYNTHETIC_DIALOGUE_SIZE = 5


@dataclass(frozen=True, slots=True)
class HotpotQATurn:
    """
    Represents a single turn (one Q&A pair) within a synthesized HotpotQA dialogue.
    """
    turn_id: str
    question: str
    answer: str
    level: str
    type: str
    context_titles: List[str]
    context_sentences: List[List[str]]

    @staticmethod
    def from_hf_example(example: Mapping[str, Any]) -> "HotpotQATurn":
        ex_id = str(example.get("id", ""))
        question = str(example.get("question", ""))
        answer = str(example.get("answer", ""))
        level = str(example.get("level", "unknown"))
        q_type = str(example.get("type", "unknown"))

        context_field = example.get("context")
        if not isinstance(context_field, Mapping):
             # Fallback or strict error
             context_field = {"title": [], "sentences": []}

        titles = context_field.get("title", [])
        sentences = context_field.get("sentences", [])
        
        # Ensure strict typing for context lists
        clean_titles = [str(t) for t in (titles if isinstance(titles, list) else [])]
        clean_sentences: List[List[str]] = []
        
        raw_sentences = sentences if isinstance(sentences, list) else []
        for grp in raw_sentences:
            if isinstance(grp, list):
                clean_sentences.append([str(s) for s in grp])
            else:
                clean_sentences.append([str(grp)])

        return HotpotQATurn(
            turn_id=ex_id,
            question=question,
            answer=answer,
            level=level,
            type=q_type,
            context_titles=clean_titles,
            context_sentences=clean_sentences,
        )

    def get_formatted_context(self) -> str:
        """Format the context paragraphs for this specific turn."""
        parts: List[str] = []
        for title, sents in zip(self.context_titles, self.context_sentences):
            text_block = "".join(sents)  # Usually sents are pre-tokenized/spaced
            parts.append(f"Title: {title}\n{text_block}")
        return "\n\n".join(parts)


@dataclass(frozen=True, slots=True)
class HotpotQADialogue:
    """
    A synthesized dialogue consisting of multiple HotpotQA turns grouped by topic.
    """
    dialogue_id: str
    primary_topic: str  # The title used to group these questions
    turns: List[HotpotQATurn]

    def num_turns(self) -> int:
        return len(self.turns)


class HotpotQADatasetIndex:
    """
    In-memory index of synthesized HotpotQA dialogues.
    """

    def __init__(self, dialogues: Dict[str, HotpotQADialogue]) -> None:
        self._dialogues = dialogues

    @classmethod
    def from_hf(
        cls, 
        *, 
        split: str = "validation",
        max_turns_per_dialogue: int = DEFAULT_SYNTHETIC_DIALOGUE_SIZE
    ) -> "HotpotQADatasetIndex":
        """
        Load HotpotQA and group questions into multi-turn dialogues.
        
        Strategy:
        1. Group examples by the title of their first supporting fact (heuristic for 'Topic').
        2. Chunk these groups into dialogues of size `max_turns_per_dialogue`.
        """
        ds = load_dataset(HOTPOTQA_DATASET_NAME, HOTPOTQA_CONFIG_NAME, split=split)
        
        # 1. Group by topic
        topic_map: Dict[str, List[HotpotQATurn]] = defaultdict(list)
        
        for ex in ds:
            try:
                turn = HotpotQATurn.from_hf_example(cast(Mapping[str, Any], ex))
                
                # Heuristic: Use the first supporting fact title as the "Topic"
                # If supporting facts are missing, use the first context title, or "Misc"
                supp_facts = ex.get("supporting_facts")
                topic = "Miscellaneous"
                if supp_facts and isinstance(supp_facts, Mapping):
                    supp_titles = supp_facts.get("title", [])
                    if supp_titles and len(supp_titles) > 0:
                        topic = str(supp_titles[0])
                elif turn.context_titles:
                     topic = turn.context_titles[0]
                
                topic_map[topic].append(turn)
            except Exception:
                # Silently skip malformed examples to ensure robustness
                continue

        # 2. Convert groups to Dialogues with chunking
        dialogues: Dict[str, HotpotQADialogue] = {}
        
        # Deterministic sort by topic name
        for topic in sorted(topic_map.keys()):
            turns = topic_map[topic]
            # Sort turns by ID to ensure deterministic chunking order
            turns.sort(key=lambda t: t.turn_id)
            
            # Chunking
            for i in range(0, len(turns), max_turns_per_dialogue):
                chunk_turns = turns[i : i + max_turns_per_dialogue]
                
                # Derive a stable ID: "topic_hash_chunkIndex"
                # We use hash to handle special chars in titles
                topic_hash = hashlib.md5(topic.encode("utf-8")).hexdigest()[:8]
                dialogue_id = f"hp_{topic_hash}_{i // max_turns_per_dialogue}"
                
                dialogues[dialogue_id] = HotpotQADialogue(
                    dialogue_id=dialogue_id,
                    primary_topic=topic,
                    turns=chunk_turns
                )

        return cls(dialogues)

    def iter_dialogues(self) -> Iterable[HotpotQADialogue]:
        return self._dialogues.values()