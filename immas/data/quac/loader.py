"""
immas.data.quac.loader

Utilities for loading and indexing the QuAC dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, cast

from datasets import load_dataset


QUAC_DATASET_NAME = "quac"


@dataclass(frozen=True, slots=True)
class QuacDialogue:
    """
    One QuAC dialogue.

    Unlike CoQA, QuAC has structured context parts:
    - wikipedia_page_title
    - section_title
    - background (first paragraph of the article)
    - context (the specific section text)
    """

    dialogue_id: str
    source: str  # Usually 'wikipedia' for QuAC
    title: str
    section_title: str
    background: str
    context: str
    questions: List[str]
    answers: List[str]

    @staticmethod
    def from_hf_example(example: Mapping[str, Any]) -> "QuacDialogue":
        """
        Build a QuacDialogue from one HuggingFace dataset example.
        """
        # QuAC always comes from Wikipedia
        source = "wikipedia"

        dialogue_id = str(example.get("dialogue_id", ""))
        if not dialogue_id:
            raise ValueError("Missing 'dialogue_id' in QuAC example")

        title = str(example.get("wikipedia_page_title", ""))
        section_title = str(example.get("section_title", ""))
        background = str(example.get("background", ""))
        context = str(example.get("context", ""))

        questions_field = example.get("questions")
        if not isinstance(questions_field, (list, tuple)):
            raise TypeError(
                f"Expected 'questions' to be a list, got {type(questions_field)!r}"
            )
        questions = [str(q) for q in questions_field]

        # QuAC has 'orig_answers' which contains the teacher's original answer text.
        # Structure: {'texts': ['ans1', 'ans2', ...], 'answer_starts': [...]}
        answers = _extract_orig_answers_list(
            dialogue_id=dialogue_id, orig_answers_field=example.get("orig_answers")
        )

        if len(questions) != len(answers):
            raise ValueError(
                f"Questions/answers length mismatch for dialogue_id={dialogue_id}: "
                f"{len(questions)} questions vs {len(answers)} answers"
            )

        return QuacDialogue(
            dialogue_id=dialogue_id,
            source=source,
            title=title,
            section_title=section_title,
            background=background,
            context=context,
            questions=questions,
            answers=answers,
        )

    def num_turns(self) -> int:
        """Number of turns in the dialogue."""
        return len(self.questions)

    def get_full_context_text(self) -> str:
        """
        Helper to construct the full context text for prompts.
        """
        parts = []
        if self.title:
            parts.append(f"Topic: {self.title}")
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        if self.background:
            parts.append(f"Background: {self.background}")
        parts.append("Context:")
        parts.append(self.context)
        return "\n".join(parts)


class QuacDatasetIndex:
    """
    In-memory index from dialogue_id to QuacDialogue.
    """

    def __init__(self, dialogues: Dict[str, QuacDialogue]) -> None:
        self._dialogues = dialogues

    @classmethod
    def from_hf(cls, *, split: str = "validation") -> "QuacDatasetIndex":
        """
        Load a QuAC split and build an index.
        Parameters
        ----------
        split
            HF split name: "train" or "validation".
        """
        ds = load_dataset(QUAC_DATASET_NAME, split=split)
        dialogues: Dict[str, QuacDialogue] = {}
        for ex in ds:
            d = QuacDialogue.from_hf_example(cast(Mapping[str, Any], ex))
            dialogues[d.dialogue_id] = d
        return cls(dialogues)

    def get_dialogue(self, dialogue_id: str) -> QuacDialogue:
        """Get a dialogue by id or raise KeyError."""
        return self._dialogues[dialogue_id]

    def get_answer(self, *, dialogue_id: str, turn_number: int) -> str:
        """
        Get the gold answer for a (dialogue_id, turn_number).
        Parameters
        ----------
        turn_number
            1-based turn index (Q1/A1 is turn_number=1).
        """
        if turn_number < 1:
            raise ValueError(f"turn_number must be >= 1, got {turn_number}")
        d = self.get_dialogue(dialogue_id)
        idx = turn_number - 1
        if idx >= len(d.answers):
            raise IndexError(
                f"turn_number out of range for dialogue_id={dialogue_id}: "
                f"{turn_number} > {len(d.answers)}"
            )
        return d.answers[idx]

    def iter_dialogues(self) -> Iterable[QuacDialogue]:
        """Iterate over all dialogues."""
        return self._dialogues.values()


def _extract_orig_answers_list(
    *, dialogue_id: str, orig_answers_field: Any
) -> List[str]:
    """
    Extract canonical answers from the 'orig_answers' field.

    In QuAC HF dataset, `orig_answers` is typically a dict:
    {
      'texts': ['answer turn 1', 'answer turn 2', ...],
      'answer_starts': [int, int, ...]
    }
    """
    if orig_answers_field is None:
        raise KeyError(f"Missing 'orig_answers' field for dialogue_id={dialogue_id}")

    if not isinstance(orig_answers_field, Mapping):
        raise TypeError(
            f"Expected 'orig_answers' to be a dict, got {type(orig_answers_field)!r} "
            f"for dialogue_id={dialogue_id}"
        )

    texts = orig_answers_field.get("texts")
    if not isinstance(texts, (list, tuple)):
        raise TypeError(
            f"Expected 'orig_answers.texts' to be a list, got {type(texts)!r} "
            f"for dialogue_id={dialogue_id}"
        )

    return [str(t) for t in texts]
