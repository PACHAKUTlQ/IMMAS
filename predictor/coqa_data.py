"""
Utilities for loading and indexing the CoQA dataset.

This module is shared by both the fake API server and the client to keep data
extraction consistent and reduce "it works on my machine" mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

import hashlib

from typing import Any, Dict, Iterable, List, Mapping, Optional

from datasets import load_dataset


COQA_DATASET_NAME = "stanfordnlp/coqa"


@dataclass(frozen=True, slots=True)
class CoqaDialogue:
    """One CoQA dialogue: a story + a list of question/answer turns."""

    dialogue_id: str
    source: str
    story: str
    questions: List[str]
    answers: List[str]

    @staticmethod
    def from_hf_example(example: Mapping[str, Any]) -> "CoqaDialogue":
        """
        Build a CoqaDialogue from one HuggingFace dataset example.
        Notes
        -----
        Some installations of "stanfordnlp/coqa" provide no id field. In that
        case, we derive a stable id from content (story + questions).
        """
        source = str(example.get("source", "unknown"))
        story = str(example.get("story", ""))
        questions_field = example.get("questions")
        if not isinstance(questions_field, list):
            raise TypeError(
                f"Expected 'questions' to be a list, got {type(questions_field)!r}"
            )
        questions = [str(q) for q in questions_field]
        dialogue_id = _extract_or_derive_dialogue_id(
            example, story=story, questions=questions
        )
        answers = _extract_answers_list(
            dialogue_id=dialogue_id, answers_field=example.get("answers")
        )
        if len(questions) != len(answers):
            raise ValueError(
                f"Questions/answers length mismatch for dialogue_id={dialogue_id}: "
                f"{len(questions)} questions vs {len(answers)} answers"
            )
        return CoqaDialogue(
            dialogue_id=dialogue_id,
            source=source,
            story=story,
            questions=questions,
            answers=answers,
        )

    def num_turns(self) -> int:
        """Number of turns in the dialogue."""
        return len(self.questions)


class CoqaDatasetIndex:
    """
    In-memory index from dialogue_id to CoqaDialogue.
    For simplicity, we load the entire split once.
    """

    def __init__(self, dialogues: Dict[str, CoqaDialogue]) -> None:
        self._dialogues = dialogues

    @classmethod
    def from_hf(cls, *, split: str = "validation") -> "CoqaDatasetIndex":
        """
        Load a CoQA split and build an index.
        Parameters
        ----------
        split
            HF split name: typically "train" or "validation".
        """
        ds = load_dataset(COQA_DATASET_NAME, split=split)
        dialogues: Dict[str, CoqaDialogue] = {}
        for ex in ds:
            d = CoqaDialogue.from_hf_example(ex)
            # If a collision ever happens (unlikely), later we can disambiguate by
            # adding source or answers into the hash.
            dialogues[d.dialogue_id] = d
        return cls(dialogues)

    def get_dialogue(self, dialogue_id: str) -> CoqaDialogue:
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

    def iter_dialogues(self) -> Iterable[CoqaDialogue]:
        """Iterate over all dialogues."""
        return self._dialogues.values()


def _extract_dialogue_id(example: Mapping[str, Any]) -> str:
    for key in ("dialogue_id", "id", "story_id"):
        if key in example:
            return str(example[key])
    raise KeyError(
        "Could not find a dialogue id field. Tried keys: dialogue_id, id, story_id. "
        f"Available keys: {sorted(example.keys())}"
    )


def _extract_or_derive_dialogue_id(
    example: Mapping[str, Any], *, story: str, questions: List[str]
) -> str:
    """
    Extract a dialogue id if present; otherwise derive a deterministic synthetic id.
    The installed 'stanfordnlp/coqa' variant in your environment appears to have no
    id keys at all, so this function will usually fall back to hashing.
    """
    for key in ("dialogue_id", "id", "story_id"):
        if key in example:
            return str(example[key])
    # Derive a stable id from content (story + questions).
    # Include questions to avoid collisions if the same story appears multiple times.
    payload = story + "\n" + "\n".join(questions)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"coqa_sha1_{digest}"


def _extract_answers_list(*, dialogue_id: str, answers_field: Any) -> List[str]:
    """
    Extract answers as a list of strings from HF example["answers"].
    Common shapes:
    - list[{"input_text": str, "answer_start": int, "answer_end": int}, ...]
    - dict with key "input_text" containing list[str]
    """
    if answers_field is None:
        raise KeyError(f"Missing 'answers' field for dialogue_id={dialogue_id}")
    if isinstance(answers_field, list):
        out: List[str] = []
        for i, ans in enumerate(answers_field):
            if isinstance(ans, Mapping):
                if "input_text" in ans:
                    out.append(str(ans["input_text"]))
                elif "text" in ans:
                    out.append(str(ans["text"]))
                elif "answer" in ans:
                    out.append(str(ans["answer"]))
                else:
                    raise KeyError(
                        f"Unrecognized answer mapping at index {i} for dialogue_id={
                            dialogue_id
                        }. "
                        f"Keys: {sorted(ans.keys())}"
                    )
            else:
                out.append(str(ans))
        return out
    if isinstance(answers_field, Mapping):
        for key in ("input_text", "text", "answer"):
            v = answers_field.get(key)
            if isinstance(v, list):
                return [str(x) for x in v]
        raise KeyError(
            f"Unrecognized answers mapping for dialogue_id={dialogue_id}. "
            f"Keys: {sorted(answers_field.keys())}"
        )
    raise TypeError(
        f"Expected 'answers' to be a list or mapping for dialogue_id={dialogue_id}, "
        f"got {type(answers_field)!r}"
    )
