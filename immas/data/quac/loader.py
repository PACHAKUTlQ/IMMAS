"""
immas.data.quac.loader

Utilities for loading and indexing the QuAC dataset.

- Uses dialogue_id provided by the dataset (falls back to a derived id if missing).
- Supports multiple gold answers per turn (dev/validation typically has 5 refs).
- Applies QuAC official "handle_cannot" logic to references:
  - If #CANNOTANSWER >= #non-cannot spans => refs = ["CANNOTANSWER"]
  - Else drop all "CANNOTANSWER" refs and keep span refs
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, cast

from datasets import load_dataset

QUAC_DATASET_NAME = "allenai/quac"
QUAC_CONFIG_NAME = "plain_text"


def handle_cannot(refs: List[str]) -> List[str]:
    """
    Official QuAC behavior:
    - If CANNOTANSWER refs are majority (or tied) => keep only CANNOTANSWER
    - Else drop CANNOTANSWER refs
    """
    cleaned = [str(r) for r in refs if str(r or "").strip()]
    num_cannot = sum(1 for r in cleaned if r.strip() == "CANNOTANSWER")
    num_spans = sum(1 for r in cleaned if r.strip() != "CANNOTANSWER")
    if num_cannot >= num_spans:
        return ["CANNOTANSWER"]
    return [r for r in cleaned if r.strip() != "CANNOTANSWER"]


def _derive_dialogue_id(*, context: str, questions: List[str]) -> str:
    payload = (context or "") + "\n" + "\n".join(questions)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"quac_sha1_{digest}"


def _ensure_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]


def _extract_answers_texts_per_turn(
    dialogue_id: str, answers_field: Any
) -> List[List[str]]:
    """
    HF QuAC 'answers' commonly arrives as a dict-of-lists-of-lists:
      answers_field["texts"] == List[List[str]]
    but be tolerant to other shapes.
    """
    if answers_field is None:
        raise KeyError(f"Missing 'answers' field for dialogue_id={dialogue_id}")

    # Common HF shape (like SQuAD): {"texts": [[...], ...], "answer_starts": [[...], ...]}
    if isinstance(answers_field, Mapping):
        texts = answers_field.get("texts")
        if isinstance(texts, list):
            return [_ensure_str_list(t) for t in texts]

    # Alternative: list[{"texts": [...], ...}, ...]
    if isinstance(answers_field, list):
        out: List[List[str]] = []
        for item in answers_field:
            if isinstance(item, Mapping):
                out.append(_ensure_str_list(item.get("texts")))
            else:
                out.append([str(item)])
        return out

    raise TypeError(
        f"Unrecognized 'answers' type for dialogue_id={dialogue_id}: {
            type(answers_field)!r
        }"
    )


@dataclass(frozen=True, slots=True)
class QuacDialogue:
    dialogue_id: str
    wikipedia_page_title: str
    background: str
    section_title: str
    context: str
    turn_ids: List[str]
    questions: List[str]
    answers: List[List[str]]  # processed refs per turn (after handle_cannot)

    @staticmethod
    def from_hf_example(example: Mapping[str, Any]) -> "QuacDialogue":
        dialogue_id = str(example.get("dialogue_id") or "").strip()
        wikipedia_page_title = str(example.get("wikipedia_page_title") or "")
        background = str(example.get("background") or "")
        section_title = str(example.get("section_title") or "")
        context = str(example.get("context") or "")

        questions_field = example.get("questions")
        if not isinstance(questions_field, list):
            raise TypeError(
                f"Expected 'questions' to be a list, got {type(questions_field)!r}"
            )
        questions = [str(q) for q in questions_field]

        if not dialogue_id:
            dialogue_id = _derive_dialogue_id(context=context, questions=questions)

        turn_ids_field = example.get("turn_ids")
        if isinstance(turn_ids_field, list):
            turn_ids = [str(t) for t in turn_ids_field]
        else:
            # Fallback if missing
            turn_ids = [f"{dialogue_id}_q#{i}" for i in range(len(questions))]

        answers_raw = _extract_answers_texts_per_turn(
            dialogue_id, example.get("answers")
        )
        # If lengths mismatch, truncate to be safe (avoid runtime crashes).
        n = min(len(questions), len(answers_raw))
        questions = questions[:n]
        turn_ids = turn_ids[:n]
        answers_raw = answers_raw[:n]

        answers = [handle_cannot([str(a) for a in refs]) for refs in answers_raw]

        return QuacDialogue(
            dialogue_id=dialogue_id,
            wikipedia_page_title=wikipedia_page_title,
            background=background,
            section_title=section_title,
            context=context,
            turn_ids=turn_ids,
            questions=questions,
            answers=answers,
        )

    def num_turns(self) -> int:
        return len(self.questions)


class QuacDatasetIndex:
    """
    In-memory index from dialogue_id to QuacDialogue.
    Loads the entire split once.
    """

    def __init__(self, dialogues: Dict[str, QuacDialogue]) -> None:
        self._dialogues = dialogues

    @classmethod
    def from_hf(cls, *, split: str = "validation") -> "QuacDatasetIndex":
        ds = load_dataset(QUAC_DATASET_NAME, QUAC_CONFIG_NAME, split=split)
        dialogues: Dict[str, QuacDialogue] = {}
        for ex in ds:
            d = QuacDialogue.from_hf_example(cast(Mapping[str, Any], ex))
            dialogues[d.dialogue_id] = d
        return cls(dialogues)

    def get_dialogue(self, dialogue_id: str) -> QuacDialogue:
        return self._dialogues[dialogue_id]

    def get_answers(self, *, dialogue_id: str, turn_number: int) -> List[str]:
        """
        Return list of gold refs for (dialogue_id, turn_number), 1-based.
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
        return list(d.answers[idx])

    def get_answer(self, *, dialogue_id: str, turn_number: int) -> str:
        """
        Compatibility helper (single-string). Returns the first reference if present.
        """
        refs = self.get_answers(dialogue_id=dialogue_id, turn_number=turn_number)
        return refs[0] if refs else ""

    def iter_dialogues(self) -> Iterable[QuacDialogue]:
        return self._dialogues.values()
