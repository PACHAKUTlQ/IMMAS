"""
immas.data.hotpotqa.loader

Utilities for loading and indexing the HotpotQA dataset.
Designed to handle the 'distractor' configuration by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, cast

from datasets import load_dataset


HOTPOTQA_DATASET_NAME = "hotpot_qa"
HOTPOTQA_CONFIG_NAME = "distractor"


@dataclass(frozen=True, slots=True)
class HotpotQAExample:
    """
    One HotpotQA example: a question, an answer, and a set of supporting documents.
    
    HotpotQA is a single-turn, multi-hop QA dataset.
    """

    id: str
    question: str
    answer: str
    level: str  # 'easy', 'medium', 'hard'
    type: str   # 'comparison', 'bridge'
    # Context in HotpotQA is a list of documents (title + sentences)
    context_titles: List[str]
    context_sentences: List[List[str]]

    @staticmethod
    def from_hf_example(example: Mapping[str, Any]) -> "HotpotQAExample":
        """
        Build a HotpotQAExample from one HuggingFace dataset example.
        """
        # Parse basic fields
        ex_id = str(example.get("id", ""))
        question = str(example.get("question", ""))
        answer = str(example.get("answer", ""))
        level = str(example.get("level", "unknown"))
        q_type = str(example.get("type", "unknown"))

        # Parse context
        # HF 'context' structure: {'title': ['t1', 't2'], 'sentences': [['s1'], ['s2a', 's2b']]}
        context_field = example.get("context")
        if not isinstance(context_field, Mapping):
             raise TypeError(
                f"Expected 'context' to be a dict/mapping, got {type(context_field)!r} "
                f"for id={ex_id}"
            )

        titles = context_field.get("title")
        if not isinstance(titles, list):
             raise TypeError(f"Expected context['title'] to be a list for id={ex_id}")
        
        sentences = context_field.get("sentences")
        if not isinstance(sentences, list):
             raise TypeError(f"Expected context['sentences'] to be a list for id={ex_id}")
        
        if len(titles) != len(sentences):
            raise ValueError(
                f"Context title/sentences length mismatch for id={ex_id}: "
                f"{len(titles)} titles vs {len(sentences)} sentence groups"
            )

        # Sanitize contents
        clean_titles = [str(t) for t in titles]
        clean_sentences: List[List[str]] = []
        for grp in sentences:
            if isinstance(grp, list):
                clean_sentences.append([str(s) for s in grp])
            else:
                # Fallback if dataset returns unexpected structure (e.g. numpy array)
                clean_sentences.append([str(grp)])

        return HotpotQAExample(
            id=ex_id,
            question=question,
            answer=answer,
            level=level,
            type=q_type,
            context_titles=clean_titles,
            context_sentences=clean_sentences,
        )

    def get_formatted_context(self) -> str:
        """
        Format the context paragraphs into a single string for the prompt.
        Format:
        
        Title 1
        Sentence 1a. Sentence 1b.
        
        Title 2
        Sentence 2a.
        """
        parts: List[str] = []
        for title, sents in zip(self.context_titles, self.context_sentences):
            # Join sentences with space. Some datasets have raw sentences without trailing spaces.
            text_block = " ".join(sents)
            parts.append(f"{title}\n{text_block}")
        return "\n\n".join(parts)


class HotpotQADatasetIndex:
    """
    In-memory index from id to HotpotQAExample.
    """

    def __init__(self, examples: Dict[str, HotpotQAExample]) -> None:
        self._examples = examples

    @classmethod
    def from_hf(cls, *, split: str = "validation") -> "HotpotQADatasetIndex":
        """
        Load a HotpotQA split and build an index.
        """
        ds = load_dataset(HOTPOTQA_DATASET_NAME, HOTPOTQA_CONFIG_NAME, split=split)
        examples: Dict[str, HotpotQAExample] = {}
        for ex in ds:
            item = HotpotQAExample.from_hf_example(cast(Mapping[str, Any], ex))
            examples[item.id] = item
        return cls(examples)

    def get_example(self, example_id: str) -> HotpotQAExample:
        """Get an example by id or raise KeyError."""
        return self._examples[example_id]

    def iter_examples(self) -> Iterable[HotpotQAExample]:
        """Iterate over all examples."""
        return self._examples.values()