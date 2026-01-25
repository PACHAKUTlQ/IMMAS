#!/usr/bin/env python3
"""Build multi-turn request JSONL for one-click online experiments.

Supports (v0):
- CoQA (datasets: coqa)
- QuAC (datasets: quac)

Output format (one request per line):
{
  "dataset": "coqa",
  "conversation_id": "...",
  "turn_id": 3,
  "messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...],
  "ground_truth": "...",
  "metric": "f1"|"em"
}

History policy:
- user+assistant full history by default (per你的确认)
- optional truncation to last N turns

This script is optional; you can also provide your own JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def coqa_iter_conversations(ds) -> Tuple[str, str, List[str], List[str]]:
    # Each row contains: id, story, questions, answers
    for row in ds:
        conv_id = str(row.get("id") or row.get("story_id") or "")
        story = str(row.get("story") or "")
        questions = row.get("questions") or []
        answers = row.get("answers") or {}
        answer_texts = answers.get("input_text") or []
        yield conv_id, story, questions, answer_texts


def quac_iter_conversations(ds) -> Tuple[str, str, List[str], List[str]]:
    # QuAC rows contain: dialog_id, context, questions, answers (list of dicts)
    for row in ds:
        conv_id = str(row.get("dialog_id") or row.get("id") or "")
        context = str(row.get("context") or "")
        questions = row.get("questions") or []
        answers = row.get("answers") or []
        answer_texts: List[str] = []
        for a in answers:
            if isinstance(a, dict):
                answer_texts.append(str(a.get("text") or ""))
            else:
                answer_texts.append(str(a))
        yield conv_id, context, questions, answer_texts


def build_requests(
    dataset_name: str,
    split: str,
    max_conversations: Optional[int],
    max_turns_per_conv: Optional[int],
    history_turns: Optional[int],
    metric: str,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Missing dependency `datasets`. Install: pip install datasets"
        ) from e

    def _try_load(names: List[str]):
        last_err: Optional[Exception] = None
        for name in names:
            try:
                return name, load_dataset(name, split=split)
            except Exception as e:  # noqa: BLE001
                last_err = e
        if last_err is None:
            raise RuntimeError("No dataset names provided")
        raise last_err

    if dataset_name == "coqa":
        hf_name, ds = _try_load(["coqa", "allenai/coqa"])
        iterator = coqa_iter_conversations(ds)
    elif dataset_name == "quac":
        hf_name, ds = _try_load(["quac", "allenai/quac"])
        iterator = quac_iter_conversations(ds)
    else:
        raise SystemExit(f"Unsupported dataset: {dataset_name} (supported: coqa, quac)")

    # Informational: which dataset id actually loaded.
    # (Keeps behavior stable across environments/mirrors.)
    _ = hf_name

    out: List[Dict[str, Any]] = []
    conv_count = 0

    for conv_id, context, questions, answers in iterator:
        conv_count += 1
        if max_conversations is not None and conv_count > max_conversations:
            break

        turns = min(len(questions), len(answers))
        if max_turns_per_conv is not None:
            turns = min(turns, max_turns_per_conv)

        # Keep passage/context as a stable prefix message.
        history_messages: List[Dict[str, str]] = []
        if context.strip():
            history_messages.append(
                {
                    "role": "system",
                    "content": "You are given a passage. Answer the user's question based on it.\n\nPassage:\n"
                    + context.strip(),
                }
            )

        for t in range(turns):
            q = str(questions[t])
            a = str(answers[t])

            # Build messages: [system passage] + full user+assistant history, then current user question
            system_prefix: List[Dict[str, str]] = []
            turn_history = list(history_messages)
            if turn_history and (turn_history[0].get("role") == "system"):
                system_prefix = [turn_history[0]]
                turn_history = turn_history[1:]

            messages = system_prefix + turn_history + [{"role": "user", "content": q}]

            # Truncate history to last N turns (turn= user+assistant), while keeping system prefix.
            if history_turns is not None:
                # Each past turn contributes 2 messages.
                keep_msgs = history_turns * 2
                if keep_msgs > 0 and len(turn_history) > keep_msgs:
                    turn_history = turn_history[-keep_msgs:]
                messages = (
                    system_prefix + turn_history + [{"role": "user", "content": q}]
                )

            out.append(
                {
                    "dataset": dataset_name,
                    "conversation_id": conv_id or f"{dataset_name}-{conv_count}",
                    "turn_id": t + 1,
                    "messages": messages,
                    "ground_truth": a,
                    "metric": metric,
                }
            )

            # Update history with the gold answer (as assistant) for next turns.
            history_messages.append({"role": "user", "content": q})
            history_messages.append({"role": "assistant", "content": a})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coqa", "quac"], required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--max-conversations",
        type=int,
        default=0,
        help="Max number of conversations to process. Use 0 for all.",
    )
    ap.add_argument(
        "--max-turns-per-conv",
        type=int,
        default=0,
        help="Max number of turns per conversation. Use 0 for all.",
    )
    ap.add_argument("--history-turns", default="all", help="all or integer N")
    ap.add_argument("--metric", default="f1", choices=["f1", "em"])
    args = ap.parse_args()

    history_turns: Optional[int]
    if str(args.history_turns).lower() == "all":
        history_turns = None
    else:
        history_turns = int(args.history_turns)

    max_conversations = (
        None if int(args.max_conversations) <= 0 else int(args.max_conversations)
    )
    max_turns_per_conv = (
        None if int(args.max_turns_per_conv) <= 0 else int(args.max_turns_per_conv)
    )

    rows = build_requests(
        dataset_name=args.dataset,
        split=args.split,
        max_conversations=max_conversations,
        max_turns_per_conv=max_turns_per_conv,
        history_turns=history_turns,
        metric=str(args.metric),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} requests to {out_path}")


if __name__ == "__main__":
    main()
