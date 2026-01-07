"""
Client demo:
- Loads CoQA
- Generates multi-turn prompts (full story + full history)
- Calls the fake OpenAI server
- Updates an online predictor (River)
- Computes text prefix match ratio as a feature
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Tuple

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from coqa_data import COQA_DATASET_NAME, CoqaDialogue
from coqa_prompt import CoqaPromptFormatter
from predictor import AgentPredictor, PredictorInput
from prefix_cache import TextPrefixCache


HistoryTurn = Tuple[str, str]


def normalize_eval(text: str) -> str:
    """Normalize text for simple equality checks."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def main() -> None:
    split = os.environ.get("COQA_SPLIT", "validation")
    model_name = os.environ.get("FAKE_MODEL_NAME", "fake-coqa")

    # Fake API endpoint
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")

    ds = load_dataset(COQA_DATASET_NAME, split=split)

    predictor = AgentPredictor()
    cache = TextPrefixCache()

    # Keep the demo small by default; scale up later.
    max_dialogues = int(os.environ.get("MAX_DIALOGUES", "3"))
    max_turns_per_dialogue = int(os.environ.get("MAX_TURNS", "5"))

    print(
        f"Running CoQA split={split} with max_dialogues={max_dialogues}, max_turns={
            max_turns_per_dialogue
        }"
    )
    print("============================================================")

    for ex in tqdm(ds.select(range(min(max_dialogues, len(ds)))), desc="Dialogues"):
        dialogue = CoqaDialogue.from_hf_example(ex)

        history: List[HistoryTurn] = []
        n_turns = min(dialogue.num_turns(), max_turns_per_dialogue)

        for turn_idx in range(n_turns):
            turn_number = turn_idx + 1
            question = dialogue.questions[turn_idx]
            gold_answer = dialogue.answers[turn_idx]

            prompt = CoqaPromptFormatter.format_turn(
                dialogue_id=dialogue.dialogue_id,
                source=dialogue.source,
                story=dialogue.story,
                history=history,
                question=question,
                turn_number=turn_number,
            )

            kvmatch = cache.match_ratio(
                model=model_name, dialogue_id=dialogue.dialogue_id, prompt_text=prompt
            )

            inp = PredictorInput(
                model=model_name,
                source=dialogue.source,
                dialogue_id=dialogue.dialogue_id,
                turn_number=turn_number,
                prompt_text=prompt,
                kvmatch=kvmatch,
            )

            pred = predictor.predict(inp)

            print(
                f"\nDialogue={dialogue.dialogue_id} source={dialogue.source} turn={
                    turn_number
                }"
            )
            print(f"  Feature kvmatch={kvmatch:.3f} prompt_chars={len(prompt)}")
            print(
                f"  Predict latency_ms={pred['latency_ms'][0]:.2f} cost_tokens={
                    pred['cost_tokens'][0]:.1f} perf={pred['performance'][0]:.3f}"
            )

            t0 = time.time()
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the question using the story and the conversation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=64,
            )
            latency_ms = (time.time() - t0) * 1000.0

            answer = (resp.choices[0].message.content or "").strip()
            total_tokens = int(resp.usage.total_tokens) if resp.usage else 0

            correct = normalize_eval(answer) == normalize_eval(gold_answer)

            print(
                f"  Observe latency_ms={latency_ms:.2f} cost_tokens={
                    total_tokens
                } correct={correct}"
            )
            if not correct:
                print(f"  gold={gold_answer!r}")
                print(f"  got ={answer!r}")

            predictor.update(
                inp,
                real_latency_ms=latency_ms,
                real_cost_tokens=total_tokens,
                real_perf_correct=correct,
            )

            # Update client-side history with the returned answer (so future prompts match).
            history.append((question, answer))

            # Update prefix cache for feature computation on the next turn.
            # Prompt ends with "A{turn}:", so we append a space + answer to create "A{turn}: <answer>".
            cache.update(
                model=model_name,
                dialogue_id=dialogue.dialogue_id,
                cached_text=prompt + " " + answer,
            )


if __name__ == "__main__":
    main()
