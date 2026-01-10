"""
Parallel client demo for the fake OpenAI-compatible CoQA server.

Key behavior
------------
- Runs multiple CoQA dialogues concurrently (parallel requests).
- Keeps each individual dialogue sequential across turns (history dependency).
- Uses an asyncio.Semaphore to cap in-flight requests (MAX_CONCURRENCY).
- Updates the River online predictor after each request.
- Computes text prefix match ratio (kvmatch) as before.

Environment variables
---------------------
COQA_SPLIT            (default: "validation")
FAKE_MODEL_NAME       (default: "fake-coqa")
MAX_DIALOGUES         (default: "3")
MAX_TURNS             (default: "5")
MAX_CONCURRENCY       (default: "8")   # in-flight HTTP requests
VERBOSE               (default: "0")   # set to "1" to print per-request details
SHUFFLE_DIALOGUES     (default: "0")   # set to "1" to shuffle the selected dialogues
SEED                  (default: "0")   # shuffle seed

Notes
-----
- Requires openai>=1.x (AsyncOpenAI). If you hit import issues, tell me your openai
  version and I'll adapt it (threadpool fallback).
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

from coqa_data import COQA_DATASET_NAME, CoqaDialogue
from coqa_prompt import CoqaPromptFormatter
from load_tracker import AsyncLoadTracker
from predictor import AgentPredictor, PredictorInput
from prefix_cache import TextPrefixCache


HistoryTurn = Tuple[str, str]


def normalize_eval(text: str) -> str:
    """Normalize text for simple equality checks."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def inline_for_cache(text: str) -> str:
    """
    Match the formatter's "inline" behavior so cached_text is a true prefix
    of the next prompt even if the answer contains newlines.
    """
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


@dataclass(slots=True)
class GlobalStats:
    """Aggregate stats across all requests."""

    n_requests: int = 0
    n_correct: int = 0
    n_errors: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0


async def run_dialogue(
    *,
    dialogue: CoqaDialogue,
    max_turns: int,
    model_name: str,
    client: AsyncOpenAI,
    send_sem: asyncio.Semaphore,
    client_load: AsyncLoadTracker,
    predictor: AgentPredictor,
    predictor_lock: asyncio.Lock,
    cache: TextPrefixCache,
    cache_lock: asyncio.Lock,
    stats: GlobalStats,
    stats_lock: asyncio.Lock,
    pbar: tqdm,
    pbar_lock: asyncio.Lock,
    verbose: bool,
    print_lock: asyncio.Lock,
) -> None:
    """
    Run one dialogue sequentially across turns, while allowing parallelism across dialogues.
    """
    history: List[HistoryTurn] = []
    n_turns = min(dialogue.num_turns(), max_turns)

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

        async with cache_lock:
            kvmatch = cache.match_ratio(
                model=model_name, dialogue_id=dialogue.dialogue_id, prompt_text=prompt
            )

        # Cap concurrent in-flight requests globally.
        async with send_sem:
            # Track client-side in-flight and RPS for features.
            async with client_load.track() as load:
                inp = PredictorInput(
                    model=model_name,
                    source=dialogue.source,
                    dialogue_id=dialogue.dialogue_id,
                    turn_number=turn_number,
                    prompt_text=prompt,
                    kvmatch=kvmatch,
                    client_inflight=load.inflight_requests,
                    client_rps_1s=load.rps,
                )

                async with predictor_lock:
                    pred = predictor.predict(inp)

                if verbose:
                    async with print_lock:
                        tqdm.write(
                            f"\nDialogue={dialogue.dialogue_id} source={
                                dialogue.source
                            } turn={turn_number}\n"
                            f"  Feature kvmatch={kvmatch:.3f} prompt_chars={
                                len(prompt)
                            } "
                            f"client_inflight={load.inflight_requests} client_rps_1s={
                                load.rps:.2f}\n"
                            f"  Predict latency_ms={pred['latency_ms'][0]:.2f} "
                            f"cost_tokens={pred['cost_tokens'][0]:.1f} "
                            f"perf={pred['performance'][0]:.3f}"
                        )

                t0 = time.perf_counter()
                try:
                    resp = await client.chat.completions.create(
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
                except Exception as e:
                    async with stats_lock:
                        stats.n_errors += 1
                    async with pbar_lock:
                        pbar.update(1)
                    if verbose:
                        async with print_lock:
                            tqdm.write(
                                f"  ERROR dialogue={dialogue.dialogue_id} turn={
                                    turn_number
                                }: {type(e).__name__}: {e}"
                            )
                    # Abort this dialogue on error.
                    return

                latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        correct = normalize_eval(answer) == normalize_eval(gold_answer)

        async with predictor_lock:
            predictor.update(
                inp,
                real_latency_ms=float(latency_ms),
                real_cost_tokens=int(total_tokens),
                real_perf_correct=bool(correct),
            )

        # Update conversation state for the next turn.
        history.append((question, answer))

        # Update prefix cache so next prompt is a strict prefix extension.
        cache_text = prompt + " " + inline_for_cache(answer)
        async with cache_lock:
            cache.update(
                model=model_name,
                dialogue_id=dialogue.dialogue_id,
                cached_text=cache_text,
            )

        async with stats_lock:
            stats.n_requests += 1
            stats.n_correct += int(correct)
            stats.total_latency_ms += float(latency_ms)
            stats.total_tokens += int(total_tokens)

        if verbose:
            async with print_lock:
                tqdm.write(
                    f"  Observe latency_ms={latency_ms:.2f} cost_tokens={
                        total_tokens
                    } correct={correct}"
                )
                if not correct:
                    tqdm.write(f"  gold={gold_answer!r}")
                    tqdm.write(f"  got ={answer!r}")

        async with pbar_lock:
            pbar.update(1)


async def main_async() -> None:
    split = os.environ.get("COQA_SPLIT", "validation")
    model_name = os.environ.get("FAKE_MODEL_NAME", "fake-coqa")

    max_dialogues = int(os.environ.get("MAX_DIALOGUES", "3"))
    max_turns_per_dialogue = int(os.environ.get("MAX_TURNS", "5"))
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "8"))

    verbose = os.environ.get("VERBOSE", "0").strip() in {"1", "true", "yes", "y", "on"}
    shuffle = os.environ.get("SHUFFLE_DIALOGUES", "0").strip() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    seed = int(os.environ.get("SEED", "0"))

    if max_concurrency < 1:
        raise ValueError(f"MAX_CONCURRENCY must be >= 1, got {max_concurrency}")

    # Load dataset (sync, once).
    ds = load_dataset(COQA_DATASET_NAME, split=split)
    n = min(max_dialogues, len(ds))
    examples = list(ds.select(range(n)))
    dialogues = [CoqaDialogue.from_hf_example(ex) for ex in examples]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(dialogues)

    # Shared state (guarded by locks for safety under concurrency).
    predictor = AgentPredictor()
    predictor_lock = asyncio.Lock()

    cache = TextPrefixCache()
    cache_lock = asyncio.Lock()

    stats = GlobalStats()
    stats_lock = asyncio.Lock()

    # Client-side load estimate (for predictor features).
    client_load = AsyncLoadTracker(window_s=1.0)

    # Global request concurrency cap.
    send_sem = asyncio.Semaphore(max_concurrency)

    # Progress + printing.
    total_requests = sum(min(d.num_turns(), max_turns_per_dialogue) for d in dialogues)
    pbar = tqdm(total=total_requests, desc="Requests", dynamic_ncols=True)
    pbar_lock = asyncio.Lock()
    print_lock = asyncio.Lock()

    print(
        f"Running CoQA split={split} dialogues={len(dialogues)} turns/dialogue<= {
            max_turns_per_dialogue
        } "
        f"MAX_CONCURRENCY={max_concurrency}"
    )
    print("============================================================")

    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")
    try:
        tasks = [
            asyncio.create_task(
                run_dialogue(
                    dialogue=d,
                    max_turns=max_turns_per_dialogue,
                    model_name=model_name,
                    client=client,
                    send_sem=send_sem,
                    client_load=client_load,
                    predictor=predictor,
                    predictor_lock=predictor_lock,
                    cache=cache,
                    cache_lock=cache_lock,
                    stats=stats,
                    stats_lock=stats_lock,
                    pbar=pbar,
                    pbar_lock=pbar_lock,
                    verbose=verbose,
                    print_lock=print_lock,
                )
            )
            for d in dialogues
        ]

        # Let all dialogues run; errors are handled per-dialogue.
        await asyncio.gather(*tasks)
    finally:
        pbar.close()
        await client.close()

    # Summary
    avg_latency = (
        (stats.total_latency_ms / stats.n_requests) if stats.n_requests else 0.0
    )
    acc = (stats.n_correct / stats.n_requests) if stats.n_requests else 0.0
    avg_tokens = (stats.total_tokens / stats.n_requests) if stats.n_requests else 0.0

    print("\nSummary")
    print("-------")
    print(f"Requests: {stats.n_requests}   Errors: {stats.n_errors}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Avg latency (ms): {avg_latency:.2f}")
    print(f"Avg total tokens: {avg_tokens:.1f}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
