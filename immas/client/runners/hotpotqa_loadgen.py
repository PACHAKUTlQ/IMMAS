"""
immas.client.runners.hotpotqa_loadgen

Pure load generator for HotpotQA:
- Request-level concurrency (since HotpotQA is single-turn)
- Calls the router's /v1/chat/completions
- Formats context from multiple documents into a single prompt
"""

from __future__ import annotations

import asyncio
import os
import random
import time

from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

from immas.data.hotpotqa.loader import (
    HOTPOTQA_DATASET_NAME,
    HOTPOTQA_CONFIG_NAME,
    HotpotQAExample,
)


def short_id(ex_id: str, n: int = 10) -> str:
    return ex_id if len(ex_id) <= n else ex_id[:n]


@dataclass(slots=True)
class GlobalStats:
    n_requests: int = 0
    n_errors: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0


def _make_messages(example: HotpotQAExample) -> List[Dict[str, Any]]:
    """
    Construct the messages for a single-turn HotpotQA request.
    Includes System Prompt and User Prompt (Context + Question).
    """
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant. Answer the question based strictly on the provided "
            "context paragraphs. If the context does not contain the answer, state that you do not know. "
            "Think carefully before answering."
        ),
    }
    
    formatted_context = example.get_formatted_context()
    user_content = (
        f"Context:\n{formatted_context}\n\n"
        f"Question: {example.question}"
    )
    
    user_msg = {
        "role": "user",
        "content": user_content,
    }
    return [system, user_msg]


async def run_example(
    *,
    example: HotpotQAExample,
    model_name: str,
    openai_client: AsyncOpenAI,
    send_sem: asyncio.Semaphore,
    stats: GlobalStats,
    stats_lock: asyncio.Lock,
    pbar: tqdm,
    pbar_lock: asyncio.Lock,
    verbose: bool,
    print_lock: asyncio.Lock,
    run_id: str,
) -> None:
    """
    Execute a single HotpotQA test case (single turn).
    """
    messages = _make_messages(example)
    
    # HotpotQA is 1-turn, so we hardcode turn number to 1
    turn_number = 1

    async with send_sem:
        t0 = time.monotonic()
        try:
            resp = await openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=64,
                extra_headers={
                    "X-IMMAS-RUN-ID": run_id,
                    "X-IMMAS-DIALOGUE-ID": example.id,
                    "X-IMMAS-TURN-NUMBER": str(turn_number),
                    "X-IMMAS-SOURCE": "hotpotqa",
                },
            )
        except Exception as e:
            async with stats_lock:
                stats.n_errors += 1
            async with pbar_lock:
                pbar.update(1)
            if verbose:
                async with print_lock:
                    tqdm.write(
                        f"ERROR id={short_id(example.id)} "
                        f"err={type(e).__name__}: {e}"
                    )
            return

        t1 = time.monotonic()

    obs_latency_ms = (t1 - t0) * 1000.0
    # No answer appending logic needed for history since it's single turn,
    # but we extract metrics similarly.
    answer = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    obs_total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    async with stats_lock:
        stats.n_requests += 1
        stats.total_latency_ms += float(obs_latency_ms)
        stats.total_tokens += int(obs_total_tokens)

    async with pbar_lock:
        pbar.update(1)

    if verbose:
        async with print_lock:
            tqdm.write(
                "DONE "
                f"id={short_id(example.id)} "
                f"obs_latency_ms={obs_latency_ms:.1f} obs_tok={obs_total_tokens}"
            )


async def main_async() -> None:
    # Env vars
    split = os.environ.get("HOTPOTQA_SPLIT", "validation")
    model_name = os.environ.get("MODEL_NAME", "fake-hotpotqa")
    openai_base_url_v1 = os.environ.get("OPENAI_BASE_URL", "http://localhost:9000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-local")

    max_examples = int(os.environ.get("MAX_EXAMPLES", "10"))
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "8"))
    
    if max_concurrency < 1:
        raise ValueError(f"MAX_CONCURRENCY must be >= 1, got {max_concurrency}")

    verbose = os.environ.get("VERBOSE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    shuffle = os.environ.get("SHUFFLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    seed = int(os.environ.get("SEED", "0"))

    run_id = os.environ.get("RUN_ID", "").strip() or time.strftime("hotpotqa_%Y%m%d_%H%M%S")

    # Load dataset
    ds = load_dataset(HOTPOTQA_DATASET_NAME, HOTPOTQA_CONFIG_NAME, split=split)
    examples: List[HotpotQAExample] = []
    
    # Simple iterator
    for i, ex in enumerate(ds):
        if i >= max_examples:
            break
        # Parsing safety
        try:
            item = HotpotQAExample.from_hf_example(ex)
            examples.append(item)
        except Exception as e:
            if verbose:
                print(f"Skipping example index {i} due to error: {e}")
            continue

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(examples)

    stats = GlobalStats()
    stats_lock = asyncio.Lock()

    send_sem = asyncio.Semaphore(max_concurrency)

    # HotpotQA is 1 request per example
    total_requests = len(examples)
    pbar = tqdm(total=total_requests, desc="Requests", dynamic_ncols=True)
    pbar_lock = asyncio.Lock()
    print_lock = asyncio.Lock()

    print(
        f"Loadgen split={split} examples={len(examples)} "
        f"MAX_CONCURRENCY={max_concurrency} run_id={run_id}"
    )
    print(f"Router base_url={openai_base_url_v1}")
    print("============================================================")

    openai_client = AsyncOpenAI(base_url=openai_base_url_v1, api_key=api_key)

    try:
        tasks = [
            asyncio.create_task(
                run_example(
                    example=ex,
                    model_name=model_name,
                    openai_client=openai_client,
                    send_sem=send_sem,
                    stats=stats,
                    stats_lock=stats_lock,
                    pbar=pbar,
                    pbar_lock=pbar_lock,
                    verbose=verbose,
                    print_lock=print_lock,
                    run_id=run_id,
                )
            )
            for ex in examples
        ]
        await asyncio.gather(*tasks)
    finally:
        pbar.close()
        await openai_client.close()

    avg_latency = (
        (stats.total_latency_ms / stats.n_requests) if stats.n_requests else 0.0
    )
    avg_tokens = (stats.total_tokens / stats.n_requests) if stats.n_requests else 0.0

    print("\nSummary")
    print("-------")
    print(f"Requests: {stats.n_requests}   Errors: {stats.n_errors}")
    print(f"Avg latency (ms): {avg_latency:.2f}")
    print(f"Avg total tokens: {avg_tokens:.1f}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()