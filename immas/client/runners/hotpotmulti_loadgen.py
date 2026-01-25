"""
immas.client.runners.hotpotqa_loadgen

Multi-turn load generator for HotpotQA.
- Simulates a conversation by chaining related questions (synthesized dialogues).
- Maintains conversation history (system + user/assistant turns).
- Context is updated per-turn but history is preserved.
"""

from __future__ import annotations

import asyncio
import os
import random
import time

from dataclasses import dataclass
from typing import Any, Dict, List

from openai import AsyncOpenAI
from tqdm import tqdm

from immas.data.hotpotqa.loader import (
    HotpotQADialogue,
    HotpotQADatasetIndex,
    HotpotQATurn,
    DEFAULT_SYNTHETIC_DIALOGUE_SIZE
)


def short_id(did: str, n: int = 12) -> str:
    return did if len(did) <= n else did[:n]


@dataclass(slots=True)
class GlobalStats:
    n_requests: int = 0
    n_errors: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0


def _make_initial_system_message() -> Dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are a knowledgeable assistant. You will be presented with a series of questions. "
            "For each question, a specific context (Wikipedia paragraphs) will be provided. "
            "Answer the question based strictly on the provided context. "
            "You should remember the history of our conversation to answer follow-up questions if necessary."
        ),
    }


async def run_dialogue(
    *,
    dialogue: HotpotQADialogue,
    max_turns: int,
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
    Run a single synthesized dialogue sequentially.
    """
    # 1. Initialize conversation history
    messages: List[Dict[str, Any]] = [_make_initial_system_message()]
    
    # 2. Determine actual turns to run
    n_turns = min(dialogue.num_turns(), max_turns)

    for turn_idx in range(n_turns):
        turn_obj: HotpotQATurn = dialogue.turns[turn_idx]
        turn_number = turn_idx + 1

        # 3. Construct the prompt for this turn
        # We explicitly label the Context and Question.
        formatted_context = turn_obj.get_formatted_context()
        user_content = (
            f"Context (Topic: {dialogue.primary_topic}):\n{formatted_context}\n\n"
            f"Question: {turn_obj.question}"
        )
        
        # Append User message
        messages.append({"role": "user", "content": user_content})

        # 4. Send request
        async with send_sem:
            t0 = time.monotonic()
            try:
                resp = await openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=128,  # HotpotQA answers are usually short spans
                    extra_headers={
                        "X-IMMAS-RUN-ID": run_id,
                        "X-IMMAS-DIALOGUE-ID": dialogue.dialogue_id,
                        "X-IMMAS-TURN-NUMBER": str(turn_number),
                        "X-IMMAS-SOURCE": "hotpotqa_synthetic",
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
                            f"ERROR did={short_id(dialogue.dialogue_id)} "
                            f"turn={turn_number} err={type(e).__name__}: {e}"
                        )
                # Break the dialogue on error to maintain state consistency
                return

            t1 = time.monotonic()

        obs_latency_ms = (t1 - t0) * 1000.0
        answer = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        obs_total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        # 5. Append Assistant message to history (Accumulate Context)
        messages.append({"role": "assistant", "content": answer})

        # 6. Update stats
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
                    f"did={short_id(dialogue.dialogue_id)} turn={turn_number} "
                    f"obs_latency_ms={obs_latency_ms:.1f} obs_tok={obs_total_tokens}"
                )


async def main_async() -> None:
    # Env vars
    split = os.environ.get("HOTPOTQA_SPLIT", "validation")
    model_name = os.environ.get("MODEL_NAME", "fake-hotpotqa")
    openai_base_url_v1 = os.environ.get("OPENAI_BASE_URL", "http://localhost:9000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-local")

    # Configs
    max_dialogues = int(os.environ.get("MAX_DIALOGUES", "3"))
    max_turns_per_dialogue = int(os.environ.get("MAX_TURNS", str(DEFAULT_SYNTHETIC_DIALOGUE_SIZE)))
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "8"))
    
    if max_concurrency < 1:
        raise ValueError(f"MAX_CONCURRENCY must be >= 1, got {max_concurrency}")

    verbose = os.environ.get("VERBOSE", "0").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    shuffle = os.environ.get("SHUFFLE", "0").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    seed = int(os.environ.get("SEED", "0"))

    run_id = os.environ.get("RUN_ID", "").strip() or time.strftime("hotpotqa_multi_%Y%m%d_%H%M%S")

    print(f"Loading HotpotQA (split={split})... This may take a moment to group by topic.")
    
    # Use Loader to group/chunk dialogues
    # Note: We pass max_turns here to affect how the Loader chunks the raw data
    index = HotpotQADatasetIndex.from_hf(
        split=split, 
        max_turns_per_dialogue=max_turns_per_dialogue
    )
    
    all_dialogues = list(index.iter_dialogues())
    
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_dialogues)

    # Slice strictly
    target_dialogues = all_dialogues[:max_dialogues]

    stats = GlobalStats()
    stats_lock = asyncio.Lock()
    send_sem = asyncio.Semaphore(max_concurrency)

    # Total requests = sum of turns in selected dialogues
    # (Use min() to handle case where dialogue has fewer turns than max)
    total_requests = sum(min(d.num_turns(), max_turns_per_dialogue) for d in target_dialogues)
    
    pbar = tqdm(total=total_requests, desc="Requests", dynamic_ncols=True)
    pbar_lock = asyncio.Lock()
    print_lock = asyncio.Lock()

    print(
        f"Loadgen Plan: {len(target_dialogues)} dialogues, "
        f"concurrency={max_concurrency}, run_id={run_id}"
    )
    print(f"Router: {openai_base_url_v1}")
    print("============================================================")

    openai_client = AsyncOpenAI(base_url=openai_base_url_v1, api_key=api_key)

    try:
        tasks = [
            asyncio.create_task(
                run_dialogue(
                    dialogue=d,
                    max_turns=max_turns_per_dialogue,
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
            for d in target_dialogues
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