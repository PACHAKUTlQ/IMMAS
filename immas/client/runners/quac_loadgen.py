"""
immas.client.runners.quac_loadgen

Pure load generator for QuAC:
- dialogue-level concurrency
- sequential turns within a dialogue
- calls the router's /v1/chat/completions
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

# Assumes the loader file is placed at immas.data.quac.loader
from immas.data.quac.loader import QUAC_DATASET_NAME, QuacDialogue


def short_dialogue_id(dialogue_id: str, n: int = 10) -> str:
    return dialogue_id if len(dialogue_id) <= n else dialogue_id[:n]


@dataclass(slots=True)
class GlobalStats:
    n_requests: int = 0
    n_errors: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0


def _make_initial_messages(dialogue: QuacDialogue) -> List[Dict[str, Any]]:
    # QuAC prompts benefit from including the background and section title.
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant participating in an information-seeking conversation. "
            "Answer the user's questions based strictly on the provided context text. "
            "If the answer cannot be found in the context, reply with 'CANNOTANSWER'."
        ),
    }
    
    # Construct a structured context block
    context_text = dialogue.get_full_context_text()
    
    context_msg = {
        "role": "user",
        "content": f"{context_text}\n\n(End of Context. I will now ask questions about this text.)",
    }
    return [system, context_msg]


async def run_dialogue(
    *,
    dialogue: QuacDialogue,
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
    messages: List[Dict[str, Any]] = _make_initial_messages(dialogue)
    n_turns = min(dialogue.num_turns(), max_turns)

    for turn_idx in range(n_turns):
        turn_number = turn_idx + 1
        question = dialogue.questions[turn_idx]

        # Append the new user turn
        messages.append({"role": "user", "content": f"Q{turn_number}: {question}"})

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
                        "X-IMMAS-DIALOGUE-ID": dialogue.dialogue_id,
                        "X-IMMAS-TURN-NUMBER": str(turn_number),
                        "X-IMMAS-SOURCE": dialogue.source,
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
                            f"ERROR did={short_dialogue_id(dialogue.dialogue_id)} "
                            f"turn={turn_number} err={type(e).__name__}: {e}"
                        )
                return

            t1 = time.monotonic()

        obs_latency_ms = (t1 - t0) * 1000.0
        answer = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        obs_total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        # Append assistant turn verbatim
        messages.append({"role": "assistant", "content": answer})

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
                    f"did={short_dialogue_id(dialogue.dialogue_id)} turn={turn_number} "
                    f"obs_latency_ms={obs_latency_ms:.1f} obs_tok={obs_total_tokens}"
                )


async def main_async() -> None:
    split = os.environ.get("QUAC_SPLIT", "validation")
    model_name = os.environ.get("MODEL_NAME", "fake-quac")
    openai_base_url_v1 = os.environ.get("OPENAI_BASE_URL", "http://localhost:9000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-local")

    max_dialogues = int(os.environ.get("MAX_DIALOGUES", "3"))
    max_turns_per_dialogue = int(os.environ.get("MAX_TURNS", "5"))
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
    shuffle = os.environ.get("SHUFFLE_DIALOGUES", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    seed = int(os.environ.get("SEED", "0"))

    run_id = os.environ.get("RUN_ID", "").strip() or time.strftime("quac_%Y%m%d_%H%M%S")

    # Load QuAC dataset
    ds = load_dataset(QUAC_DATASET_NAME, split=split)
    dialogues: List[QuacDialogue] = []
    
    # Iterate and convert, skipping malformed ones if any (though strict parser raises)
    count = 0
    for ex in ds:
        if count >= max_dialogues:
            break
        # Sometimes datasets have filtering/preprocessing, simple iteration here
        try:
            d = QuacDialogue.from_hf_example(ex)
            dialogues.append(d)
            count += 1
        except Exception as e:
            if verbose:
                print(f"Skipping a dialogue due to error: {e}")
            continue

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(dialogues)

    stats = GlobalStats()
    stats_lock = asyncio.Lock()

    send_sem = asyncio.Semaphore(max_concurrency)

    total_requests = sum(min(d.num_turns(), max_turns_per_dialogue) for d in dialogues)
    pbar = tqdm(total=total_requests, desc="Requests", dynamic_ncols=True)
    pbar_lock = asyncio.Lock()
    print_lock = asyncio.Lock()

    print(
        f"Loadgen split={split} dialogues={len(dialogues)} turns<={
            max_turns_per_dialogue
        } "
        f"MAX_CONCURRENCY={max_concurrency} run_id={run_id}"
    )
    print(f"Router base_url={openai_base_url_v1}")
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
            for d in dialogues
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