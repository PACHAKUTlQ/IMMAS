"""
Parallel CoQA client with:
- dialogue-level concurrency (turns sequential within dialogue)
- global semaphore limiting in-flight requests
- structured JSONL logging
- optional fetch of server debug trace to approximate TTFT from server fields
- compact, self-contained per-request printing (optional)

Env vars
--------
COQA_SPLIT              default "validation"
FAKE_MODEL_NAME         default "fake-coqa"
OPENAI_BASE_URL         default "http://localhost:8000/v1"
MAX_DIALOGUES           default "3"
MAX_TURNS               default "5"
MAX_CONCURRENCY         default "8"
VERBOSE                 default "0"   (print one line per completed request)
RUN_ID                  default auto timestamp
RUN_LOG_PATH            default "coqa_run.jsonl" (set "" to disable)
RUN_LOG_APPEND          default "0"
FETCH_SERVER_DEBUG      default "1"
SERVER_DEBUG_TIMEOUT_S  default "2.0"
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

from predictor.data.coqa.loader import COQA_DATASET_NAME, CoqaDialogue
from predictor.data.coqa.prompt import CoqaPromptFormatter
from predictor.sim.load import AsyncLoadTracker
from predictor.core.engine import AgentPredictor, PredictorInput
from predictor.core.prefix_cache import TextPrefixCache
from predictor.client.logger import AsyncJsonlLogger, RequestLogRecord


HistoryTurn = Tuple[str, str]


def normalize_eval(text: str) -> str:
    """Normalize text for simple equality checks."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def inline_for_cache(text: str) -> str:
    """Match formatter inline behavior so cached_text is a true prefix."""
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def short_dialogue_id(dialogue_id: str, n: int = 10) -> str:
    """Shorten long IDs for readable logs."""
    return dialogue_id if len(dialogue_id) <= n else dialogue_id[:n]


async def fetch_server_trace(
    *,
    http: httpx.AsyncClient,
    base_url_v1: str,
    completion_id: str,
    timeout_s: float,
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Fetch debug trace from server without affecting OpenAI response schema.

    Returns None if endpoint not present or trace not found (e.g., wrong worker).
    """
    url = f"{base_url_v1}/internal/chat_completions/{completion_id}"
    for attempt in range(max_retries + 1):
        try:
            r = await http.get(url, timeout=timeout_s)
            if r.status_code == 200:
                return dict(r.json())
            if r.status_code in (404, 501):
                # 404 can happen with multiple workers or quick eviction; retry briefly.
                if attempt < max_retries:
                    await asyncio.sleep(0.01 * (attempt + 1))
                    continue
                return None
            return None
        except httpx.HTTPError:
            return None
    return None


@dataclass(slots=True)
class GlobalStats:
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
    openai_client: AsyncOpenAI,
    openai_base_url_v1: str,
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
    logger: Optional[AsyncJsonlLogger],
    debug_http: Optional[httpx.AsyncClient],
    fetch_server_debug: bool,
    server_debug_timeout_s: float,
    run_id: str,
) -> None:
    """Run turns sequentially for one dialogue; allow parallelism across dialogues."""
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

        # Global concurrency cap for model calls
        async with send_sem:
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

                pred_latency_ms = float(pred["latency_ms"][0])
                pred_cost_tokens = float(pred["cost_tokens"][0])
                pred_perf_prob = float(pred["performance"][0])

                t0 = time.perf_counter()
                try:
                    resp = await openai_client.chat.completions.create(
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
                    err = f"{type(e).__name__}: {e}"
                    async with stats_lock:
                        stats.n_errors += 1
                    async with pbar_lock:
                        pbar.update(1)
                    if verbose:
                        async with print_lock:
                            tqdm.write(
                                f"ERROR did={short_dialogue_id(dialogue.dialogue_id)} "
                                f"turn={turn_number} inflight={
                                    load.inflight_requests
                                } rps={load.rps:.2f} err={err}"
                            )
                    if logger:
                        rec = RequestLogRecord(
                            run_id=run_id,
                            t_start_monotonic=t0,
                            t_end_monotonic=time.perf_counter(),
                            model=model_name,
                            source=dialogue.source,
                            dialogue_id=dialogue.dialogue_id,
                            turn_number=turn_number,
                            prompt_chars=len(prompt),
                            kvmatch=float(kvmatch),
                            client_inflight=int(load.inflight_requests),
                            client_rps_1s=float(load.rps),
                            pred_latency_ms=pred_latency_ms,
                            pred_cost_tokens=pred_cost_tokens,
                            pred_perf_prob=pred_perf_prob,
                            completion_id="",
                            obs_latency_ms=0.0,
                            obs_total_tokens=0,
                            correct=False,
                            error=err,
                        )
                        await logger.log(rec)
                    return  # abort dialogue on request error

                t1 = time.perf_counter()
                obs_latency_ms = (t1 - t0) * 1000.0

        # Outside send_sem: do extra debug fetch / updates without reducing load generation.
        completion_id = str(resp.id)
        answer = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        obs_total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        correct = normalize_eval(answer) == normalize_eval(gold_answer)

        # Optional: pull server-side simulated TTFT/total/utilization.
        srv = None
        if fetch_server_debug and debug_http is not None and completion_id:
            srv = await fetch_server_trace(
                http=debug_http,
                base_url_v1=openai_base_url_v1,
                completion_id=completion_id,
                timeout_s=server_debug_timeout_s,
            )

        async with predictor_lock:
            predictor.update(
                inp,
                real_latency_ms=float(obs_latency_ms),
                real_cost_tokens=int(obs_total_tokens),
                real_perf_correct=bool(correct),
            )

        history.append((question, answer))
        async with cache_lock:
            cache.update(
                model=model_name,
                dialogue_id=dialogue.dialogue_id,
                cached_text=prompt + " " + inline_for_cache(answer),
            )

        async with stats_lock:
            stats.n_requests += 1
            stats.n_correct += int(correct)
            stats.total_latency_ms += float(obs_latency_ms)
            stats.total_tokens += int(obs_total_tokens)

        async with pbar_lock:
            pbar.update(1)

        # Structured JSONL record
        if logger:
            sim = (srv or {}).get("sim") if isinstance(srv, dict) else None
            sim_load = (srv or {}).get("sim_load") if isinstance(srv, dict) else None
            rec = RequestLogRecord(
                run_id=run_id,
                t_start_monotonic=float(t0),
                t_end_monotonic=float(t1),
                model=model_name,
                source=dialogue.source,
                dialogue_id=dialogue.dialogue_id,
                turn_number=turn_number,
                prompt_chars=len(prompt),
                kvmatch=float(kvmatch),
                client_inflight=int(inp.client_inflight),
                client_rps_1s=float(inp.client_rps_1s),
                pred_latency_ms=pred_latency_ms,
                pred_cost_tokens=pred_cost_tokens,
                pred_perf_prob=pred_perf_prob,
                completion_id=completion_id,
                obs_latency_ms=float(obs_latency_ms),
                obs_total_tokens=int(obs_total_tokens),
                correct=bool(correct),
                srv_sim_ttft_s=float(sim.get("ttft_s"))
                if isinstance(sim, dict) and sim.get("ttft_s") is not None
                else None,
                srv_sim_total_s=float(sim.get("total_s"))
                if isinstance(sim, dict) and sim.get("total_s") is not None
                else None,
                srv_sim_stall_s=float(sim.get("stall_s"))
                if isinstance(sim, dict) and sim.get("stall_s") is not None
                else None,
                srv_utilization=float(sim_load.get("utilization"))
                if isinstance(sim_load, dict)
                and sim_load.get("utilization") is not None
                else None,
                srv_effective_inflight=int(sim_load.get("effective_inflight"))
                if isinstance(sim_load, dict)
                and sim_load.get("effective_inflight") is not None
                else None,
                srv_rps=float(sim_load.get("rps"))
                if isinstance(sim_load, dict) and sim_load.get("rps") is not None
                else None,
            )
            await logger.log(rec)

        # Self-contained, one-line log (no more misleading “multiple observations”)
        if verbose:
            srv_ttft_ms = (
                (rec.srv_sim_ttft_s * 1000.0)
                if rec.srv_sim_ttft_s is not None
                else None
            )
            srv_stall_ms = (
                (rec.srv_sim_stall_s * 1000.0)
                if rec.srv_sim_stall_s is not None
                else None
            )
            async with print_lock:
                tqdm.write(
                    "DONE "
                    f"did={short_dialogue_id(dialogue.dialogue_id)} turn={turn_number} "
                    f"inflight={inp.client_inflight} rps={inp.client_rps_1s:.2f} "
                    f"kvm={kvmatch:.3f} prompt_chars={len(prompt)} "
                    f"pred_lat_ms={pred_latency_ms:.1f} obs_lat_ms={
                        obs_latency_ms:.1f} "
                    f"pred_cost={pred_cost_tokens:.1f} obs_tok={obs_total_tokens} "
                    f"correct={int(correct)} "
                    + (
                        f"srv_u={rec.srv_utilization:.2f} srv_ttft_ms={
                            srv_ttft_ms:.1f} srv_stall_ms={srv_stall_ms:.1f}"
                        if rec.srv_utilization is not None
                        and srv_ttft_ms is not None
                        and srv_stall_ms is not None
                        else ""
                    )
                )


async def main_async() -> None:
    split = os.environ.get("COQA_SPLIT", "validation")
    model_name = os.environ.get("FAKE_MODEL_NAME", "fake-coqa")
    openai_base_url_v1 = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")

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

    run_id = os.environ.get("RUN_ID", "").strip() or time.strftime("coqa_%Y%m%d_%H%M%S")
    run_log_path = os.environ.get("RUN_LOG_PATH", "coqa_run.jsonl")
    run_log_append = os.environ.get("RUN_LOG_APPEND", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    fetch_server_debug = os.environ.get("FETCH_SERVER_DEBUG", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    server_debug_timeout_s = float(os.environ.get("SERVER_DEBUG_TIMEOUT_S", "2.0"))

    ds = load_dataset(COQA_DATASET_NAME, split=split)
    dialogues = []
    for i, ex in enumerate(ds):
        if i >= max_dialogues:
            break
        dialogues.append(CoqaDialogue.from_hf_example(ex))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(dialogues)

    predictor = AgentPredictor()
    predictor_lock = asyncio.Lock()

    cache = TextPrefixCache()
    cache_lock = asyncio.Lock()

    stats = GlobalStats()
    stats_lock = asyncio.Lock()

    client_load = AsyncLoadTracker(window_s=1.0)
    send_sem = asyncio.Semaphore(max_concurrency)

    total_requests = sum(min(d.num_turns(), max_turns_per_dialogue) for d in dialogues)
    pbar = tqdm(total=total_requests, desc="Requests", dynamic_ncols=True)
    pbar_lock = asyncio.Lock()
    print_lock = asyncio.Lock()

    print(
        f"Running split={split} dialogues={len(dialogues)} turns<= {
            max_turns_per_dialogue
        } "
        f"MAX_CONCURRENCY={max_concurrency} run_id={run_id}"
    )
    print(f"OpenAI base_url={openai_base_url_v1}")
    print("============================================================")

    openai_client = AsyncOpenAI(base_url=openai_base_url_v1, api_key="sk-local")

    debug_http = httpx.AsyncClient() if fetch_server_debug else None

    logger_cm = (
        AsyncJsonlLogger(run_log_path, append=run_log_append, flush_every=1)
        if run_log_path.strip()
        else None
    )

    try:
        if logger_cm is None:
            logger = None
            await _run_tasks(
                dialogues=dialogues,
                max_turns_per_dialogue=max_turns_per_dialogue,
                model_name=model_name,
                openai_client=openai_client,
                openai_base_url_v1=openai_base_url_v1,
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
                logger=logger,
                debug_http=debug_http,
                fetch_server_debug=fetch_server_debug,
                server_debug_timeout_s=server_debug_timeout_s,
                run_id=run_id,
            )
        else:
            async with logger_cm as logger:
                await _run_tasks(
                    dialogues=dialogues,
                    max_turns_per_dialogue=max_turns_per_dialogue,
                    model_name=model_name,
                    openai_client=openai_client,
                    openai_base_url_v1=openai_base_url_v1,
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
                    logger=logger,
                    debug_http=debug_http,
                    fetch_server_debug=fetch_server_debug,
                    server_debug_timeout_s=server_debug_timeout_s,
                    run_id=run_id,
                )
    finally:
        pbar.close()
        await openai_client.close()
        if debug_http is not None:
            await debug_http.aclose()

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
    if run_log_path.strip():
        print(f"Run log: {run_log_path} (run_id={run_id})")


async def _run_tasks(
    *,
    dialogues: List[CoqaDialogue],
    max_turns_per_dialogue: int,
    model_name: str,
    openai_client: AsyncOpenAI,
    openai_base_url_v1: str,
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
    logger: Optional[AsyncJsonlLogger],
    debug_http: Optional[httpx.AsyncClient],
    fetch_server_debug: bool,
    server_debug_timeout_s: float,
    run_id: str,
) -> None:
    tasks = [
        asyncio.create_task(
            run_dialogue(
                dialogue=d,
                max_turns=max_turns_per_dialogue,
                model_name=model_name,
                openai_client=openai_client,
                openai_base_url_v1=openai_base_url_v1,
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
                logger=logger,
                debug_http=debug_http,
                fetch_server_debug=fetch_server_debug,
                server_debug_timeout_s=server_debug_timeout_s,
                run_id=run_id,
            )
        )
        for d in dialogues
    ]
    await asyncio.gather(*tasks)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
