"""
immas.router.warmup

Startup warmup for backends and online predictors using real dataset dialogues.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid

from dataclasses import dataclass
from typing import Any, Mapping

from datasets import load_dataset

from immas.data.coqa.loader import COQA_DATASET_NAME, CoqaDialogue
from immas.openai.chat import extract_first_assistant_message, serialize_chat_messages
from immas.openai.usage import parse_usage
from immas.router.components.performance import PerformanceEvalContext
from immas.router.components.predictor import PredictorInput
from immas.router.components.prefix_cache import match_prefix
from immas.router.state import RouterState

_log = logging.getLogger(__name__)

_SYSTEM_BASE = (
    "Answer the user's questions using the story. Be factual. Think very carefully "
    "and show your thinking steps, and then output the answer at the last line, "
    "below your thinking."
)

_WARMUP_TRAIN_LATENCY_MS: float = 250.0


@dataclass(frozen=True, slots=True)
class WarmupStats:
    backend_id: str
    ok: int = 0
    err: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    def add_ok(self, *, latency_ms: float, total_tokens: int) -> "WarmupStats":
        return WarmupStats(
            backend_id=self.backend_id,
            ok=self.ok + 1,
            err=self.err,
            total_latency_ms=self.total_latency_ms + float(latency_ms),
            total_tokens=self.total_tokens + int(total_tokens),
        )

    def add_err(self) -> "WarmupStats":
        return WarmupStats(
            backend_id=self.backend_id,
            ok=self.ok,
            err=self.err + 1,
            total_latency_ms=self.total_latency_ms,
            total_tokens=self.total_tokens,
        )


def _initial_messages(
    dialogue: CoqaDialogue, *, system_text: str
) -> list[dict[str, Any]]:
    story_msg = {
        "role": "user",
        "content": f"Story (source={dialogue.source}, id={dialogue.dialogue_id}):\n{dialogue.story}",
    }
    return [{"role": "system", "content": system_text}, story_msg]


def _load_dialogues(
    *, split: str, max_dialogues: int, shuffle: bool, seed: int
) -> list[CoqaDialogue]:
    ds = load_dataset(COQA_DATASET_NAME, split=split)
    dialogues: list[CoqaDialogue] = []
    for i, ex in enumerate(ds):
        if i >= max_dialogues:
            break
        dialogues.append(CoqaDialogue.from_hf_example(ex))

    if shuffle and dialogues:
        rng = random.Random(int(seed))
        rng.shuffle(dialogues)

    return dialogues


async def warmup_router(state: RouterState) -> None:
    """
    Warm up backends and bootstrap predictors using dataset-based multi-turn chats.

    Warmup requests are not logged and do not update the router prefix cache.

    Important
    ---------
    The predictor update uses a fixed latency label (same for all backends) to
    avoid contaminating the model with backend "first request" latency anomalies.
    """
    cfg = state.cfg.router.warmup
    if not cfg.enabled:
        return
    if cfg.max_dialogues <= 0 or cfg.max_turns_per_dialogue <= 0:
        return
    if not state.backends:
        return

    nonce = uuid.uuid4().hex[:12]
    system_text = f"{cfg.system_prefix} {nonce}\n{_SYSTEM_BASE}"

    try:
        dialogues = _load_dialogues(
            split=str(cfg.coqa_split or "validation"),
            max_dialogues=int(cfg.max_dialogues),
            shuffle=bool(cfg.shuffle_dialogues),
            seed=int(cfg.seed),
        )
    except Exception:
        _log.exception("Warmup dataset load failed; skipping warmup")
        return

    if not dialogues:
        return

    global_sem = asyncio.Semaphore(int(cfg.max_concurrency))

    # Local warmup-only prefix state (does not touch router prefix cache).
    cached_prompt_by_backend_dialogue: dict[tuple[str, str], str] = {}

    stats_by_backend: dict[str, WarmupStats] = {
        b.backend_id: WarmupStats(backend_id=b.backend_id) for b in state.backends
    }

    async def _run_dialogue_on_backend(
        *, backend_id: str, dialogue: CoqaDialogue, run_id: str
    ) -> None:
        backend = next((b for b in state.backends if b.backend_id == backend_id), None)
        if backend is None:
            return
        model = state.backend_model_by_id.get(backend_id, "")
        if not model:
            return

        messages: list[dict[str, Any]] = _initial_messages(
            dialogue, system_text=system_text
        )
        n_turns = min(dialogue.num_turns(), int(cfg.max_turns_per_dialogue))

        for turn_idx in range(n_turns):
            turn_number = turn_idx + 1
            question = dialogue.questions[turn_idx]
            messages.append({"role": "user", "content": f"Q{turn_number}: {question}"})

            prompt_repr = serialize_chat_messages(messages)

            cache_key = (backend_id, dialogue.dialogue_id)
            cached_text = cached_prompt_by_backend_dialogue.get(cache_key)
            pm = match_prefix(prompt_text=prompt_repr, cached_text=cached_text)

            inp = PredictorInput(
                backend_id=backend_id,
                model=model,
                source=dialogue.source,
                dialogue_id=dialogue.dialogue_id,
                turn_number=int(turn_number),
                prompt_repr=prompt_repr,
                kvmatch_text=float(pm.ratio),
                router_inflight=0,
                router_rps_1s=0.0,
                backend_inflight=0,
                backend_rps_1s=0.0,
                backend_capacity=max(
                    1, int(state.backend_capacity_by_id.get(backend_id, 1))
                ),
            )

            body: dict[str, Any] = {
                "model": model,
                "messages": list(messages),
                "temperature": 0,
                "max_tokens": int(cfg.max_tokens),
                "stream": False,
            }

            headers: Mapping[str, str] = {
                "X-IMMAS-RUN-ID": run_id,
                "X-IMMAS-DIALOGUE-ID": dialogue.dialogue_id,
                "X-IMMAS-TURN-NUMBER": str(turn_number),
                "X-IMMAS-SOURCE": dialogue.source,
            }

            async with global_sem:
                try:
                    backend_sem = state.backend_semaphores.get(
                        backend_id
                    ) or asyncio.Semaphore(1)
                    async with backend_sem:
                        _t0 = time.monotonic()
                        status, resp_json = await asyncio.wait_for(
                            backend.forward_chat_completions(body, headers=headers),
                            timeout=float(cfg.timeout_s),
                        )
                        _t1 = time.monotonic()
                        _ = _t1 - _t0
                except asyncio.TimeoutError:
                    stats_by_backend[backend_id] = stats_by_backend[
                        backend_id
                    ].add_err()
                    return
                except Exception:
                    stats_by_backend[backend_id] = stats_by_backend[
                        backend_id
                    ].add_err()
                    return

            if not (200 <= int(status) < 300) or not isinstance(resp_json, Mapping):
                stats_by_backend[backend_id] = stats_by_backend[backend_id].add_err()
                return

            usage = parse_usage(resp_json)
            obs_total_tokens = int(usage.total_tokens)

            ctx = PerformanceEvalContext(
                run_id=run_id,
                dialogue_id=dialogue.dialogue_id,
                turn_number=int(turn_number),
                source=dialogue.source,
                request_body=body,
                response_json=resp_json,
            )
            correct = bool(state.perf_evaluator.evaluate(ctx))

            await state.predictors.update_one(
                inp,
                real_latency_ms=float(_WARMUP_TRAIN_LATENCY_MS),
                real_cost_tokens=int(obs_total_tokens),
                real_perf_correct=bool(correct),
            )

            assistant = extract_first_assistant_message(resp_json)
            if assistant is None:
                stats_by_backend[backend_id] = stats_by_backend[backend_id].add_err()
                return

            # Multi-turn continuation uses the actual returned assistant message.
            messages.append(assistant.to_openai_message())
            cached_prompt_by_backend_dialogue[cache_key] = serialize_chat_messages(
                messages
            )

            stats_by_backend[backend_id] = stats_by_backend[backend_id].add_ok(
                latency_ms=float(_WARMUP_TRAIN_LATENCY_MS),
                total_tokens=int(obs_total_tokens),
            )

    run_id = f"warmup_{nonce}"
    tasks: list[asyncio.Task[None]] = []
    for b in state.backends:
        for d in dialogues:
            tasks.append(
                asyncio.create_task(
                    _run_dialogue_on_backend(
                        backend_id=b.backend_id, dialogue=d, run_id=run_id
                    )
                )
            )

    await asyncio.gather(*tasks, return_exceptions=True)

    # Summary logging only (no JSONL writes).
    parts: list[str] = []
    for bid in sorted(stats_by_backend.keys()):
        st = stats_by_backend[bid]
        n = st.ok
        if n > 0:
            avg_lat = st.total_latency_ms / float(n)
            avg_tok = st.total_tokens / float(n)
            parts.append(
                f"{bid}: ok={st.ok} err={st.err} avg_lat_ms={avg_lat:.1f} avg_tok={
                    avg_tok:.1f}"
            )
        else:
            parts.append(f"{bid}: ok=0 err={st.err}")
    _log.info("Warmup complete (%s)", "; ".join(parts))
