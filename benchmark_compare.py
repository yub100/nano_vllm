import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep


REPO_ROOT = Path(__file__).resolve().parent
ORIGINAL_ROOT = (REPO_ROOT / "../nano-vllm").resolve()
RESULT_PREFIX = "JSON_RESULT="


@dataclass
class RequestSpec:
    prompt_token_ids: list[int]
    max_new_tokens: int
    group: str
    arrival_s: float = 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def format_stat(value: float | None, unit: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}{unit}"


def speedup_pct(baseline: float | None, candidate: float | None, lower_is_better: bool = False) -> float | None:
    if baseline in (None, 0) or candidate is None:
        return None
    if lower_is_better:
        return (baseline - candidate) / baseline * 100
    return (candidate - baseline) / baseline * 100


def make_request_factory(args: argparse.Namespace, rng: random.Random):
    def make_request(prompt_len: int, group: str, arrival_s: float = 0.0) -> RequestSpec:
        output_len = rng.randint(args.min_output_len, args.max_output_len)
        prompt = [rng.randint(0, args.vocab_upper_bound) for _ in range(prompt_len)]
        return RequestSpec(prompt_token_ids=prompt, max_new_tokens=output_len, group=group, arrival_s=arrival_s)

    return make_request


def build_batch_requests(args: argparse.Namespace, rng: random.Random) -> list[RequestSpec]:
    make_request = make_request_factory(args, rng)
    requests: list[RequestSpec] = []

    if args.workload == "uniform":
        for _ in range(args.num_seqs):
            prompt_len = rng.randint(args.min_input_len, args.max_input_len)
            requests.append(make_request(prompt_len, "all"))
        return requests

    short_count = args.num_seqs // 2
    long_count = args.num_seqs - short_count
    short_requests = [
        make_request(rng.randint(args.short_input_len_min, args.short_input_len_max), "short")
        for _ in range(short_count)
    ]
    long_requests = [
        make_request(rng.randint(args.long_input_len_min, args.long_input_len_max), "long")
        for _ in range(long_count)
    ]
    rng.shuffle(short_requests)
    rng.shuffle(long_requests)

    while short_requests or long_requests:
        if short_requests:
            requests.append(short_requests.pop())
        if long_requests:
            requests.append(long_requests.pop())
    return requests


def build_online_requests(args: argparse.Namespace, rng: random.Random) -> list[RequestSpec]:
    make_request = make_request_factory(args, rng)
    requests: list[RequestSpec] = []

    for _ in range(args.online_initial_long_requests):
        prompt_len = rng.randint(args.long_input_len_min, args.long_input_len_max)
        requests.append(make_request(prompt_len, "initial_long", 0.0))

    for _ in range(args.online_initial_short_requests):
        prompt_len = rng.randint(args.short_input_len_min, args.short_input_len_max)
        requests.append(make_request(prompt_len, "initial_short", 0.0))

    next_short_arrival = args.online_start_gap_ms / 1000.0
    interval_s = args.online_arrival_interval_ms / 1000.0
    for _ in range(args.online_late_short_requests):
        prompt_len = rng.randint(args.short_input_len_min, args.short_input_len_max)
        requests.append(make_request(prompt_len, "late_short", next_short_arrival))
        next_short_arrival += interval_s

    requests.sort(key=lambda request: (request.arrival_s, request.group))
    return requests


def build_requests(args: argparse.Namespace) -> list[RequestSpec]:
    rng = random.Random(args.seed)
    if args.workload == "online":
        return build_online_requests(args, rng)
    return build_batch_requests(args, rng)


def load_impl(impl: str):
    if impl == "current":
        sys.path.insert(0, str(REPO_ROOT))
        from llm import LLM  # pylint: disable=import-outside-toplevel
        from sampling_params import SamplingParams  # pylint: disable=import-outside-toplevel

        def make_sampling(max_new_tokens: int, temperature: float):
            return SamplingParams(temperature=temperature, ignore_eos=True, max_token=max_new_tokens)

        return LLM, make_sampling

    if impl == "original":
        sys.path.insert(0, str(ORIGINAL_ROOT))
        from nanovllm import LLM, SamplingParams  # pylint: disable=import-outside-toplevel

        def make_sampling(max_new_tokens: int, temperature: float):
            return SamplingParams(temperature=temperature, ignore_eos=True, max_tokens=max_new_tokens)

        return LLM, make_sampling

    raise ValueError(f"Unknown impl: {impl}")


def build_engine_kwargs(args: argparse.Namespace, impl: str) -> dict:
    kwargs = {
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if impl == "current":
        kwargs["max_num_chunk_tokens"] = args.max_num_chunk_tokens
    return kwargs


def get_metric_stat(result: dict, metric_name: str, group: str, stat: str) -> float | None:
    return result.get(metric_name, {}).get(group, {}).get(stat)


def build_single_headline_metrics(result: dict) -> dict[str, float]:
    metrics = {
        "request_per_s": result.get("request_per_s"),
        "prompt_tok_per_s": result.get("prompt_tok_per_s"),
        "decode_tok_per_s": result.get("decode_tok_per_s"),https://pic3.zhimg.com/v2-c6607afcbdaaa2c3c521efbf775d4798_1440w.jpg
        "total_tok_per_s": result.get("total_tok_per_s"),
        "ttft_all_p50_s": get_metric_stat(result, "ttft_s", "all", "p50"),
        "ttft_all_p95_s": get_metric_stat(result, "ttft_s", "all", "p95"),
        "latency_all_p50_s": get_metric_stat(result, "latency_s", "all", "p50"),
        "latency_all_p95_s": get_metric_stat(result, "latency_s", "all", "p95"),
        "queue_delay_all_p50_s": get_metric_stat(result, "queue_delay_s", "all", "p50"),
        "queue_delay_all_p95_s": get_metric_stat(result, "queue_delay_s", "all", "p95"),
        "late_short_ttft_p50_s": get_metric_stat(result, "ttft_s", "late_short", "p50"),
        "late_short_ttft_p95_s": get_metric_stat(result, "ttft_s", "late_short", "p95"),
        "late_short_latency_p50_s": get_metric_stat(result, "latency_s", "late_short", "p50"),
        "late_short_latency_p95_s": get_metric_stat(result, "latency_s", "late_short", "p95"),
        "late_short_queue_delay_p50_s": get_metric_stat(result, "queue_delay_s", "late_short", "p50"),
        "late_short_queue_delay_p95_s": get_metric_stat(result, "queue_delay_s", "late_short", "p95"),
    }
    return {name: value for name, value in metrics.items() if value is not None}


def build_compare_headline_metrics(summary: dict) -> dict[str, dict[str, float | bool | None]]:
    baseline = get_baseline_result(summary)
    candidate = get_candidate_result(summary)
    metric_specs = [
        ("request_per_s", baseline.get("request_per_s"), candidate.get("request_per_s"), False),
        ("prompt_tok_per_s", baseline.get("prompt_tok_per_s"), candidate.get("prompt_tok_per_s"), False),
        ("decode_tok_per_s", baseline.get("decode_tok_per_s"), candidate.get("decode_tok_per_s"), False),
        ("total_tok_per_s", baseline.get("total_tok_per_s"), candidate.get("total_tok_per_s"), False),
        ("ttft_all_p50_s", get_metric_stat(baseline, "ttft_s", "all", "p50"), get_metric_stat(candidate, "ttft_s", "all", "p50"), True),
        ("ttft_all_p95_s", get_metric_stat(baseline, "ttft_s", "all", "p95"), get_metric_stat(candidate, "ttft_s", "all", "p95"), True),
        ("latency_all_p50_s", get_metric_stat(baseline, "latency_s", "all", "p50"), get_metric_stat(candidate, "latency_s", "all", "p50"), True),
        ("latency_all_p95_s", get_metric_stat(baseline, "latency_s", "all", "p95"), get_metric_stat(candidate, "latency_s", "all", "p95"), True),
        ("queue_delay_all_p50_s", get_metric_stat(baseline, "queue_delay_s", "all", "p50"), get_metric_stat(candidate, "queue_delay_s", "all", "p50"), True),
        ("queue_delay_all_p95_s", get_metric_stat(baseline, "queue_delay_s", "all", "p95"), get_metric_stat(candidate, "queue_delay_s", "all", "p95"), True),
        ("late_short_ttft_p50_s", get_metric_stat(baseline, "ttft_s", "late_short", "p50"), get_metric_stat(candidate, "ttft_s", "late_short", "p50"), True),
        ("late_short_ttft_p95_s", get_metric_stat(baseline, "ttft_s", "late_short", "p95"), get_metric_stat(candidate, "ttft_s", "late_short", "p95"), True),
        ("late_short_latency_p50_s", get_metric_stat(baseline, "latency_s", "late_short", "p50"), get_metric_stat(candidate, "latency_s", "late_short", "p50"), True),
        ("late_short_latency_p95_s", get_metric_stat(baseline, "latency_s", "late_short", "p95"), get_metric_stat(candidate, "latency_s", "late_short", "p95"), True),
        ("late_short_queue_delay_p50_s", get_metric_stat(baseline, "queue_delay_s", "late_short", "p50"), get_metric_stat(candidate, "queue_delay_s", "late_short", "p50"), True),
        ("late_short_queue_delay_p95_s", get_metric_stat(baseline, "queue_delay_s", "late_short", "p95"), get_metric_stat(candidate, "queue_delay_s", "late_short", "p95"), True),
    ]
    headline_metrics = {}
    for metric_name, baseline, candidate, lower_is_better in metric_specs:
        if baseline is None or candidate is None:
            continue
        headline_metrics[metric_name] = {
            "original": baseline,
            "current": candidate,
            "change_pct": speedup_pct(baseline, candidate, lower_is_better=lower_is_better),
            "lower_is_better": lower_is_better,
        }
    return headline_metrics


def build_comparison_metrics(baseline: dict, candidate: dict) -> dict[str, float | None]:
    comparison = {
        "decode_tok_per_s_speedup_pct": speedup_pct(baseline["decode_tok_per_s"], candidate["decode_tok_per_s"]),
        "total_tok_per_s_speedup_pct": speedup_pct(baseline["total_tok_per_s"], candidate["total_tok_per_s"]),
        "ttft_all_p50_improvement_pct": speedup_pct(
            baseline["ttft_s"]["all"]["p50"],
            candidate["ttft_s"]["all"]["p50"],
            lower_is_better=True,
        ),
        "ttft_all_p95_improvement_pct": speedup_pct(
            baseline["ttft_s"]["all"]["p95"],
            candidate["ttft_s"]["all"]["p95"],
            lower_is_better=True,
        ),
        "latency_all_p50_improvement_pct": speedup_pct(
            baseline["latency_s"]["all"]["p50"],
            candidate["latency_s"]["all"]["p50"],
            lower_is_better=True,
        ),
        "latency_all_p95_improvement_pct": speedup_pct(
            baseline["latency_s"]["all"]["p95"],
            candidate["latency_s"]["all"]["p95"],
            lower_is_better=True,
        ),
    }
    for metric_name in ("ttft_s", "latency_s", "queue_delay_s"):
        for group in ("short", "long", "late_short", "initial_long"):
            add_group_improvements(comparison, baseline, candidate, metric_name, group)
    return comparison


def format_metric_value(metric_name: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if metric_name.endswith("_tok_per_s"):
        return format_stat(value, " tok/s")
    if metric_name.endswith("_per_s"):
        return format_stat(value, " req/s")
    if metric_name.endswith("_s"):
        return format_stat(value, "s")
    if metric_name.endswith("_pct"):
        return format_stat(value, "%")
    return format_stat(value)


def aggregate_numeric(values: list[int | float], aggregation: str) -> float | int:
    if aggregation == "mean":
        return sum(values) / len(values)
    return percentile([float(value) for value in values], 0.50)


def aggregate_value(values: list, aggregation: str):
    non_none_values = [value for value in values if value is not None]
    if not non_none_values:
        return None

    first = non_none_values[0]
    if isinstance(first, dict):
        keys = []
        for value in non_none_values:
            for key in value.keys():
                if key not in keys:
                    keys.append(key)
        aggregated = {}
        for key in keys:
            aggregated[key] = aggregate_value([value.get(key) for value in non_none_values if key in value], aggregation)
        return aggregated

    if isinstance(first, list):
        return non_none_values

    if isinstance(first, bool):
        return first

    if isinstance(first, (int, float)):
        if all(value == first for value in non_none_values):
            return first
        return aggregate_numeric(non_none_values, aggregation)

    return first


def aggregate_single_results(results: list[dict], aggregation: str) -> dict:
    aggregated = aggregate_value(results, aggregation)
    aggregated.pop("run_index", None)
    aggregated["aggregation"] = aggregation
    aggregated["repeat_count"] = len(results)
    aggregated["runs"] = results
    aggregated["headline_metrics"] = build_single_headline_metrics(aggregated)
    return aggregated


def get_baseline_result(summary: dict) -> dict:
    return summary.get("baseline", summary.get("original"))


def get_candidate_result(summary: dict) -> dict:
    return summary.get("candidate", summary.get("current"))


def get_baseline_label(summary: dict) -> str:
    return summary.get("baseline_label", "original")


def get_candidate_label(summary: dict) -> str:
    return summary.get("candidate_label", "current")


def collect_group_metrics(values_by_seq: dict[int, float], seq_group: dict[int, str]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = {}
    for seq_id, value in values_by_seq.items():
        grouped.setdefault(seq_group[seq_id], []).append(value)
        grouped.setdefault("all", []).append(value)
    return {group: summarize(values) for group, values in grouped.items()}


def build_per_request_rows(
    seq_by_id: dict[int, object],
    request_meta: dict[int, RequestSpec],
    metrics: dict[str, dict[int, float]],
    submit_elapsed_by_seq: dict[int, float] | None = None,
) -> list[dict[str, float | int | str | None]]:
    rows = []
    submit_elapsed_by_seq = submit_elapsed_by_seq or {}
    for seq_id in sorted(request_meta):
        seq = seq_by_id[seq_id]
        request = request_meta[seq_id]
        rows.append(
            {
                "seq_id": seq_id,
                "group": request.group,
                "prompt_tokens": len(request.prompt_token_ids),
                "requested_output_tokens": request.max_new_tokens,
                "output_tokens": seq.num_completion_tokens,
                "arrival_s": request.arrival_s,
                "submit_s": submit_elapsed_by_seq.get(seq_id, 0.0),
                "queue_delay_s": metrics.get("queue_delay_s", {}).get(seq_id),
                "ttft_s": metrics.get("ttft_s", {}).get(seq_id),
                "latency_s": metrics.get("latency_s", {}).get(seq_id),
            }
        )
    return rows


def finalize_result(
    args: argparse.Namespace,
    requests: list[RequestSpec],
    seq_by_id: dict[int, object],
    request_meta: dict[int, RequestSpec],
    wall_time: float,
    metrics: dict[str, dict[int, float]],
) -> dict:
    prompt_tokens = sum(len(request.prompt_token_ids) for request in requests)
    requested_output_tokens = sum(request.max_new_tokens for request in requests)
    actual_output_tokens = sum(seq.num_completion_tokens for seq in seq_by_id.values())
    seq_group = {seq_id: request.group for seq_id, request in request_meta.items()}
    group_counts: dict[str, int] = {}
    for request in requests:
        group_counts[request.group] = group_counts.get(request.group, 0) + 1

    result = {
        "impl": args.impl,
        "model_path": args.model_path,
        "workload": args.workload,
        "num_seqs": len(requests),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "max_num_chunk_tokens": args.max_num_chunk_tokens if args.impl == "current" else None,
        "prompt_tokens": prompt_tokens,
        "requested_output_tokens": requested_output_tokens,
        "output_tokens": actual_output_tokens,
        "wall_time_s": wall_time,
        "request_per_s": len(requests) / wall_time,
        "prompt_tok_per_s": prompt_tokens / wall_time,
        "decode_tok_per_s": actual_output_tokens / wall_time,
        "total_tok_per_s": (prompt_tokens + actual_output_tokens) / wall_time,
        "avg_prompt_tokens": prompt_tokens / len(requests),
        "avg_output_tokens": actual_output_tokens / len(seq_by_id),
        "request_groups": group_counts,
    }

    if args.workload == "online":
        result["arrival_window_s"] = max((request.arrival_s for request in requests), default=0.0)

    for metric_name, values_by_seq in metrics.items():
        result[metric_name] = collect_group_metrics(values_by_seq, seq_group)
    result["headline_metrics"] = build_single_headline_metrics(result)
    return result


def run_batch_impl(args: argparse.Namespace, llm, make_sampling, requests: list[RequestSpec]) -> tuple[dict, list[dict[str, float | int | str | None]]]:
    request_meta: dict[int, RequestSpec] = {}
    for request in requests:
        sampling = make_sampling(request.max_new_tokens, args.temperature)
        llm.add_request(request.prompt_token_ids, sampling)
        seq = llm.scheduler.waiting[-1]
        request_meta[seq.seq_id] = request

    seq_by_id = {seq.seq_id: seq for seq in llm.scheduler.waiting}
    ttft_by_seq: dict[int, float] = {}
    latency_by_seq: dict[int, float] = {}
    started_at = perf_counter()

    while not llm.is_finished():
        llm.step()
        elapsed = perf_counter() - started_at
        for seq_id, seq in seq_by_id.items():
            if seq_id not in ttft_by_seq and seq.num_completion_tokens > 0:
                ttft_by_seq[seq_id] = elapsed
            if seq_id not in latency_by_seq and seq.is_finished:
                latency_by_seq[seq_id] = elapsed

    wall_time = perf_counter() - started_at
    metrics = {"ttft_s": ttft_by_seq, "latency_s": latency_by_seq}
    result = finalize_result(
        args,
        requests,
        seq_by_id,
        request_meta,
        wall_time,
        metrics,
    )
    return result, build_per_request_rows(seq_by_id, request_meta, metrics)


def run_online_impl(args: argparse.Namespace, llm, make_sampling, requests: list[RequestSpec]) -> tuple[dict, list[dict[str, float | int | str | None]]]:
    pending_requests: deque[RequestSpec] = deque(sorted(requests, key=lambda request: (request.arrival_s, request.group)))
    request_meta: dict[int, RequestSpec] = {}
    seq_by_id: dict[int, object] = {}
    submit_elapsed_by_seq: dict[int, float] = {}
    queue_delay_by_seq: dict[int, float] = {}
    ttft_by_seq: dict[int, float] = {}
    latency_by_seq: dict[int, float] = {}
    started_at = perf_counter()
    idle_sleep_s = max(args.idle_sleep_ms, 0.0) / 1000.0

    while pending_requests or not llm.is_finished():
        elapsed = perf_counter() - started_at
        while pending_requests and pending_requests[0].arrival_s <= elapsed:
            request = pending_requests.popleft()
            submit_elapsed = perf_counter() - started_at
            sampling = make_sampling(request.max_new_tokens, args.temperature)
            llm.add_request(request.prompt_token_ids, sampling)
            seq = llm.scheduler.waiting[-1]
            seq_by_id[seq.seq_id] = seq
            request_meta[seq.seq_id] = request
            submit_elapsed_by_seq[seq.seq_id] = submit_elapsed
            queue_delay_by_seq[seq.seq_id] = max(submit_elapsed - request.arrival_s, 0.0)

        if llm.is_finished():
            if not pending_requests:
                break
            sleep_for = min(idle_sleep_s, max(pending_requests[0].arrival_s - elapsed, 0.0))
            if sleep_for > 0:
                sleep(sleep_for)
            continue

        llm.step()
        elapsed = perf_counter() - started_at
        for seq_id, seq in seq_by_id.items():
            request = request_meta[seq_id]
            if seq_id not in ttft_by_seq and seq.num_completion_tokens > 0:
                ttft_by_seq[seq_id] = elapsed - request.arrival_s
            if seq_id not in latency_by_seq and seq.is_finished:
                latency_by_seq[seq_id] = elapsed - request.arrival_s

    wall_time = perf_counter() - started_at
    metrics = {
        "queue_delay_s": queue_delay_by_seq,
        "ttft_s": ttft_by_seq,
        "latency_s": latency_by_seq,
    }
    result = finalize_result(
        args,
        requests,
        seq_by_id,
        request_meta,
        wall_time,
        metrics,
    )
    return result, build_per_request_rows(seq_by_id, request_meta, metrics, submit_elapsed_by_seq)


def run_single_impl(args: argparse.Namespace) -> tuple[dict, list[dict[str, float | int | str | None]]]:
    requests = build_requests(args)
    LLM, make_sampling = load_impl(args.impl)
    engine_kwargs = build_engine_kwargs(args, args.impl)
    llm = LLM(args.model_path, **engine_kwargs)

    warmup_sampling = make_sampling(args.warmup_tokens, args.temperature)
    llm.generate([[1, 2, 3, 4]], warmup_sampling, use_tqdm=False)

    if args.workload == "online":
        return run_online_impl(args, llm, make_sampling, requests)
    return run_batch_impl(args, llm, make_sampling, requests)


def namespace_to_cli(
    args: argparse.Namespace,
    impl: str,
    *,
    max_num_chunk_tokens: int | None = None,
    repeat: int = 1,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "benchmark_compare.py"),
        "--impl",
        impl,
        "--model-path",
        args.model_path,
        "--workload",
        args.workload,
        "--num-seqs",
        str(args.num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-chunk-tokens",
        str(args.max_num_chunk_tokens if max_num_chunk_tokens is None else max_num_chunk_tokens),
        "--min-output-len",
        str(args.min_output_len),
        "--max-output-len",
        str(args.max_output_len),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--vocab-upper-bound",
        str(args.vocab_upper_bound),
        "--idle-sleep-ms",
        str(args.idle_sleep_ms),
        "--repeat",
        str(repeat),
        "--aggregation",
        args.aggregation,
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.workload == "uniform":
        cmd.extend(
            [
                "--min-input-len",
                str(args.min_input_len),
                "--max-input-len",
                str(args.max_input_len),
            ]
        )
    else:
        cmd.extend(
            [
                "--short-input-len-min",
                str(args.short_input_len_min),
                "--short-input-len-max",
                str(args.short_input_len_max),
                "--long-input-len-min",
                str(args.long_input_len_min),
                "--long-input-len-max",
                str(args.long_input_len_max),
            ]
        )
    if args.workload == "online":
        cmd.extend(
            [
                "--online-initial-long-requests",
                str(args.online_initial_long_requests),
                "--online-initial-short-requests",
                str(args.online_initial_short_requests),
                "--online-late-short-requests",
                str(args.online_late_short_requests),
                "--online-start-gap-ms",
                str(args.online_start_gap_ms),
                "--online-arrival-interval-ms",
                str(args.online_arrival_interval_ms),
            ]
        )
    return cmd


def extract_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise ValueError("Benchmark child process did not emit a JSON result.")


def run_child_command(cmd: list[str], label: str) -> dict:
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"Benchmark child failed for {label} with exit code {completed.returncode}")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return extract_result(completed.stdout)


def run_repeated_leaf(
    args: argparse.Namespace,
    impl: str,
    *,
    chunk_tokens: int | None = None,
    label: str | None = None,
) -> tuple[dict, list[dict]]:
    label = label or impl
    results = []
    for repeat_idx in range(args.repeat):
        print(f"running {label} repeat {repeat_idx + 1}/{args.repeat}...")
        cmd = namespace_to_cli(args, impl, max_num_chunk_tokens=chunk_tokens, repeat=1)
        result = run_child_command(cmd, label)
        result["run_index"] = repeat_idx
        if chunk_tokens is not None:
            result["max_num_chunk_tokens"] = chunk_tokens
        results.append(result)
    return aggregate_single_results(results, args.aggregation), results


def add_group_improvements(comparison: dict, original: dict, current: dict, metric_name: str, group: str):
    original_metric = original.get(metric_name, {})
    current_metric = current.get(metric_name, {})
    if group not in original_metric or group not in current_metric:
        return
    for stat in ("p50", "p95"):
        comparison[f"{group}_{metric_name}_{stat}_improvement_pct"] = speedup_pct(
            original_metric[group][stat],
            current_metric[group][stat],
            lower_is_better=True,
        )


def build_result_csv_rows(result: dict) -> list[dict[str, str | float | int | None]]:
    rows = []
    overall_fields = [
        "num_seqs",
        "prompt_tokens",
        "requested_output_tokens",
        "output_tokens",
        "wall_time_s",
        "repeat_count",
        "request_per_s",
        "prompt_tok_per_s",
        "decode_tok_per_s",
        "total_tok_per_s",
        "avg_prompt_tokens",
        "avg_output_tokens",
        "arrival_window_s",
    ]
    for field in overall_fields:
        if field not in result:
            continue
        rows.append(
            {
                "scope": "single",
                "section": "overall",
                "impl": result["impl"],
                "metric": field,
                "group": "",
                "stat": "value",
                "value": result[field],
            }
        )

    for metric_name in ("queue_delay_s", "ttft_s", "latency_s"):
        if metric_name not in result:
            continue
        for group, stats in sorted(result[metric_name].items()):
            for stat_name, value in stats.items():
                rows.append(
                    {
                        "scope": "single",
                        "section": "distribution",
                        "impl": result["impl"],
                        "metric": metric_name,
                        "group": group,
                        "stat": stat_name,
                        "value": value,
                    }
                )

    for metric_name, value in sorted(result.get("headline_metrics", {}).items()):
        rows.append(
            {
                "scope": "single",
                "section": "headline",
                "impl": result["impl"],
                "metric": metric_name,
                "group": "",
                "stat": "value",
                "value": value,
            }
        )
    return rows


def build_compare_csv_rows(summary: dict) -> list[dict[str, str | float | int | None]]:
    rows = []
    baseline = get_baseline_result(summary)
    candidate = get_candidate_result(summary)
    baseline_label = get_baseline_label(summary)
    candidate_label = get_candidate_label(summary)
    baseline_rows = build_result_csv_rows(baseline)
    candidate_rows = build_result_csv_rows(candidate)
    for row in baseline_rows:
        row["label"] = baseline_label
    for row in candidate_rows:
        row["label"] = candidate_label
    rows.extend(baseline_rows)
    rows.extend(candidate_rows)

    for metric_name, value in sorted(summary["comparison"].items()):
        rows.append(
            {
                "scope": "compare",
                "section": "comparison",
                "impl": summary.get("mode", "compare"),
                "label": "",
                "metric": metric_name,
                "group": "",
                "stat": "change_pct",
                "value": value,
            }
        )

    for metric_name, metric_values in sorted(summary.get("headline_metrics", {}).items()):
        for stat_name in ("original", "current", "change_pct"):
            rows.append(
                {
                    "scope": "compare",
                    "section": "headline",
                    "impl": summary.get("mode", "compare"),
                    "label": "",
                    "metric": metric_name,
                    "group": "",
                    "stat": stat_name,
                    "value": metric_values.get(stat_name),
                }
            )
    return rows


def build_markdown_for_single(result: dict) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        f"- impl: `{result['impl']}`",
        f"- workload: `{result['workload']}`",
        f"- num_seqs: `{result['num_seqs']}`",
        f"- prompt_tokens: `{result['prompt_tokens']}`",
        f"- output_tokens: `{result['output_tokens']}`",
        f"- repeat_count: `{result.get('repeat_count', 1)}`",
        f"- aggregation: `{result.get('aggregation', 'single')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric_name, value in sorted(result.get("headline_metrics", {}).items()):
        lines.append(f"| `{metric_name}` | {format_metric_value(metric_name, value)} |")
    return "\n".join(lines) + "\n"


def build_markdown_for_compare(summary: dict) -> str:
    baseline = get_baseline_result(summary)
    candidate = get_candidate_result(summary)
    baseline_label = get_baseline_label(summary)
    candidate_label = get_candidate_label(summary)
    lines = [
        "# Benchmark Comparison",
        "",
        f"- mode: `{summary.get('mode', 'compare')}`",
        f"- workload: `{candidate['workload']}`",
        f"- num_seqs: `{candidate['num_seqs']}`",
        f"- repeat_count: `{summary.get('repeat_count', 1)}`",
        f"- aggregation: `{summary.get('aggregation', 'single')}`",
        "",
        f"| Metric | {baseline_label} | {candidate_label} | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric_name, metric_values in sorted(summary.get("headline_metrics", {}).items()):
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                metric_name,
                format_metric_value(metric_name, metric_values.get("original")),
                format_metric_value(metric_name, metric_values.get("current")),
                format_metric_value("change_pct", metric_values.get("change_pct")),
            )
        )
    return "\n".join(lines) + "\n"


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_output(path_str: str, payload: dict):
    path = Path(path_str)
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_csv_output(path_str: str, rows: list[dict[str, str | float | int | None]]):
    path = Path(path_str)
    ensure_parent_dir(path)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_output(path_str: str, content: str):
    path = Path(path_str)
    ensure_parent_dir(path)
    path.write_text(content, encoding="utf-8")


def write_outputs(
    args: argparse.Namespace,
    payload: dict,
    per_request_rows: list[dict[str, str | float | int | None]] | None = None,
):
    if args.json_out:
        write_json_output(args.json_out, payload)
    if args.csv_out:
        rows = build_compare_csv_rows(payload) if args.impl in {"compare", "ablation"} else build_result_csv_rows(payload)
        write_csv_output(args.csv_out, rows)
    if args.markdown_out:
        content = build_markdown_for_compare(payload) if args.impl in {"compare", "ablation"} else build_markdown_for_single(payload)
        write_markdown_output(args.markdown_out, content)
    if args.per_request_csv_out:
        if args.impl in {"compare", "ablation"}:
            raise ValueError("--per-request-csv-out only supports --impl current/original, not compare/ablation.")
        write_csv_output(args.per_request_csv_out, per_request_rows or [])


def run_compare(args: argparse.Namespace) -> dict:
    baseline, baseline_runs = run_repeated_leaf(args, "original", label="original")
    candidate, candidate_runs = run_repeated_leaf(args, "current", label="current")
    comparison = build_comparison_metrics(baseline, candidate)
    summary = {
        "mode": "compare",
        "baseline_label": "original",
        "candidate_label": "current",
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "repeat_count": args.repeat,
        "aggregation": args.aggregation,
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
    }
    summary["original"] = baseline
    summary["current"] = candidate
    summary["headline_metrics"] = build_compare_headline_metrics(summary)
    return summary


def run_ablation(args: argparse.Namespace) -> dict:
    baseline_chunk = args.ablation_baseline_chunk_tokens
    candidate_chunk = args.ablation_candidate_chunk_tokens
    baseline_label = f"current(chunk={baseline_chunk})"
    candidate_label = f"current(chunk={candidate_chunk})"
    baseline, baseline_runs = run_repeated_leaf(
        args,
        "current",
        chunk_tokens=baseline_chunk,
        label=baseline_label,
    )
    candidate, candidate_runs = run_repeated_leaf(
        args,
        "current",
        chunk_tokens=candidate_chunk,
        label=candidate_label,
    )
    comparison = build_comparison_metrics(baseline, candidate)
    summary = {
        "mode": "ablation",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "repeat_count": args.repeat,
        "aggregation": args.aggregation,
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
        "ablation": {
            "baseline_chunk_tokens": baseline_chunk,
            "candidate_chunk_tokens": candidate_chunk,
        },
    }
    summary["headline_metrics"] = build_compare_headline_metrics(summary)
    return summary


def print_single(result: dict):
    header = (
        f"[{result['impl']}] reqs={result['num_seqs']} "
        f"prompt_tok={result['prompt_tokens']} output_tok={result['output_tokens']} "
        f"wall={format_stat(result['wall_time_s'], 's')} "
        f"req={format_stat(result['request_per_s'], ' req/s')} "
        f"prompt={format_stat(result['prompt_tok_per_s'], ' tok/s')} "
        f"decode={format_stat(result['decode_tok_per_s'], ' tok/s')} "
        f"total={format_stat(result['total_tok_per_s'], ' tok/s')}"
    )
    if "arrival_window_s" in result:
        header += f" arrival_window={format_stat(result['arrival_window_s'], 's')}"
    if "repeat_count" in result:
        header += f" repeats={result['repeat_count']} aggregation={result.get('aggregation', 'single')}"
    print(header)
    for metric_name in ("queue_delay_s", "ttft_s", "latency_s"):
        if metric_name not in result:
            continue
        for group, stats in sorted(result[metric_name].items()):
            print(
                f"  {metric_name}.{group}: "
                f"p50={format_stat(stats['p50'], 's')} "
                f"p95={format_stat(stats['p95'], 's')} "
                f"p99={format_stat(stats['p99'], 's')}"
            )
    headline = result.get("headline_metrics", {})
    if headline:
        print(
            "  headline: "
            + " ".join(f"{metric}={format_metric_value(metric, value)}" for metric, value in sorted(headline.items()))
        )


def print_compare(summary: dict):
    print(
        f"mode={summary.get('mode', 'compare')} "
        f"repeats={summary.get('repeat_count', 1)} "
        f"aggregation={summary.get('aggregation', 'single')}"
    )
    baseline = get_baseline_result(summary)
    candidate = get_candidate_result(summary)
    print(f"[{get_baseline_label(summary)}]")
    print_single(baseline)
    print(f"[{get_candidate_label(summary)}]")
    print_single(candidate)
    comparison = summary["comparison"]
    preferred_order = [
        "decode_tok_per_s_speedup_pct",
        "total_tok_per_s_speedup_pct",
        "ttft_all_p50_improvement_pct",
        "ttft_all_p95_improvement_pct",
        "latency_all_p50_improvement_pct",
        "latency_all_p95_improvement_pct",
        "late_short_queue_delay_s_p50_improvement_pct",
        "late_short_queue_delay_s_p95_improvement_pct",
        "late_short_ttft_s_p50_improvement_pct",
        "late_short_ttft_s_p95_improvement_pct",
        "late_short_latency_s_p50_improvement_pct",
        "late_short_latency_s_p95_improvement_pct",
    ]
    emitted = set()
    parts = []
    for key in preferred_order:
        if key in comparison:
            parts.append(f"{key}={format_stat(comparison[key], '%')}")
            emitted.add(key)
    for key in sorted(comparison):
        if key in emitted:
            continue
        parts.append(f"{key}={format_stat(comparison[key], '%')}")
    print(
        "comparison: "
        + f"baseline={get_baseline_label(summary)} "
        + f"candidate={get_candidate_label(summary)} "
        + " ".join(parts)
    )
    headline = summary.get("headline_metrics", {})
    if headline:
        print(
            "headline: "
            + " ".join(
                "{}(orig={}, cur={}, delta={})".format(
                    metric_name,
                    format_metric_value(metric_name, metric_values.get("original")),
                    format_metric_value(metric_name, metric_values.get("current")),
                    format_metric_value("change_pct", metric_values.get("change_pct")),
                )
                for metric_name, metric_values in sorted(headline.items())
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare this repo against ../nano-vllm on the same synthetic or online workload.")
    parser.add_argument("--impl", choices=("compare", "current", "original", "ablation"), default="compare")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--workload", choices=("uniform", "mixed", "online"), default="mixed")
    parser.add_argument("--num-seqs", type=int, default=128)
    parser.add_argument("--min-input-len", type=int, default=128)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--short-input-len-min", type=int, default=32)
    parser.add_argument("--short-input-len-max", type=int, default=128)
    parser.add_argument("--long-input-len-min", type=int, default=1024)
    parser.add_argument("--long-input-len-max", type=int, default=2048)
    parser.add_argument("--min-output-len", type=int, default=128)
    parser.add_argument("--max-output-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-chunk-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--aggregation", choices=("median", "mean"), default="median")
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--vocab-upper-bound", type=int, default=10000)
    parser.add_argument("--idle-sleep-ms", type=float, default=1.0)
    parser.add_argument("--online-initial-long-requests", type=int, default=16)
    parser.add_argument("--online-initial-short-requests", type=int, default=0)
    parser.add_argument("--online-late-short-requests", type=int, default=64)
    parser.add_argument("--online-start-gap-ms", type=float, default=100.0)
    parser.add_argument("--online-arrival-interval-ms", type=float, default=50.0)
    parser.add_argument("--ablation-baseline-chunk-tokens", type=int, default=4096)
    parser.add_argument("--ablation-candidate-chunk-tokens", type=int, default=256)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--json-out")
    parser.add_argument("--csv-out")
    parser.add_argument("--markdown-out")
    parser.add_argument("--per-request-csv-out")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if args.impl in {"compare", "original"} and not ORIGINAL_ROOT.is_dir():
        raise FileNotFoundError(f"Original repo does not exist: {ORIGINAL_ROOT}")
    if args.impl in {"compare", "original"} and args.max_num_batched_tokens < args.max_model_len:
        raise ValueError(
            "For original nano-vllm compatibility, --max-num-batched-tokens must be >= --max-model-len. "
            "Use a chunk size smaller than max_model_len to evaluate chunked prefill fairly."
        )
    if args.repeat <= 0:
        raise ValueError("--repeat must be >= 1.")
    if args.workload == "online":
        total_online_requests = (
            args.online_initial_long_requests
            + args.online_initial_short_requests
            + args.online_late_short_requests
        )
        if total_online_requests <= 0:
            raise ValueError("Online workload requires at least one request.")
        if args.online_start_gap_ms < 0 or args.online_arrival_interval_ms < 0 or args.idle_sleep_ms < 0:
            raise ValueError("Online timing arguments must be non-negative.")
    if args.ablation_baseline_chunk_tokens <= 0 or args.ablation_candidate_chunk_tokens <= 0:
        raise ValueError("Ablation chunk sizes must be positive.")
    if args.impl in {"compare", "ablation"} and args.per_request_csv_out:
        raise ValueError("--per-request-csv-out only supports --impl current/original, not compare/ablation.")
    if args.repeat > 1 and args.per_request_csv_out:
        raise ValueError("--per-request-csv-out currently supports only --repeat 1.")

    if args.impl == "compare":
        summary = run_compare(args)
        print_compare(summary)
        write_outputs(args, summary)
        print(RESULT_PREFIX + json.dumps(summary, ensure_ascii=True))
        return
    if args.impl == "ablation":
        summary = run_ablation(args)
        print_compare(summary)
        write_outputs(args, summary)
        print(RESULT_PREFIX + json.dumps(summary, ensure_ascii=True))
        return

    if args.repeat > 1:
        result, _ = run_repeated_leaf(args, args.impl, label=args.impl)
        per_request_rows = None
    else:
        result, per_request_rows = run_single_impl(args)
    print_single(result)
    write_outputs(args, result, per_request_rows)
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
