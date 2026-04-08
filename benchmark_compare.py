import argparse
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


def collect_group_metrics(values_by_seq: dict[int, float], seq_group: dict[int, str]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = {}
    for seq_id, value in values_by_seq.items():
        grouped.setdefault(seq_group[seq_id], []).append(value)
        grouped.setdefault("all", []).append(value)
    return {group: summarize(values) for group, values in grouped.items()}


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
        "decode_tok_per_s": actual_output_tokens / wall_time,
        "total_tok_per_s": (prompt_tokens + actual_output_tokens) / wall_time,
    }

    if args.workload == "online":
        result["arrival_window_s"] = max((request.arrival_s for request in requests), default=0.0)

    for metric_name, values_by_seq in metrics.items():
        result[metric_name] = collect_group_metrics(values_by_seq, seq_group)
    return result


def run_batch_impl(args: argparse.Namespace, llm, make_sampling, requests: list[RequestSpec]) -> dict:
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
    return finalize_result(
        args,
        requests,
        seq_by_id,
        request_meta,
        wall_time,
        {"ttft_s": ttft_by_seq, "latency_s": latency_by_seq},
    )


def run_online_impl(args: argparse.Namespace, llm, make_sampling, requests: list[RequestSpec]) -> dict:
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
    return finalize_result(
        args,
        requests,
        seq_by_id,
        request_meta,
        wall_time,
        {
            "queue_delay_s": queue_delay_by_seq,
            "ttft_s": ttft_by_seq,
            "latency_s": latency_by_seq,
        },
    )


def run_single_impl(args: argparse.Namespace) -> dict:
    requests = build_requests(args)
    LLM, make_sampling = load_impl(args.impl)
    engine_kwargs = build_engine_kwargs(args, args.impl)
    llm = LLM(args.model_path, **engine_kwargs)

    warmup_sampling = make_sampling(args.warmup_tokens, args.temperature)
    llm.generate([[1, 2, 3, 4]], warmup_sampling, use_tqdm=False)

    if args.workload == "online":
        return run_online_impl(args, llm, make_sampling, requests)
    return run_batch_impl(args, llm, make_sampling, requests)


def namespace_to_cli(args: argparse.Namespace, impl: str) -> list[str]:
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
        str(args.max_num_chunk_tokens),
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


def run_compare(args: argparse.Namespace) -> dict:
    results = {}
    for impl in ("original", "current"):
        cmd = namespace_to_cli(args, impl)
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
            raise RuntimeError(f"Benchmark child failed for {impl} with exit code {completed.returncode}")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        results[impl] = extract_result(completed.stdout)

    original = results["original"]
    current = results["current"]
    comparison = {
        "decode_tok_per_s_speedup_pct": speedup_pct(original["decode_tok_per_s"], current["decode_tok_per_s"]),
        "total_tok_per_s_speedup_pct": speedup_pct(original["total_tok_per_s"], current["total_tok_per_s"]),
        "ttft_all_p50_improvement_pct": speedup_pct(
            original["ttft_s"]["all"]["p50"],
            current["ttft_s"]["all"]["p50"],
            lower_is_better=True,
        ),
        "ttft_all_p95_improvement_pct": speedup_pct(
            original["ttft_s"]["all"]["p95"],
            current["ttft_s"]["all"]["p95"],
            lower_is_better=True,
        ),
        "latency_all_p50_improvement_pct": speedup_pct(
            original["latency_s"]["all"]["p50"],
            current["latency_s"]["all"]["p50"],
            lower_is_better=True,
        ),
        "latency_all_p95_improvement_pct": speedup_pct(
            original["latency_s"]["all"]["p95"],
            current["latency_s"]["all"]["p95"],
            lower_is_better=True,
        ),
    }
    for metric_name in ("ttft_s", "latency_s", "queue_delay_s"):
        for group in ("short", "long", "late_short", "initial_long"):
            add_group_improvements(comparison, original, current, metric_name, group)
    return {
        "original": original,
        "current": current,
        "comparison": comparison,
    }


def print_single(result: dict):
    header = (
        f"[{result['impl']}] reqs={result['num_seqs']} "
        f"prompt_tok={result['prompt_tokens']} output_tok={result['output_tokens']} "
        f"wall={format_stat(result['wall_time_s'], 's')} "
        f"decode={format_stat(result['decode_tok_per_s'], ' tok/s')} "
        f"total={format_stat(result['total_tok_per_s'], ' tok/s')}"
    )
    if "arrival_window_s" in result:
        header += f" arrival_window={format_stat(result['arrival_window_s'], 's')}"
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


def print_compare(summary: dict):
    print_single(summary["original"])
    print_single(summary["current"])
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
    print("comparison: " + " ".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare this repo against ../nano-vllm on the same synthetic or online workload.")
    parser.add_argument("--impl", choices=("compare", "current", "original"), default="compare")
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
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--vocab-upper-bound", type=int, default=10000)
    parser.add_argument("--idle-sleep-ms", type=float, default=1.0)
    parser.add_argument("--online-initial-long-requests", type=int, default=16)
    parser.add_argument("--online-initial-short-requests", type=int, default=0)
    parser.add_argument("--online-late-short-requests", type=int, default=64)
    parser.add_argument("--online-start-gap-ms", type=float, default=100.0)
    parser.add_argument("--online-arrival-interval-ms", type=float, default=50.0)
    parser.add_argument("--enforce-eager", action="store_true")
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

    if args.impl == "compare":
        summary = run_compare(args)
        print_compare(summary)
        print(RESULT_PREFIX + json.dumps(summary, ensure_ascii=True))
        return

    result = run_single_impl(args)
    print_single(result)
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
