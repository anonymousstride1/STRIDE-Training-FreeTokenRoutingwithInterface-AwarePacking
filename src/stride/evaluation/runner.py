from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from tqdm import tqdm

from ..models.base import VLMAdapter
from .data import read_jsonl
from .metrics import EVALUATION_VERSION, score_prediction
from .benchmark_metrics import BENCHMARK_METRICS, canonical_benchmark, summarize_benchmark


def format_evaluation_prompt(question: str, metric: str) -> str:
    """Match generation style to the benchmark's declared answer format."""
    instructions = {
        "yes_no": "Answer with exactly one word: yes or no.",
        "multiple_choice": "Answer with only the option letter: A, B, C, D, or E.",
        "exact_match": "Answer with only the shortest possible word or phrase. Do not explain.",
        "vqa_accuracy": "Answer with only the shortest possible word or phrase. Do not explain.",
        "contains": "Answer briefly and directly. Do not explain.",
        "caption_token_f1": "Provide a detailed, accurate description of the image. Describe visible content without inventing details.",
    }
    instruction = instructions.get(metric, "Answer briefly and directly. Do not explain.")
    return f"{question.rstrip()}\n{instruction}"


def _git_commit() -> str | None:
    return None


def _git_dirty() -> bool | None:
    return None


def environment_manifest() -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def router_implementation_hash() -> str:
    """Fingerprint inference-critical source so dirty-tree resumes stay safe."""
    package = Path(__file__).resolve().parents[1]
    paths = (
        package / 'routing.py',
        package / 'config.py',
        package / 'models' / 'llava.py',
        package / 'evaluation' / 'benchmark_metrics.py',
        package / 'models' / 'packing.py',
        package / 'models' / 'timing.py',
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(package).as_posix().encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resume_configs_match(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Compare inference-critical fields, excluding mutable Git status.

    Result files themselves commonly make an otherwise clean checkout dirty
    after the first job. That provenance change must not prevent crash resume;
    the implementation fingerprint still protects routing-critical source.
    """
    ignored = {'git_dirty'}
    return (
        {key: value for key, value in previous.items() if key not in ignored}
        == {key: value for key, value in current.items() if key not in ignored}
    )


def evaluate_jsonl(
    adapter: VLMAdapter,
    dataset_path: str | Path,
    output_dir: str | Path,
    method: str,
    budget: int,
    max_samples: int | None = None,
    resume: bool = True,
    generation_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
    progress_position: int = 0,
    sample_offset: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_file = output / "predictions.jsonl"
    completed: Counter[str] = Counter()
    active_router_config = getattr(adapter, 'router_config', None)
    run_config = {
        'git_commit': _git_commit(),
        'git_dirty': _git_dirty(),
        'router_implementation_sha256': router_implementation_hash(),
        'dataset': str(Path(dataset_path)),
        'adapter': type(adapter).__name__,
        'method': method,
        'budget': budget,
        'seed': seed,
        'max_samples': max_samples,
        'sample_offset': sample_offset,
        'router_config': (
            active_router_config.to_dict()
            if active_router_config is not None
            else None
        ),
        'generation_kwargs': generation_kwargs or {},
        'evaluation_version': EVALUATION_VERSION,
    }
    run_config_file = output / 'run_config.json'
    preserve_run_config = False
    if resume and run_config_file.exists():
        previous = json.loads(run_config_file.read_text(encoding='utf-8'))
        if not resume_configs_match(previous, run_config):
            raise ValueError(
                f'refusing to resume {output}: run configuration changed; '
                'move or remove that run directory first'
            )
        preserve_run_config = True
    elif resume and prediction_file.exists() and prediction_file.stat().st_size:
        raise ValueError(
            f'refusing to resume legacy predictions without run_config.json: '
            f'{prediction_file}'
        )
    if not preserve_run_config:
        run_config_file.write_text(
            json.dumps(run_config, indent=2), encoding='utf-8'
        )
    if resume and prediction_file.exists():
        with prediction_file.open("r", encoding="utf-8") as handle:
            completed = Counter(
                str(json.loads(line)["id"]) for line in handle if line.strip()
            )
    examples = list(read_jsonl(dataset_path))
    if sample_offset < 0:
        raise ValueError('sample_offset must be non-negative')
    examples = examples[sample_offset:]
    if max_samples is not None:
        examples = examples[:max_samples]
    generation_kwargs = generation_kwargs or {}
    mode = "a" if resume else "w"
    encountered: defaultdict[str, int] = defaultdict(int)
    with prediction_file.open(mode, encoding="utf-8", buffering=1) as handle:
        for example in tqdm(
            examples,
            desc=f"{method}@{budget}",
            unit="sample",
            position=progress_position,
            leave=progress_position == 0,
            dynamic_ncols=True,
            mininterval=1.0,
        ):
            # IDs need not be unique (MME has two questions per image). Resume
            # by occurrence count instead of skipping every row with an ID
            # that appeared at least once in the existing file.
            encountered[example.sample_id] += 1
            if encountered[example.sample_id] <= completed[example.sample_id]:
                continue
            evaluation_prompt = format_evaluation_prompt(example.question, example.metric)
            result = adapter.generate(
                example.image,
                evaluation_prompt,
                method=method,
                budget=budget,
                routing_prompt=example.question,
                **generation_kwargs,
            )
            score = score_prediction(result.text, example.answers, example.metric)
            record = {
                "id": example.sample_id,
                "prediction": result.text,
                "answers": example.answers,
                "metric": example.metric,
                "question": example.question,
                "image": str(example.image),
                "evaluation_prompt": evaluation_prompt,
                "score": score,
                "method": method,
                "budget": budget,
                "seed": seed,
                "metadata": example.metadata,
                **asdict(result),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with prediction_file.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if str(row["id"]) in {e.sample_id for e in examples}]
    router_config = getattr(adapter, "router_config", None)
    benchmark = canonical_benchmark(Path(dataset_path).parent.name)
    benchmark_metrics = summarize_benchmark(rows, benchmark) if benchmark in BENCHMARK_METRICS else {}
    summary = {
        "dataset": str(dataset_path),
        "method": method,
        "budget": budget,
        "seed": seed,
        "samples": len(rows),
        "mean_input_visual_tokens": mean(row["input_visual_tokens"] for row in rows) if rows else None,
        "mean_retained_visual_tokens": mean(row["output_visual_tokens"] for row in rows) if rows else None,
        "score": mean(row["score"] for row in rows) if rows else None,
        "benchmark": benchmark,
        "metrics": benchmark_metrics,
        "mean_prefill_seconds": mean(row["prefill_seconds"] for row in rows) if rows else None,
        "mean_generation_seconds": mean(row["generation_seconds"] for row in rows) if rows else None,
        "mean_preparation_seconds": mean(row.get('preparation_seconds') if row.get('preparation_seconds') is not None else row['prefill_seconds'] for row in rows) if rows else None,
        "mean_language_generate_seconds": mean(row.get('language_generate_seconds') if row.get('language_generate_seconds') is not None else row['generation_seconds'] for row in rows) if rows else None,
        "mean_decoder_prefill_seconds": mean(row['decoder_prefill_seconds'] for row in rows if row.get('decoder_prefill_seconds') is not None) if any(row.get('decoder_prefill_seconds') is not None for row in rows) else None,
        "mean_decoder_decode_seconds": mean(row['decoder_decode_seconds'] for row in rows if row.get('decoder_decode_seconds') is not None) if any(row.get('decoder_decode_seconds') is not None for row in rows) else None,
        "timing_protocols": sorted({row.get('timing_protocol', 'legacy_unspecified') for row in rows}),
        "benchmark_scoring_protocol": 'complete_yes_no_pairs_v2' if benchmark == 'mme' else EVALUATION_VERSION,
        "mean_preprocessing_seconds": mean(row.get("preprocessing_seconds", 0.0) for row in rows) if rows else None,
        "mean_vision_seconds": mean(row.get("vision_seconds", 0.0) for row in rows) if rows else None,
        "mean_routing_seconds": mean(row.get("routing_seconds", 0.0) for row in rows) if rows else None,
        "mean_packing_seconds": mean(row.get("packing_seconds", 0.0) for row in rows) if rows else None,
        "mean_total_seconds": mean(
            row.get("total_seconds", row["prefill_seconds"] + row["generation_seconds"])
            for row in rows
        ) if rows else None,
        "mean_peak_memory_bytes": mean(
            row["peak_memory_bytes"] for row in rows if row["peak_memory_bytes"] is not None
        ) if any(row["peak_memory_bytes"] is not None for row in rows) else None,
        "mean_selection_overhead_seconds": mean(row.get("selection_overhead_seconds", row.get("routing_seconds", 0.0)) for row in rows) if rows else None,
        "mean_dense_prefill_flops": mean(row["dense_prefill_flops"] for row in rows if row.get("dense_prefill_flops") is not None) if any(row.get("dense_prefill_flops") is not None for row in rows) else None,
        "mean_routed_prefill_flops": mean(row["routed_prefill_flops"] for row in rows if row.get("routed_prefill_flops") is not None) if any(row.get("routed_prefill_flops") is not None for row in rows) else None,
        "mean_flops_reduction_percent": mean(row["flops_reduction_percent"] for row in rows if row.get("flops_reduction_percent") is not None) if any(row.get("flops_reduction_percent") is not None for row in rows) else None,
        "environment": environment_manifest(),
        "router_config": router_config.to_dict() if router_config is not None else None,
        "generation_kwargs": generation_kwargs,
        "evaluation_version": EVALUATION_VERSION,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
