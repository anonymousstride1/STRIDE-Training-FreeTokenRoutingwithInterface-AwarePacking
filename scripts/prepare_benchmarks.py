#!/usr/bin/env python
"""Materialize supported Hugging Face benchmarks into STRIDE's auditable JSONL schema."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

# Allow ``python scripts/prepare_benchmarks.py`` from a source checkout before
# the package has been installed. Installed environments continue to use the
# normal package resolution path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from PIL import Image
from tqdm import tqdm

from stride.progress import configure_library_output


REGISTRY = {
    "pope": ("lmms-lab/POPE", None, "test"),
    "textvqa": ("lmms-lab/textvqa", None, "validation"),
    "scienceqa": ("lmms-lab/ScienceQA", "ScienceQA-IMG", "test"),
    "gqa": ("lmms-lab/GQA", "testdev_balanced_instructions", "testdev"),
    "mmbench": ("lmms-lab/MMBench_EN", "default", "dev"),
    "mme": ("lmms-lab/MME", None, "test"),
}
GQA_IMAGE_CONFIG = "testdev_balanced_images"


def image_filename(sample_id: str, index: int, suffix: str = '.jpg') -> str:
    """Return a portable filename even when benchmark IDs contain paths."""
    digest = hashlib.sha1(str(sample_id).encode('utf-8')).hexdigest()[:12]
    return f'{index:08d}_{digest}{suffix}'


def read_excluded_ids(
    paths: list[Path], ledgers: list[Path] | None = None
) -> set[str]:
    """Read sample IDs that must not appear in a newly materialized split."""
    excluded: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "id" not in row:
                    raise ValueError(f"{path}:{line_number}: missing 'id'")
                excluded.add(str(row["id"]))
    for path in ledgers or []:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError(f"{path}: exclusion ledger must be a JSON list")
        excluded.update(str(value) for value in values)
    return excluded


def write_exclusion_ledger(output: Path, excluded_ids: set[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / 'excluded_ids.json').write_text(
        json.dumps(sorted(excluded_ids), indent=2), encoding='utf-8'
    )


def first(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def convert(name: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    question = str(first(row, "question", "query", "prompt", default=""))
    # POPE repeats question_id values across its three categories. Its row id
    # is globally unique, which is required for correct resume accounting.
    id_keys = ("id", "question_id") if name == "pope" else ("question_id", "id")
    sample_id = str(first(row, *id_keys, default=index))
    if name in {"scienceqa", "mmbench"}:
        choices = list(row.get("choices") or [row.get(letter) for letter in "ABCDE" if row.get(letter) is not None])
        letters = "ABCDE"
        question += "\n" + "\n".join(f"({letters[i]}) {choice}" for i, choice in enumerate(choices))
        raw_answer = row["answer"]
        answers = [letters[int(raw_answer)]] if str(raw_answer).isdigit() else [str(raw_answer).strip().upper()]
        metric = "multiple_choice"
    else:
        answers = first(row, "answers", "answer", default=[])
        if isinstance(answers, dict):
            answers = first(answers, "answer", "answers", default=[])
        if isinstance(answers, str):
            answers = [answers]
        metric = "yes_no" if name in {"pope", "mme"} else "vqa_accuracy" if name == "textvqa" else "exact_match"
    if not question.strip():
        raise ValueError(f"{name} row {index} has an empty question; keys={sorted(row)}")
    if not answers:
        raise ValueError(f"{name} row {index} has no answers; keys={sorted(row)}")
    record = {"id": sample_id, "question": question, "answers": answers, "metric": metric}
    if name == 'mme':
        record['metadata'] = {
            'category': str(first(row, 'category', 'task', 'type', default='perception')),
            'image_id': str(first(row, 'image_id', 'imageId', default=sample_id)),
        }
    elif name == 'mmbench':
        record['metadata'] = {'category': str(first(row, 'category', default='unknown'))}
    return record


def prepare_gqa(
    args: argparse.Namespace, load_dataset: Any, excluded_ids: set[str]
) -> None:
    """Join GQA's separate balanced instruction and image configurations."""
    repo, instruction_config, split = REGISTRY["gqa"]
    output = args.output_root / "gqa"
    image_dir = output / "images"
    write_exclusion_ledger(output, excluded_ids)
    image_dir.mkdir(parents=True, exist_ok=True)

    images = load_dataset(repo, GQA_IMAGE_CONFIG, split=split, streaming=args.streaming)
    images = (
        images.shuffle(seed=args.seed, buffer_size=1000)
        if args.streaming
        else images.shuffle(seed=args.seed)
    )
    # GQA stores questions and images in separate configurations.  Select a
    # bounded surplus of images when exclusions are active so skipped old
    # question IDs do not leave the requested held-out set undersized.
    image_limit = (
        args.max_samples + min(len(excluded_ids), args.max_samples)
        if args.max_samples is not None
        else None
    )
    image_rows = itertools.islice(images, image_limit) if image_limit is not None else images
    image_paths: dict[str, Path] = {}
    for row in tqdm(
        image_rows,
        total=image_limit,
        desc="gqa images",
        unit="image",
        mininterval=1.0,
    ):
        image_id = str(row["id"])
        image = first(row, "image", "decoded_image")
        if not isinstance(image, Image.Image):
            raise TypeError(f"GQA image {image_id} is not a decoded PIL image")
        image_path = image_dir / image_filename(image_id, len(image_paths))
        if not image_path.exists():
            image.convert("RGB").save(image_path, quality=95)
        image_paths[image_id] = image_path
    if image_limit is not None and len(image_paths) != image_limit:
        raise RuntimeError(f"Expected {image_limit} GQA images, found {len(image_paths)}")

    instructions = load_dataset(repo, instruction_config, split=split, streaming=args.streaming)
    instructions = (
        instructions.shuffle(seed=args.seed, buffer_size=1000)
        if args.streaming
        else instructions.shuffle(seed=args.seed)
    )
    selected = 0
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        with tqdm(
            total=args.max_samples,
            desc="gqa questions",
            unit="sample",
            mininterval=1.0,
        ) as progress:
            for row in instructions:
                image_id = str(row["imageId"])
                if image_id not in image_paths:
                    continue
                record = convert("gqa", row, selected)
                if record["id"] in excluded_ids:
                    continue
                record["image"] = str(image_paths[image_id].relative_to(output))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                selected += 1
                progress.update(1)
                if args.max_samples is not None and selected >= args.max_samples:
                    break
    if args.max_samples is not None and selected != args.max_samples:
        raise RuntimeError(
            f"Found only {selected} GQA questions for {len(image_paths)} selected images"
        )
    manifest = {
        "source": repo,
        "config": instruction_config,
        "image_config": GQA_IMAGE_CONFIG,
        "split": split,
        "samples": selected,
        "streaming": args.streaming,
        "seed": args.seed,
        "excluded_jsonl": [str(path) for path in args.exclude_jsonl],
        "excluded_id_ledgers": [str(path) for path in getattr(args, "exclude_ids_json", [])],
        "excluded_ids": len(excluded_ids),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=sorted(REGISTRY))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-library-warnings",
        action="store_true",
        help="show optional datasets/Hugging Face warnings (hidden by default)",
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="repeatable JSONL whose sample IDs must be excluded from the new split",
    )
    parser.add_argument(
        "--exclude-ids-json",
        type=Path,
        action="append",
        default=[],
        help="repeatable JSON-list ledger of IDs that must be excluded",
    )
    args = parser.parse_args()
    configure_library_output(args.show_library_warnings)
    excluded_ids = read_excluded_ids(args.exclude_jsonl, args.exclude_ids_json)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install evaluation dependencies: pip install -e '.[eval]'") from exc

    if args.benchmark == "gqa":
        prepare_gqa(args, load_dataset, excluded_ids)
        return

    repo, config, split = REGISTRY[args.benchmark]
    dataset = load_dataset(repo, config, split=split, streaming=args.streaming)
    if args.streaming:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=1000)
        iterator = iter(dataset)
        limit = args.max_samples
    else:
        dataset = dataset.shuffle(seed=args.seed)
        available = len(dataset) - len(excluded_ids)
        limit = min(available, args.max_samples) if args.max_samples else available
        iterator = (dataset[index] for index in range(len(dataset)))
    output = args.output_root / args.benchmark
    write_exclusion_ledger(output, excluded_ids)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = 0
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        with tqdm(total=limit, desc=args.benchmark, unit="sample", mininterval=1.0) as progress:
            for index, row in enumerate(iterator):
                record = convert(args.benchmark, row, index)
                if record["id"] in excluded_ids:
                    continue
                image = first(row, "image", "decoded_image")
                if not isinstance(image, Image.Image):
                    raise TypeError(f"row {index} has no decoded PIL image; keys={sorted(row)}")
                image_path = image_dir / image_filename(record['id'], index)
                image.convert("RGB").save(image_path, quality=95)
                record["image"] = str(image_path.relative_to(output))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                selected += 1
                progress.update(1)
                if limit is not None and selected >= limit:
                    break
    if limit is not None and selected != limit:
        raise RuntimeError(
            f"Found only {selected} non-excluded samples; requested {limit}"
        )
    manifest = {
        "source": repo,
        "config": config,
        "split": split,
        "samples": selected,
        "streaming": args.streaming,
        "seed": args.seed,
        "excluded_jsonl": [str(path) for path in args.exclude_jsonl],
        "excluded_id_ledgers": [str(path) for path in getattr(args, "exclude_ids_json", [])],
        "excluded_ids": len(excluded_ids),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
