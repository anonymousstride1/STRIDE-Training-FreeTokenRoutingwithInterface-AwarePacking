from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable

from .metrics import caption_rouge_l, caption_tokens, extract_choice, normalize_answer, score_prediction


BENCHMARK_METRICS = {
    'gqa': ('accuracy', 'exact_match', 'f1'),
    'mmbench': ('accuracy', 'f1'),
    'mme': ('perception_score', 'f1'),
    'pope': ('f1', 'accuracy'),
    'scienceqa': ('accuracy', 'exact_match', 'f1'),
    'textvqa': ('accuracy', 'exact_match', 'f1'),
    'sharegpt4v': ('caption_token_f1', 'caption_rouge_l_f1'),
}
MME_PERCEPTION_CATEGORIES = {
    'existence', 'count', 'position', 'color', 'posters', 'celebrity',
    'scene', 'landmark', 'artwork', 'ocr',
}


def canonical_benchmark(value: str) -> str:
    name = value.lower().replace('_', '').replace('-', '')
    aliases = {'scienceqaimg': 'scienceqa', 'scienceqa': 'scienceqa', 'textvqa': 'textvqa',
               'mmbench': 'mmbench', 'mme': 'mme', 'pope': 'pope', 'gqa': 'gqa'}
    return aliases.get(name, name)


def token_f1(prediction: str, references: Iterable[str]) -> float:
    pred = normalize_answer(prediction).split()
    best = 0.0
    for reference in references:
        ref = normalize_answer(reference).split()
        overlap = sum((Counter(pred) & Counter(ref)).values())
        if not pred and not ref:
            best = max(best, 1.0)
        elif overlap:
            precision, recall = overlap / len(pred), overlap / len(ref)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _binary_label(text: str) -> str:
    value = normalize_answer(text).split()
    return value[0] if value and value[0] in {'yes', 'no'} else ''


def mme_image_key(row: dict[str, Any]) -> str:
    """Recover legacy PIL-repr identifiers using the stable benchmark row ID."""
    value = (row.get('metadata') or {}).get('image_id')
    if value is None or str(value).startswith('<PIL.'):
        value = row.get('id')
    if value is None:
        raise ValueError('MME requires a stable image identifier')
    return str(value)


def mme_complete_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score only unambiguous yes/no pairs; never treat a singleton as Acc+."""
    groups = defaultdict(list)
    for row in rows:
        category = str((row.get('metadata') or {}).get('category', 'perception')).lower().replace(' ', '_')
        if category in MME_PERCEPTION_CATEGORIES or category == 'perception':
            groups[(category, mme_image_key(row))].append(row)
    return [row for group in groups.values()
            if len(group) == 2 and {_binary_label(r['answers'][0]) for r in group} == {'yes', 'no'}
            for row in group]


def _binary_f1(rows: list[dict[str, Any]]) -> float:
    tp = fp = fn = 0
    for row in rows:
        pred = _binary_label(row['prediction'])
        target = _binary_label(row['answers'][0])
        tp += pred == target == 'yes'
        fp += pred == 'yes' and target != 'yes'
        fn += pred != 'yes' and target == 'yes'
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def _macro_choice_f1(rows: list[dict[str, Any]]) -> float:
    labels = sorted({extract_choice(a) for row in rows for a in row['answers']} - {''})
    scores = []
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            pred, target = extract_choice(row['prediction']), extract_choice(row['answers'][0])
            tp += pred == target == label
            fp += pred == label and target != label
            fn += pred != label and target == label
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return mean(scores) if scores else 0.0


def summarize_benchmark(rows: list[dict[str, Any]], benchmark: str) -> dict[str, float]:
    """Benchmark metrics; MME uses the available complete-pair subset only."""
    name = canonical_benchmark(benchmark)
    if not rows:
        return {metric: 0.0 for metric in BENCHMARK_METRICS[name]}
    if name == 'sharegpt4v':
        return {
            'caption_token_f1': mean(score_prediction(r['prediction'], tuple(r['answers']), 'caption_token_f1') for r in rows),
            'caption_rouge_l_f1': mean(caption_rouge_l(r['prediction'], tuple(r['answers'])) for r in rows),
            'mean_prediction_words': mean(len(caption_tokens(r['prediction'])) for r in rows),
            'mean_reference_words': mean(len(caption_tokens(r['answers'][0])) for r in rows),
        }
    exact = mean(score_prediction(r['prediction'], tuple(r['answers']), 'exact_match') for r in rows)
    if name in {'pope', 'mme'}:
        accuracy = mean(score_prediction(r['prediction'], tuple(r['answers']), 'yes_no') for r in rows)
        f1 = _binary_f1(rows)
        if name == 'pope':
            return {'f1': f1, 'accuracy': accuracy}
        rows = mme_complete_pairs(rows)
        # Official MME perception: category Acc + Acc+ (both questions for an image correct), x100.
        categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            metadata = row.get('metadata') or {}
            category = str(metadata.get('category', 'perception')).lower().replace(' ', '_')
            if category in MME_PERCEPTION_CATEGORIES or category == 'perception':
                categories[category].append(row)
        perception_rows = [row for category_rows in categories.values() for row in category_rows]
        f1 = _binary_f1(perception_rows)
        perception = 0.0
        for category_rows in categories.values():
            correct = [score_prediction(r['prediction'], tuple(r['answers']), 'yes_no') for r in category_rows]
            by_image: dict[str, list[float]] = defaultdict(list)
            for row, value in zip(category_rows, correct):
                metadata = row.get('metadata') or {}
                by_image[mme_image_key(row)].append(value)
            acc_plus = mean(float(all(values)) for values in by_image.values())
            perception += 100 * (mean(correct) + acc_plus)
        return {'perception_score': perception, 'f1': f1}
    if name in {'scienceqa', 'mmbench'}:
        accuracy = mean(score_prediction(r['prediction'], tuple(r['answers']), 'multiple_choice') for r in rows)
        result = {'accuracy': accuracy, 'f1': _macro_choice_f1(rows)}
        if name == 'scienceqa':
            result['exact_match'] = exact
        return result
    f1 = mean(token_f1(r['prediction'], r['answers']) for r in rows)
    if name == 'textvqa':
        accuracy = mean(score_prediction(r['prediction'], tuple(r['answers']), 'vqa_accuracy') for r in rows)
    else:
        accuracy = exact
    return {'accuracy': accuracy, 'exact_match': exact, 'f1': f1}
