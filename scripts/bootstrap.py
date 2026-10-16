import numpy as np
from collections import defaultdict
from stride.evaluation.benchmark_metrics import MME_PERCEPTION_CATEGORIES
from stride.evaluation.metrics import score_prediction
TASKS=['gqa', 'mmbench', 'mme', 'pope', 'scienceqa', 'textvqa']
def bootstrap_task(task, sets, count=2000):
    rng = np.random.default_rng(20261011 + TASKS.index(task))
    n = len(sets[0])
    if task != 'mme':
        picks = rng.integers(n, size=(count, n))
        if task == 'pope':
            from stride.evaluation.benchmark_metrics import _binary_label
            truth = np.array([_binary_label(r['answers'][0]) == 'yes' for r in sets[0]])
            answer = [np.array([_binary_label(r['prediction']) == 'yes' for r in rows]) for rows in sets]
            scores = []
            for pred in answer:
                tp = (pred & truth)[picks].sum(axis=1)
                fp = (pred & ~truth)[picks].sum(axis=1)
                fn = (~pred & truth)[picks].sum(axis=1)
                scores.append(2 * tp / np.maximum(2 * tp + fp + fn, 1))
            return scores
        return [np.array([r['score'] for r in rows])[picks].mean(axis=1) for rows in sets]
    grouped = defaultdict(lambda : defaultdict(list))
    for (i, row) in enumerate(sets[0]):
        metadata = row.get('metadata') or {}
        category = str(metadata.get('category', 'perception')).lower().replace(' ', '_')
        if category in MME_PERCEPTION_CATEGORIES or category == 'perception':
            grouped[category][str(row['id'])].append(i)
    result = [np.zeros(count) for _ in sets]
    for groups in grouped.values():
        indices = [v for v in groups.values() if len(v) == 2]
        if not indices:
            continue
        picks = rng.integers(len(indices), size=(count, len(indices)))
        sizes = np.array([len(v) for v in indices])
        for (j, rows) in enumerate(sets):
            correctness = [[score_prediction(rows[i]['prediction'], tuple(rows[i]['answers']), 'yes_no') for i in ids] for ids in indices]
            sums = np.array([sum(v) for v in correctness])
            both = np.array([float(all(v)) for v in correctness])
            result[j] += 100 * (sums[picks].sum(axis=1) / sizes[picks].sum(axis=1) + both[picks].mean(axis=1))
    return result
