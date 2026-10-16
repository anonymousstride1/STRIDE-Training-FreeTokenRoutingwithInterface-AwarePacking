import pytest

from stride.evaluation.benchmark_metrics import summarize_benchmark, token_f1


def test_token_f1_uses_best_reference():
    assert token_f1('red sports car', ['blue car', 'red car']) == pytest.approx(0.8)


def test_pope_reports_binary_f1_and_accuracy():
    rows = [
        {'prediction': 'yes', 'answers': ['yes']},
        {'prediction': 'no', 'answers': ['yes']},
        {'prediction': 'no', 'answers': ['no']},
    ]
    metrics = summarize_benchmark(rows, 'pope')
    assert metrics['accuracy'] == pytest.approx(2 / 3)
    assert metrics['f1'] == pytest.approx(2 / 3)


def test_mme_uses_acc_plus_and_ignores_cognition_categories():
    rows = [
        {'id': '1a', 'prediction': 'yes', 'answers': ['yes'], 'metadata': {'category': 'existence', 'image_id': '1'}},
        {'id': '1b', 'prediction': 'no', 'answers': ['no'], 'metadata': {'category': 'existence', 'image_id': '1'}},
        {'id': '2a', 'prediction': 'no', 'answers': ['yes'], 'metadata': {'category': 'existence', 'image_id': '2'}},
        {'id': 'x', 'prediction': 'yes', 'answers': ['yes'], 'metadata': {'category': 'commonsense_reasoning', 'image_id': 'x'}},
    ]
    assert summarize_benchmark(rows, 'mme')['perception_score'] == pytest.approx(200)


def test_mme_recovers_legacy_pil_ids_and_excludes_incomplete_or_ambiguous_groups():
    rows = [
        {'id': 'paired.jpg', 'prediction': 'yes', 'answers': ['yes'], 'metadata': {'category': 'existence', 'image_id': '<PIL.Image.Image at 0x123>'}},
        {'id': 'paired.jpg', 'prediction': 'yes', 'answers': ['no'], 'metadata': {'category': 'existence', 'image_id': '<PIL.Image.Image at 0x456>'}},
        {'id': 'singleton.jpg', 'prediction': 'yes', 'answers': ['yes'], 'metadata': {'category': 'existence'}},
    ]
    assert summarize_benchmark(rows, 'mme')['perception_score'] == pytest.approx(50)
    assert summarize_benchmark(rows[2:], 'mme')['perception_score'] == 0
    assert summarize_benchmark(rows[:2] + rows[:1], 'mme')['perception_score'] == 0
