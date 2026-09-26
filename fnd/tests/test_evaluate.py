import math

import pytest

from fnd.evaluate import choose_threshold, score_rows, validate_rows


def rows(labels, probabilities):
    return [{'label_binary': y, 'prob': p, 'scenario': 1 if y else 4}
            for y, p in zip(labels, probabilities)]


@pytest.mark.parametrize('metric', ['balanced_accuracy', 'f1_macro'])
def test_threshold_matches_exhaustive_search(metric):
    data = rows([0, 1, 1, 0, 1, 0, 1], [0.8, 0.999, 0.9, 0.85, 0.85, 0.1, 1.0])
    threshold = choose_threshold(data, metric)
    candidates = [0, 0.5] + [math.nextafter(r['prob'], math.inf) for r in data]
    assert score_rows(data, threshold)[metric] == max(score_rows(data, t)[metric] for t in candidates)


def test_threshold_tie_keeps_standard_operating_point():
    assert choose_threshold(rows([0, 1], [0.1, 0.9])) == 0.5


def test_missing_class_rejected():
    with pytest.raises(ValueError, match='both real and fake'):
        choose_threshold(rows([1], [0.9]))
