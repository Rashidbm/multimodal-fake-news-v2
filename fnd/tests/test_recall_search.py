import numpy as np
import pytest

from fnd.recall_search import recall_operating_point


def test_recall_selection_requires_both_domains_and_avoids_all_fake():
    rows, scores = [], []
    for group, scenarios in [('v1_five_scenarios', [1, 2, 3, 4, 5]), ('paired_news', [1, 4])]:
        for scenario in scenarios:
            for p in ([.1, .3] if scenario == 4 else [.6, .9]):
                rows.append(dict(evaluation_group=group, scenario=scenario))
                scores.append(p)
    result = recall_operating_point(rows, scores, .95)
    assert result['eligible'] and result['minimum_recall'] == 1
    assert .3 < result['threshold'] <= .6
    # Reordering inputs must not change the operating point.
    reverse = recall_operating_point(rows[::-1], scores[::-1], .95)
    assert result == reverse
    with pytest.raises(ValueError, match='Missing validation group'):
        recall_operating_point(rows[:-4], scores[:-4], .95)


def test_impossible_utility_is_explicitly_ineligible():
    rows = [dict(evaluation_group=g, scenario=s) for g, ss in [('v1_five_scenarios', range(1, 6)), ('paired_news', [1, 4])] for s in ss]
    result = recall_operating_point(rows, np.full(len(rows), .5), .9)
    assert not result['eligible']
    assert result['minimum_recall'] == 0
