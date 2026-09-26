import pytest

from fnd.evaluate_alignment import pair_values, paired_summary, paired_difference


def examples():
    metadata = [{'sample_id': f'{caption}_{scenario}', 'caption_id': caption, 'scenario': scenario}
                for caption in ['a', 'b'] for scenario in [4, 1]]
    predictions = [{'sample_id': row['sample_id'], 'prob': p}
                   for row, p in zip(metadata, [.2, .8, .6, .6])]
    return metadata, predictions


def test_paired_scoring_distinguishes_ranking_from_both_correct():
    metadata, predictions = examples()
    result = paired_summary(predictions, metadata, .5)
    assert result['genuine_recall'] == .5
    assert result['ooc_recall'] == 1
    assert result['both_correct'] == .5
    assert result['within_caption_ranking'] == .75  # one correct ordering and one tie
    assert result['mean_probability_gap'] == pytest.approx(.3)
    # A prediction exactly at the threshold is fake, consistent with score_rows.
    assert paired_summary(predictions, metadata, .6)['genuine_recall'] == .5


def test_bootstrap_resamples_caption_pairs_and_uses_model_specific_thresholds():
    metadata, predictions = examples()
    result = paired_difference(predictions, predictions, metadata, .5, .5, repetitions=50)
    assert all(v['difference'] == 0 and v['ci95'] == [0, 0] for v in result['metrics'].values())
    changed = paired_difference(predictions, predictions, metadata, .5, .7, repetitions=50)
    assert changed['metrics']['genuine_recall']['difference'] == .5
    assert changed['metrics']['ooc_recall']['difference'] == -.5


def test_pair_scoring_rejects_missing_duplicate_or_unpaired_rows():
    metadata, predictions = examples()
    with pytest.raises(ValueError, match='exactly'):
        pair_values(predictions[:-1], metadata, .5)
    with pytest.raises(ValueError, match='duplicate scenario'):
        pair_values(predictions, [metadata[0], {**metadata[1], 'scenario': 4}, *metadata[2:]], .5)
    with pytest.raises(ValueError, match='each caption'):
        pair_values(predictions[:-1], metadata[:-1], .5)


def test_pair_ranking_uses_unsaturated_scores_when_available():
    from fnd.evaluate_alignment import paired_summary
    rows = [{'sample_id': 'a', 'caption_id': 'c', 'scenario': 4},
            {'sample_id': 'b', 'caption_id': 'c', 'scenario': 1}]
    predictions = [{'sample_id': 'a', 'prob': 1.0, 'ranking_score': 30.0},
                   {'sample_id': 'b', 'prob': 1.0, 'ranking_score': 40.0}]
    result = paired_summary(predictions, rows, 0.5)
    assert result['within_caption_ranking'] == 1.0
    assert result['genuine_recall'] == 0.0
    assert result['ooc_recall'] == 1.0
