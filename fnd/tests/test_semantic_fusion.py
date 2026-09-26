import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnd.fit_semantic_fusion import training_weights, validation_weights, probability, choose_operating_point


def test_training_mass_balances_binary_classes_and_four_fake_scenarios():
    rows = [{'scenario': scenario} for scenario, count in [(1, 8), (2, 2), (3, 3), (4, 20), (5, 1)] for _ in range(count)]
    weights = training_weights(rows)
    for scenario in range(1, 6):
        mass = sum(w for r, w in zip(rows, weights) if r['scenario'] == scenario)/weights.sum()
        assert mass == pytest.approx(0.5 if scenario == 4 else 0.125)


def test_domain_selection_does_not_overweight_larger_validation_domain():
    rows = [dict(evaluation_group=g, label_binary=y) for g, n in [('v1_five_scenarios', 20), ('paired_news', 2)]
            for y in [0, 1] for _ in range(n)]
    weights = validation_weights(rows)
    for group in ['v1_five_scenarios', 'paired_news']:
        assert sum(w for r, w in zip(rows, weights) if r['evaluation_group'] == group) == pytest.approx(0.5)
    scores = np.array([0.1 if r['label_binary'] == 0 else 0.8 for r in rows])
    threshold = choose_operating_point(rows, scores)
    assert np.array_equal(scores >= threshold, [r['label_binary'] for r in rows])


def test_three_class_output_collapses_both_fake_types_not_one_column():
    x = np.array([[-3], [-2], [0], [1], [3], [4]], dtype=float)
    y = [0, 0, 1, 1, 2, 2]
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    actual = probability(model, x, 'three_class')
    np.testing.assert_allclose(actual, model.predict_proba(x)[:, 1:].sum(1))


def test_inference_rejects_threshold_different_from_evaluation(tmp_path):
    import hashlib
    import json
    import joblib
    from fnd.predict_semantic_fusion import SemanticFusionPredictor

    classifier = tmp_path/'classifier.joblib'
    joblib.dump({'threshold': 0.3}, classifier)
    bundle = tmp_path/'bundle.json'
    bundle.write_text(json.dumps({'threshold': 0.5, 'classifier': {
        'path': str(classifier), 'sha256': hashlib.sha256(classifier.read_bytes()).hexdigest()}}))
    with pytest.raises(ValueError, match='thresholds disagree'):
        SemanticFusionPredictor(bundle, device='cpu')
