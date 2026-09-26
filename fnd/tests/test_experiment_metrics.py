"""Metrics must expose failure on genuine posts, including an always-fake predictor."""
from fnd.metrics import binary_metrics


def test_always_fake_does_not_look_balanced():
    metrics = binary_metrics([0, 1, 1, 1, 1], [0.9] * 5)
    assert metrics["accuracy"] == 0.8
    assert metrics["real_recall"] == 0
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["f1_macro"] < 0.5


def test_screenshot_confusion_matrix():
    labels = [1] * 988 + [0] * 247
    probabilities = [0.9] * 926 + [0.1] * 62 + [0.1] * 105 + [0.9] * 142
    metrics = binary_metrics(labels, probabilities)
    assert metrics["real_recall"] == 105 / 247
    assert metrics["balanced_accuracy"] == (926 / 988 + 105 / 247) / 2
    assert metrics["f1_macro"] == (1852 / 2056 + 210 / 414) / 2
