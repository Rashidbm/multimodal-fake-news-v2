import numpy as np
from scipy.optimize import check_grad
from fnd.semantic_resolution import joint_objective, pair_indices
from fnd.semantic_resolution import ResidualHead
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def test_joint_gradient_including_shared_caption_ranking():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8, 5))
    y = np.array([0, 1, 0, 1, 2, 2, 0, 1])
    weights = rng.uniform(.2, 2, len(y))
    pairs = np.array([[0, 1], [2, 3], [6, 7]])
    theta = rng.normal(scale=.2, size=18)
    args = (x, y, weights, pairs, 3., .7)
    error = check_grad(lambda t: joint_objective(t, *args)[0],
                       lambda t: joint_objective(t, *args)[1], theta)
    assert error < 1e-6


def test_pairs_do_not_depend_on_row_order_or_assume_adjacent_rows():
    rows = [dict(evaluation_group='paired_news', caption_id=c, scenario=s)
            for c, s in [('b', '1'), ('a', '4'), ('b', '4'), ('a', '1')]]
    assert pair_indices(rows).tolist() == [[2, 0], [1, 3]]


def test_serialized_residual_head_matches_torch_inside_pipeline():
    import torch
    rng = np.random.default_rng(9)
    x = rng.normal(size=(7, 5)).astype('float32')
    params = {name: rng.normal(size=shape).astype('float32') for name, shape in {
        'hidden.weight': (4, 5), 'hidden.bias': (4,), 'output.weight': (3, 4),
        'output.bias': (3,), 'skip.weight': (3, 5), 'skip.bias': (3,)}.items()}
    scaler = StandardScaler().fit(x)
    z = torch.tensor(scaler.transform(x))
    p = {k: torch.tensor(v) for k, v in params.items()}
    hidden = torch.relu(z@p['hidden.weight'].T+p['hidden.bias'])
    logits = z@p['skip.weight'].T+p['skip.bias']+hidden@p['output.weight'].T+p['output.bias']
    actual = make_pipeline(scaler, ResidualHead(params)).predict_proba(x)
    np.testing.assert_allclose(actual, logits.softmax(-1).numpy(), atol=1e-6)
