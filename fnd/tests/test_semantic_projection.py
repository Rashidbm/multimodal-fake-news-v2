import numpy as np

from fnd.export_semantic_projection import fit_projection


def test_projected_representation_preserves_logits_on_unseen_features():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(80, 20))
    coefficients = rng.normal(size=(3, 20))
    basis = fit_projection(train, coefficients, output_dim=10)
    unseen = rng.normal(size=(12, 20))
    reconstructed = (unseen@basis.T)@(coefficients@basis.T).T
    np.testing.assert_allclose(reconstructed, unseen@coefficients.T, atol=1e-10)
    assert (unseen@basis.T).shape == (12, 10)
