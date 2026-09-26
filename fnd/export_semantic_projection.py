"""Export 768 fused features while preserving the selected linear classifier's logits."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .fit_semantic_fusion import feature_matrix, load_cache
from .train_clip_matcher import digest, read_rows


def fit_projection(z_train, coefficients, output_dim=768):
    """Reserve classifier directions, then add orthogonal training PCA directions."""
    _, _, right = np.linalg.svd(coefficients, full_matrices=False)
    discriminant = right  # Includes the entire coefficient row space, even if rank deficient.
    if output_dim < len(discriminant) or output_dim > z_train.shape[1]:
        raise ValueError('Projection dimension cannot preserve classifier directions')
    residual = z_train-(z_train@discriminant.T)@discriminant
    covariance = residual.T@residual/max(len(residual)-1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, np.argsort(eigenvalues)[::-1][:output_dim-len(discriminant)]].T
    principal -= (principal@discriminant.T)@discriminant
    # Orthonormalize numerical roundoff within the PCA complement.
    principal = np.linalg.qr(principal.T, mode='reduced')[0].T
    basis = np.concatenate([discriminant, principal], axis=0)
    np.testing.assert_allclose(basis@basis.T, np.eye(output_dim), atol=1e-8)
    np.testing.assert_allclose((coefficients@basis.T)@basis, coefficients, atol=1e-8)
    return basis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--blip', required=True)
    parser.add_argument('--v1', required=True)
    parser.add_argument('--clip-large')
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args(argv)
    bundle_path = Path(args.bundle)
    bundle = json.loads(bundle_path.read_text())
    classifier = joblib.load(bundle['classifier']['path'])
    rows = read_rows(args.csv)
    blip, v1 = load_cache(args.blip, args.csv, rows), load_cache(args.v1, args.csv, rows)
    large = load_cache(args.clip_large, args.csv, rows) if args.clip_large else None
    x = feature_matrix(blip, v1, classifier['kind'], large)
    scaler, linear = classifier['model']
    z = scaler.transform(x).astype(np.float64)
    train = np.array([i for i, row in enumerate(rows) if row['split'] == 'train'])
    with threadpool_limits(limits=2):
        basis = fit_projection(z[train], linear.coef_)
        semantic = z@basis.T
        weights = linear.coef_@basis.T
        logits = semantic@weights.T+linear.intercept_
        reference = z@linear.coef_.T+linear.intercept_
    error = float(np.max(np.abs(logits-reference)))
    if error > 1e-7:
        raise ValueError(f'Projection changed classifier logits by {error}')
    out = bundle_path.parent/'semantic_projection.npz'
    if out.exists():
        raise FileExistsError(out)
    np.savez(out, basis=basis, coefficients=weights, intercept=linear.intercept_, classes=linear.classes_)
    verification = dict(output_dimension=768, fit_rows=len(train), fit_split='train only',
                         evaluated_equivalence_rows=len(rows), maximum_absolute_logit_error=error,
                         classifier_sha256=digest(bundle['classifier']['path']), training_csv_sha256=digest(args.csv),
                         definition='Classifier row space plus orthogonal principal components fit only to training features. No evaluation fitting; full selected logits preserved.')
    out.with_suffix('.json').write_text(json.dumps(verification, indent=2))
    bundle['semantic_projection'] = dict(path=str(out.resolve()), sha256=digest(out))
    bundle['semantic_export'] = '768 joint features: classifier directions plus training-only PCA complement. Selected logits are preserved. New feature semantics require retraining downstream heads and new caches.'
    bundle_path.write_text(json.dumps(bundle, indent=2))
    lock_path = bundle_path.parent/'selection_lock.json'
    lock = json.loads(lock_path.read_text()); lock['candidate_bundle_sha256'] = digest(bundle_path)
    lock['semantic_export_verification'] = verification
    lock_path.write_text(json.dumps(lock, indent=2))
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
