"""Fit, evaluate and export one semantic arm: FND + BLIP + (CLIP-L | Qwen3-VL-Embedding).

Both arms go through identical code: the ``semantic_more_data`` Logistic Regression
grid, ``protected_point`` validation selection, and ``fit_projection`` for the
768-d export. Only the third cached feature block differs.

    python -m fnd.fit_semantic_arm fit      --third qwen_embedding --csv ... --blip ... --v1 ... --third-cache ... --out DIR
    python -m fnd.fit_semantic_arm evaluate --eval-csv ... --blip ... --v1 ... --arm clip_large DIR CACHE --arm qwen_embedding DIR CACHE --out DIR
    python -m fnd.fit_semantic_arm export   --fit-dir DIR --csv ... --blip ... --v1 ... --third-cache ... --out BUNDLE_DIR

Nothing here reads test rows except ``evaluate``, which fits nothing.
"""
import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluate import score_rows
from .evaluate_alignment import paired_difference, paired_summary
from .export_semantic_projection import fit_projection
from .fit_semantic_fusion import feature_matrix, load_cache, metrics
from .semantic_resolution import protected_point
from .train_clip_matcher import digest, read_rows

ROOT = Path(__file__).resolve().parents[1]
KINDS = {'clip_large': 'blip_v1_clip_large', 'qwen_embedding': 'blip_v1_qwen_embedding'}
FIVE_GROUPS = [f'v1_five_scenarios/s{s}' for s in range(1, 6)]
# The grid and selection constants of fnd/semantic_more_data.py and scripts/lock_semantic_resolution.py.
OOC_MASSES, C_VALUES, BOOTSTRAP_SEED = [.125, .25], [.001, .01], 2026091403


def features_for(third, rows, csv_path, blip_path, v1_path, third_path):
    blip, v1, extra = (load_cache(p, csv_path, rows) for p in (blip_path, v1_path, third_path))
    if third == 'qwen_embedding' and extra.get('feature_kind') != 'qwen3_vl_embedding':
        raise ValueError('The third cache is not a Qwen3-VL-Embedding cache')
    if third == 'clip_large' and 'image' not in extra:
        raise ValueError('The third cache is not a CLIP-L cache')
    kind = KINDS[third]
    x = feature_matrix(blip, v1, kind, clip_large=extra if third == 'clip_large' else None,
                       qwen=extra if third == 'qwen_embedding' else None)
    return x, kind, extra


def three_class(rows):
    return np.array([0 if int(r['scenario']) == 4 else 1 if int(r['scenario']) == 1 else 2 for r in rows])


def fit(args):
    rows = read_rows(args.csv)
    if any(r['split'] not in ['train', 'val'] for r in rows):
        raise ValueError('fit accepts only train/val rows')
    out = Path(args.out)
    out.mkdir(parents=True)
    x, kind, extra = features_for(args.third, rows, args.csv, args.blip, args.v1, args.third_cache)
    y = three_class(rows)
    val = np.flatnonzero([r['split'] == 'val' for r in rows])
    val_rows = [rows[i] for i in val]
    # Case A: the original extra caption pairs, added in the recorded fractions. Case B: all training rows.
    extra_ids, extra_captions = set(), []
    if args.extra_csv:
        extra_rows = read_rows(args.extra_csv)
        extra_ids = {r['sample_id'] for r in extra_rows}
        extra_captions = list(dict.fromkeys(r['caption_id'] for r in extra_rows))
        if len(extra_captions)*2 != len(extra_rows) or not extra_ids <= {r['sample_id'] for r in rows if r['split'] == 'train'}:
            raise ValueError('Extra rows must be complete caption pairs inside the training split')
    if args.reference_diagnostic:
        floors = json.loads(Path(args.reference_diagnostic).read_text())['recalls']
        utility = json.loads(Path(args.reference_manifest).read_text())['reference_utility']
    else:
        floors, utility = {group: 0. for group in FIVE_GROUPS}, -np.inf
    results = []
    with threadpool_limits(limits=2):
        for fraction in ([.5, 1.] if extra_ids else [1.]):
            selected = set(extra_captions[:int(len(extra_captions)*fraction)])
            train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'
                              and (r['sample_id'] not in extra_ids or r['caption_id'] in selected)])
            counts = Counter(int(rows[i]['scenario']) for i in train)
            for mass in OOC_MASSES:
                masses = {4: .5, 1: mass, 2: (.5-mass)/3, 3: (.5-mass)/3, 5: (.5-mass)/3}
                weights = np.array([len(train)*masses[int(rows[i]['scenario'])]/counts[int(rows[i]['scenario'])] for i in train])
                for c in C_VALUES:
                    name = f'extra{fraction:g}_ooc{mass:g}_c{c:g}'
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1500, random_state=42))
                    model.fit(x[train], y[train], logisticregression__sample_weight=weights)
                    if model[-1].n_iter_.max() >= 1500:
                        raise RuntimeError(f'{name} did not converge')
                    scores = 1-model.predict_proba(x[val])[:, 0]
                    point = protected_point(val_rows, scores, floors, utility)
                    report, predictions = metrics(val_rows, scores, point['threshold'])
                    artifact = out/f'{name}.joblib'
                    joblib.dump(dict(model=model, task='three_class', kind=kind, threshold=point['threshold']), artifact)
                    with (out/f'{name}_val.csv').open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
                    results.append(dict(name=name, **point, metrics=report, training_rows=len(train),
                                        scenario_counts={str(k): v for k, v in counts.items()}, artifact=artifact.name))
                    print(f'{name}: eligible={point["eligible"]} minimum={point["minimum_recall"]:.4f} '
                          f'utility={point["utility"]:.4f}', flush=True)
    (out/'results.json').write_text(json.dumps(results, indent=2))
    eligible = [r for r in results if r['eligible']] or results
    best = max(eligible, key=lambda r: (round(r['minimum_recall'], 12), round(r['utility'], 12)))
    provenance = {k: v for k, v in extra.items() if k not in ['sample_ids', 'features', 'image', 'text']}
    selection = dict(third=args.third, kind=kind, selected=best['name'], threshold=best['threshold'],
                     artifact=best['artifact'], artifact_sha256=digest(out/best['artifact']),
                     feature_dim=int(x.shape[1]), csv_sha256=digest(args.csv), test_used=False,
                     caches={k: digest(p) for k, p in [('blip', args.blip), ('v1', args.v1), ('third', args.third_cache)]},
                     reference=('recorded incumbent floors' if args.reference_diagnostic else 'none (common new split)'),
                     third_provenance=provenance)
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps({k: selection[k] for k in ['third', 'selected', 'threshold', 'feature_dim']}, indent=2), flush=True)


def load_selection(directory):
    selection = json.loads((Path(directory)/'selection.json').read_text())
    path = Path(directory)/selection['artifact']
    if digest(path) != selection['artifact_sha256']:
        raise ValueError(f'Selected classifier changed: {path}')
    saved = joblib.load(path)
    if saved['kind'] != selection['kind']:
        raise ValueError('Classifier kind does not match its selection record')
    return selection, saved


def evaluate(args):
    rows = [r for r in read_rows(args.eval_csv) if not args.splits or r['split'] in args.splits]
    out = Path(args.out)
    out.mkdir(parents=True)
    keep = np.array([i for i, r in enumerate(rows) if not args.evaluation_group or r['evaluation_group'] == args.evaluation_group])
    subset = [rows[i] for i in keep]
    arms, report = {}, dict(eval_csv_sha256=digest(args.eval_csv), rows=len(subset), evaluation_group=args.evaluation_group)
    for third, directory, cache in args.arm:
        if third not in KINDS:
            raise ValueError(f'Unknown arm {third!r}; expected one of {sorted(KINDS)}')
        selection, saved = load_selection(directory)
        if selection['third'] != third:
            raise ValueError(f'{directory} was fitted for {selection["third"]}, not {third}')
        x, kind, _ = features_for(third, rows, args.eval_csv, args.blip, args.v1, cache)
        x = x[keep]
        probabilities = 1-saved['model'].predict_proba(x)[:, 0]
        logits = saved['model'].decision_function(x)
        rank = logsumexp(logits[:, 1:], axis=1)-logits[:, 0]
        predictions = [dict(sample_id=r['sample_id'], scenario=int(r['scenario']), label_binary=int(r['label_binary']),
                            prob=float(p), ranking_score=float(s)) for r, p, s in zip(subset, probabilities, rank)]
        with (out/f'{third}_predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
        result = dict(selected=selection['selected'], threshold=saved['threshold'], metrics=score_rows(predictions, saved['threshold']))
        paired = [r for r in subset if r.get('evaluation_group') == 'paired_news']
        if paired:
            ids = {r['sample_id'] for r in paired}
            result['paired'] = paired_summary([p for p in predictions if p['sample_id'] in ids], paired, saved['threshold'])
        if third == 'clip_large' and args.expect_predictions:
            recorded = {r['sample_id']: float(r['prob']) for r in read_rows(args.expect_predictions)}
            if set(recorded) != {p['sample_id'] for p in predictions}:
                raise ValueError('Recorded baseline predictions cover different rows')
            error = max(abs(p['prob']-recorded[p['sample_id']]) for p in predictions)
            if error >= 1e-10:
                raise ValueError(f'CLIP-L baseline not reproduced: max probability error {error}')
            result['baseline_reproduction_max_error'] = error
        arms[third] = (result, predictions)
        report[third] = result
    if len(arms) == 2:
        (clip, clip_predictions), (qwen, qwen_predictions) = arms['clip_large'], arms['qwen_embedding']
        report['scenario_accuracy_gain_qwen_minus_clip'] = {
            s: qwen['metrics']['per_scenario'][s]['accuracy']-clip['metrics']['per_scenario'][s]['accuracy']
            for s in clip['metrics']['per_scenario']}
        paired = [r for r in subset if r.get('evaluation_group') == 'paired_news']
        if paired:
            ids = {r['sample_id'] for r in paired}
            report['paired_difference_qwen_minus_clip'] = paired_difference(
                [p for p in clip_predictions if p['sample_id'] in ids], [p for p in qwen_predictions if p['sample_id'] in ids],
                paired, clip['threshold'], qwen['threshold'], repetitions=5000, seed=BOOTSTRAP_SEED)
    (out/'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def export(args):
    out = Path(args.out).resolve()
    if out == (ROOT/'models/semantic').resolve():
        raise ValueError('Refusing to write into the baseline models/semantic bundle')
    selection, saved = load_selection(args.fit_dir)
    rows = read_rows(args.csv)
    if digest(args.csv) != selection['csv_sha256']:
        raise ValueError('Export must use the CSV the arm was fitted on')
    x, kind, extra = features_for(selection['third'], rows, args.csv, args.blip, args.v1, args.third_cache)
    scaler, linear = saved['model']
    z = scaler.transform(x).astype(np.float64)
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    with threadpool_limits(limits=2):
        basis = fit_projection(z[train], linear.coef_)
        semantic = z@basis.T
        weights = linear.coef_@basis.T
        error = float(np.max(np.abs(semantic@weights.T-(z@linear.coef_.T))))
    if error > 1e-7 or semantic.shape[1] != 768:
        raise ValueError(f'Projection failed: shape {semantic.shape}, logit error {error}')
    out.mkdir(parents=True)
    classifier = out/Path(selection['artifact']).name
    shutil.copy2(Path(args.fit_dir)/selection['artifact'], classifier)
    projection = out/'semantic_projection.npz'
    np.savez(projection, basis=basis, coefficients=weights, intercept=linear.intercept_, classes=linear.classes_)
    template_path = Path(args.template_bundle).resolve()
    template = json.loads(template_path.read_text())

    def relocated(spec):
        source = Path(spec['path'])
        source = source if source.is_absolute() else template_path.parent/source
        try:
            return dict(spec, path=os.path.relpath(source, out))
        except ValueError:  # different Windows drives
            return dict(spec, path=str(source))

    bundle = dict(
        name=f'Semantic {selection["third"]} arm: {selection["selected"]}', feature_kind=kind,
        classifier=dict(path=classifier.name, sha256=digest(classifier)),
        v1_checkpoint=relocated(template['v1_checkpoint']),
        blip_model=template['blip_model'], blip_revision=template['blip_revision'],
        threshold=saved['threshold'],
        output='Binary real/fake; fake score is the sum of all non-genuine category probabilities',
        input='Image pixels and caption tokens only',
        training_csv=dict(path='Not required for inference', sha256=selection['csv_sha256']),
        semantic_projection=dict(path=projection.name, sha256=digest(projection)),
        semantic_export='768 joint features: classifier directions plus training-only PCA complement. '
                        'Selected logits are preserved. New feature semantics require retraining downstream heads and new caches.')
    if selection['third'] == 'clip_large':
        bundle['clip_large'] = template['clip_large']
    else:
        bundle['qwen_embedding'] = dict(model_path_env='QWEN_MODEL_PATH', embedding_dim=extra['embedding_dim'],
                                        **{k: extra[k] for k in ['feature_kind', 'model_name', 'model_config_sha256',
                                                                 'embedder_script_sha256', 'instruction', 'min_pixels',
                                                                 'max_pixels', 'dtype']})
    (out/'bundle.json').write_text(json.dumps(bundle, indent=2))
    (out/'ARTIFACTS.json').write_text(json.dumps(dict(weights_in_git=False, files=[
        dict(role=role, filename=path.name, size_bytes=path.stat().st_size, sha256=digest(path))
        for role, path in [('classifier', classifier), ('semantic_projection', projection)]]), indent=2))
    (out/'semantic_projection.json').write_text(json.dumps(dict(
        output_dimension=768, fit_rows=len(train), fit_split='train only', maximum_absolute_logit_error=error,
        classifier_sha256=digest(classifier), training_csv_sha256=selection['csv_sha256']), indent=2))
    print(f'COMPLETE {out/"bundle.json"}  semantic {semantic.shape}  logit error {error:.2e}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest='stage', required=True)
    fit_parser = stages.add_parser('fit', help='Fit the LR grid and select on validation only')
    fit_parser.add_argument('--third', choices=sorted(KINDS), required=True)
    fit_parser.add_argument('--csv', required=True, help='Train/validation CSV')
    fit_parser.add_argument('--third-cache', required=True)
    fit_parser.add_argument('--extra-csv', help='Case A: the recorded extra training caption pairs')
    fit_parser.add_argument('--reference-diagnostic', help='Case A: incumbent_diagnostic.json (recall floors)')
    fit_parser.add_argument('--reference-manifest', help='Case A: data_volume/manifest.json (reference utility)')
    evaluate_parser = stages.add_parser('evaluate', help='Score locked arms once; nothing is fitted')
    evaluate_parser.add_argument('--eval-csv', required=True)
    evaluate_parser.add_argument('--arm', nargs=3, action='append', required=True, metavar=('THIRD', 'FIT_DIR', 'EVAL_CACHE'),
                                 help='Repeatable, e.g. --arm clip_large DIR CACHE --arm qwen_embedding DIR CACHE')
    evaluate_parser.add_argument('--splits', nargs='+', help='Rows of --eval-csv to use; default all (must match the caches)')
    evaluate_parser.add_argument('--evaluation-group', help='Score only rows of this evaluation_group')
    evaluate_parser.add_argument('--expect-predictions', help='Case A: recorded CLIP-L predictions that must reproduce')
    export_parser = stages.add_parser('export', help='Fit the 768-d projection and write a separate bundle')
    export_parser.add_argument('--fit-dir', required=True)
    export_parser.add_argument('--csv', required=True, help='The train/validation CSV used by fit')
    export_parser.add_argument('--third-cache', required=True)
    export_parser.add_argument('--template-bundle', default=str(ROOT/'models/semantic/bundle.json'),
                               help='Source of the unchanged FND checkpoint and BLIP revision')
    for stage in [fit_parser, evaluate_parser, export_parser]:
        stage.add_argument('--blip', required=True)
        stage.add_argument('--v1', required=True)
        stage.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.stage == 'fit' and bool(args.reference_diagnostic) != bool(args.reference_manifest):
        parser.error('--reference-diagnostic and --reference-manifest go together')
    {'fit': fit, 'evaluate': evaluate, 'export': export}[args.stage](args)


if __name__ == '__main__':
    main()
