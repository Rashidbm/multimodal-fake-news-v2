"""Qwen3-VL-Embedding semantic arm without the real model or dataset.

A stub ``scripts/qwen3_vl_embedding.py`` in a temporary model directory stands in
for the official embedder, so loading, encoding, caching, sharding, merging and
the downstream fit/evaluate/export run through the real code paths.
"""
import csv
import json

import joblib
import numpy as np
import pytest
import torch
from PIL import Image

from fnd import cache_qwen_embedding as qwen_cache
from fnd import fit_semantic_arm
from fnd.fit_semantic_fusion import feature_matrix, load_cache
from fnd.predict_semantic_fusion import SemanticFusionPredictor
from fnd.train_clip_matcher import digest, read_rows

STUB = '''
import hashlib
import numpy as np
import torch
from PIL import Image

DIM = {dim}


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    @property
    def dtype(self):
        return self.weight.dtype


class Qwen3VLEmbedder:
    def __init__(self, model_name_or_path, min_pixels=4096, max_pixels=1843200,
                 default_instruction="Represent the user's input.", **kwargs):
        assert kwargs.pop('local_files_only') is True
        self.min_pixels, self.max_pixels, self.default_instruction = min_pixels, max_pixels, default_instruction
        self.model = _Model().to(kwargs.pop('dtype'))

    def process(self, inputs, normalize=True):
        vectors = []
        for item in inputs:
            assert set(item) == {{'image', 'text'}} and isinstance(item['image'], Image.Image) and item['image'].mode == 'RGB'
            key = hashlib.sha256(item['text'].encode() + item['image'].tobytes()).digest()
            vectors.append(np.random.default_rng(list(key[:8])).standard_normal(DIM))
        out = torch.tensor(np.array(vectors), dtype=torch.float32)
        return torch.nn.functional.normalize(out, dim=-1) if normalize else out
'''


@pytest.fixture
def qwen_dir(tmp_path, monkeypatch):
    model = tmp_path/'Qwen3-VL-Embedding-stub'
    (model/'scripts').mkdir(parents=True)
    (model/'config.json').write_text('{"model_type": "qwen3_vl"}')
    (model/'scripts/qwen3_vl_embedding.py').write_text(STUB.format(dim=24))
    monkeypatch.setenv('QWEN_MODEL_PATH', str(model))
    for name in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
        monkeypatch.setenv(name, '1')
    return model


def write_csv(path, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return path


@pytest.fixture
def image_csv(tmp_path):
    rows = []
    for i in range(7):
        image = tmp_path/f'img{i}.png'
        Image.fromarray(np.full((12, 9+i, 3), 30*i, dtype=np.uint8)).save(image)
        rows.append(dict(sample_id=f's{i}', split='train' if i < 5 else 'val', text=f'caption {i}', image_path=str(image)))
    return write_csv(tmp_path/'pairs.csv', rows)


def test_extract_shards_merge_and_load_cache(tmp_path, qwen_dir, image_csv):
    full, parts = tmp_path/'full.pt', [tmp_path/'shard0.pt', tmp_path/'shard1.pt']
    qwen_cache.main(['--csv', str(image_csv), '--out', str(full), '--batch-size', '3'])
    for index, part in enumerate(parts):
        qwen_cache.main(['--csv', str(image_csv), '--out', str(part), '--shard', f'{index}/2'])
    merged = tmp_path/'merged.pt'
    qwen_cache.main(['--csv', str(image_csv), '--out', str(merged), '--merge', str(parts[1]), str(parts[0])])

    rows = read_rows(image_csv)
    cache = load_cache(merged, image_csv, rows)             # the loader every semantic cache uses
    assert cache['sample_ids'] == [r['sample_id'] for r in rows]
    assert cache['features'].shape == (7, 24) and cache['embedding_dim'] == 24
    assert torch.allclose(cache['features'], load_cache(full, image_csv, rows)['features'], atol=1e-6)
    assert torch.allclose(cache['features'].norm(dim=1), torch.ones(7), atol=1e-5)
    assert cache['feature_kind'] == 'qwen3_vl_embedding' and cache['instruction'] == "Represent the user's input."
    assert cache['model_config_sha256'] == digest(qwen_dir/'config.json') and cache['shard'] is None
    assert json.loads(merged.with_suffix('.json').read_text())['embedding_dim'] == 24

    with pytest.raises(ValueError, match='row order'):
        load_cache(merged, image_csv, rows[::-1])
    with pytest.raises(FileExistsError):
        qwen_cache.main(['--csv', str(image_csv), '--out', str(full)])
    with pytest.raises(ValueError, match='one file per shard'):
        qwen_cache.main(['--csv', str(image_csv), '--out', str(tmp_path/'bad.pt'), '--merge', str(parts[0])])


def test_changed_csv_is_rejected(tmp_path, qwen_dir, image_csv):
    parts = [tmp_path/'a.pt', tmp_path/'b.pt']
    for index, part in enumerate(parts):
        qwen_cache.main(['--csv', str(image_csv), '--out', str(part), '--shard', f'{index}/2'])
    rows = read_rows(image_csv)
    rows[0]['text'] = 'edited caption'
    edited = write_csv(tmp_path/'edited.csv', rows)
    with pytest.raises(ValueError, match='different CSV'):
        qwen_cache.merge(parts, edited, ['train', 'val'], tmp_path/'merged.pt')
    with pytest.raises(ValueError, match='provenance'):
        load_cache(parts[0], edited, rows[:4])


def test_smoke_writes_and_checks(tmp_path, qwen_dir, image_csv, capsys):
    out = tmp_path/'smoke.pt'
    qwen_cache.main(['--csv', str(image_csv), '--out', str(out), '--smoke', '3', '--batch-size', '3', '--allow-cpu'])
    assert 'SMOKE PASSED' in capsys.readouterr().out
    assert torch.load(out, weights_only=True)['features'].shape == (3, 24)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match='CUDA'):
            qwen_cache.main(['--csv', str(image_csv), '--out', str(tmp_path/'x.pt'), '--smoke', '2'])


def test_missing_model_or_images_fail_early(tmp_path, image_csv, monkeypatch):
    monkeypatch.delenv('QWEN_MODEL_PATH', raising=False)
    with pytest.raises(ValueError, match='QWEN_MODEL_PATH'):
        qwen_cache.load_embedder()
    rows = read_rows(image_csv)
    rows[2]['image_path'] = str(tmp_path/'absent.png')
    with pytest.raises(FileNotFoundError, match='1 images not found'):
        qwen_cache.main(['--csv', str(write_csv(tmp_path/'missing.csv', rows)), '--out', str(tmp_path/'o.pt')])


def test_path_prefix_rewrites_image_locations(tmp_path, qwen_dir, image_csv):
    rows = read_rows(image_csv)
    for row in rows:
        row['image_path'] = row['image_path'].replace(str(tmp_path), '/elsewhere')
    moved = write_csv(tmp_path/'moved.csv', rows)
    out = tmp_path/'moved.pt'
    qwen_cache.main(['--csv', str(moved), '--out', str(out), '--path-prefix', f'/elsewhere={tmp_path}'])
    assert load_cache(out, moved, rows)['features'].shape == (7, 24)


# --------------------------------------------------------------------------- feature assembly

def encoder_caches(n, qwen_dim, seed=0):
    rng = np.random.default_rng(seed)
    t = lambda *shape: torch.tensor(rng.standard_normal(shape), dtype=torch.float32)
    image, text = torch.nn.functional.normalize(t(n, 768), dim=1), torch.nn.functional.normalize(t(n, 768), dim=1)
    return (dict(hidden=t(n, 768), logit=t(n)), dict(hidden=t(n, 768), logit=t(n)),
            dict(image=image, text=text), dict(features=t(n, qwen_dim)))


def test_clip_kind_is_unchanged():
    blip, v1, clip, _ = encoder_caches(5, 8)
    x = feature_matrix(blip, v1, 'blip_v1_clip_large', clip)
    i, t = clip['image'].numpy(), clip['text'].numpy()
    expected = np.concatenate([blip['hidden'], v1['hidden'], blip['logit'][:, None], v1['logit'][:, None],
                               i, t, i*t, np.abs(i-t), (i*t).sum(1, keepdims=True)], axis=1)
    assert x.shape == (5, 4611)
    np.testing.assert_array_equal(x, expected)


@pytest.mark.parametrize('dim', [24, 4096])
def test_qwen_kind_width_is_1538_plus_d(dim):
    blip, v1, _, qwen = encoder_caches(4, dim)
    x = feature_matrix(blip, v1, 'blip_v1_qwen_embedding', qwen=qwen)
    assert x.shape == (4, 1538+dim)
    np.testing.assert_array_equal(x[:, :1538], feature_matrix(blip, v1, 'blip_v1_features'))
    np.testing.assert_array_equal(x[:, 1538:], qwen['features'].numpy())


def test_qwen_kind_rejects_missing_or_misaligned_features():
    blip, v1, clip, qwen = encoder_caches(4, 8)
    with pytest.raises(ValueError, match='Qwen3-VL-Embedding'):
        feature_matrix(blip, v1, 'blip_v1_qwen_embedding', clip_large=clip)
    with pytest.raises(ValueError, match='aligned'):
        feature_matrix(blip, v1, 'blip_v1_qwen_embedding', qwen=dict(features=qwen['features'][:3]))


# --------------------------------------------------------------------------- fit / evaluate / export

def semantic_rows(split, captions, start=0):
    """Five-scenario rows plus caption-paired genuine/OOC news rows, as the selection rule expects."""
    rows = []
    for c in range(captions):
        for scenario in range(1, 6):
            rows.append(dict(sample_id=f'{split}_v{start+c}_{scenario}', split=split, scenario=scenario,
                             label_binary=int(scenario != 4), evaluation_group='v1_five_scenarios', caption_id=f'{split}_v{start+c}'))
        for scenario in (1, 4):
            rows.append(dict(sample_id=f'{split}_n{start+c}_{scenario}', split=split, scenario=scenario,
                             label_binary=int(scenario != 4), evaluation_group='paired_news', caption_id=f'{split}_n{start+c}'))
    return rows


def write_caches(directory, name, csv_path, rows, qwen_dim=40):
    """Synthetic encoder caches in the exact on-disk formats, with a learnable class signal."""
    blip, v1, clip, qwen = encoder_caches(len(rows), qwen_dim, seed=len(rows))
    labels = torch.tensor([0 if int(r['scenario']) == 4 else 1 if int(r['scenario']) == 1 else 2 for r in rows])
    for block in (blip['hidden'], v1['hidden'], qwen['features']):
        block[:, :3] += 2.5*torch.nn.functional.one_hot(labels, 3)
    common = dict(sample_ids=[r['sample_id'] for r in rows], csv_sha256=digest(csv_path))
    paths = {}
    for key, payload in [('blip', {**common, **blip, 'model_name': 'blip'}), ('v1', {**common, **v1}),
                         ('clip', {**common, **clip, 'model_name': 'clip'}),
                         ('qwen', {**common, **qwen, 'feature_kind': 'qwen3_vl_embedding', 'embedding_dim': qwen_dim,
                                   'model_name': 'stub', 'model_config_sha256': 'c', 'embedder_script_sha256': 's',
                                   'instruction': "Represent the user's input.", 'min_pixels': 4096,
                                   'max_pixels': 1843200, 'dtype': 'bfloat16'})]:
        paths[key] = directory/f'{name}_{key}.pt'
        torch.save(payload, paths[key])
    return paths


@pytest.fixture
def experiment(tmp_path):
    train_dev = semantic_rows('train', 12)+semantic_rows('val', 5)
    extra = semantic_rows('train', 4, start=100)
    extra = [r for r in extra if r['evaluation_group'] == 'paired_news']        # complete caption pairs
    train_dev_csv = write_csv(tmp_path/'train_dev.csv', train_dev+extra)
    extra_csv = write_csv(tmp_path/'extra.csv', extra)
    test_csv = write_csv(tmp_path/'test.csv', semantic_rows('test', 4))
    return dict(dir=tmp_path, csv=train_dev_csv, extra=extra_csv, test=test_csv,
                caches=write_caches(tmp_path, 'dev', train_dev_csv, read_rows(train_dev_csv)),
                test_caches=write_caches(tmp_path, 'test', test_csv, read_rows(test_csv)))


def fit_arm(e, third, *extra_args):
    out = e['dir']/f'fit_{third}'
    key = 'qwen' if third == 'qwen_embedding' else 'clip'
    fit_semantic_arm.main(['fit', '--third', third, '--csv', str(e['csv']), '--blip', str(e['caches']['blip']),
                           '--v1', str(e['caches']['v1']), '--third-cache', str(e['caches'][key]), '--out', str(out), *extra_args])
    return out


def test_both_arms_fit_evaluate_and_qwen_exports_768(experiment, tmp_path):
    e = experiment
    diagnostic, manifest = tmp_path/'diagnostic.json', tmp_path/'manifest.json'
    diagnostic.write_text(json.dumps({'recalls': {f'v1_five_scenarios/s{s}': 0. for s in range(1, 6)}}))
    manifest.write_text(json.dumps({'reference_utility': 0.}))
    clip_dir = fit_arm(e, 'clip_large', '--extra-csv', str(e['extra']),
                       '--reference-diagnostic', str(diagnostic), '--reference-manifest', str(manifest))
    qwen_dir = fit_arm(e, 'qwen_embedding')

    clip_results = json.loads((clip_dir/'results.json').read_text())
    assert [r['name'] for r in clip_results] == [f'extra{f}_ooc{m}_c{c}' for f in ['0.5', '1'] for m in ['0.125', '0.25']
                                                 for c in ['0.001', '0.01']]
    qwen_selection = json.loads((qwen_dir/'selection.json').read_text())
    assert qwen_selection['feature_dim'] == 1538+40 and qwen_selection['kind'] == 'blip_v1_qwen_embedding'
    assert json.loads((clip_dir/'selection.json').read_text())['feature_dim'] == 4611
    assert joblib.load(qwen_dir/qwen_selection['artifact'])['model'][0].n_features_in_ == 1578
    assert len(json.loads((qwen_dir/'results.json').read_text())) == 4

    evaluation = tmp_path/'evaluation'
    fit_semantic_arm.main(['evaluate', '--eval-csv', str(e['test']), '--blip', str(e['test_caches']['blip']),
                           '--v1', str(e['test_caches']['v1']), '--out', str(evaluation),
                           '--arm', 'clip_large', str(clip_dir), str(e['test_caches']['clip']),
                           '--arm', 'qwen_embedding', str(qwen_dir), str(e['test_caches']['qwen'])])
    report = json.loads((evaluation/'evaluation.json').read_text())
    assert report['rows'] == 28 and report['qwen_embedding']['metrics']['auc'] > 0.75   # learned signal; chance is 0.5
    assert 'paired_difference_qwen_minus_clip' in report and set(report['scenario_accuracy_gain_qwen_minus_clip']) == set('12345')

    # The recorded-baseline check passes on the arm's own predictions and fails on anything else.
    recorded = evaluation/'clip_large_predictions.csv'
    fit_semantic_arm.main(['evaluate', '--eval-csv', str(e['test']), '--blip', str(e['test_caches']['blip']),
                           '--v1', str(e['test_caches']['v1']), '--out', str(tmp_path/'reproduced'),
                           '--arm', 'clip_large', str(clip_dir), str(e['test_caches']['clip']), '--expect-predictions', str(recorded)])
    rows = read_rows(recorded); rows[0]['prob'] = '0.123'
    with pytest.raises(ValueError, match='not reproduced'):
        fit_semantic_arm.main(['evaluate', '--eval-csv', str(e['test']), '--blip', str(e['test_caches']['blip']),
                               '--v1', str(e['test_caches']['v1']), '--out', str(tmp_path/'bad'),
                               '--arm', 'clip_large', str(clip_dir), str(e['test_caches']['clip']),
                               '--expect-predictions', str(write_csv(tmp_path/'edited.csv', rows))])

    bundle_dir = tmp_path/'bundle'
    fit_semantic_arm.main(['export', '--fit-dir', str(qwen_dir), '--csv', str(e['csv']), '--blip', str(e['caches']['blip']),
                           '--v1', str(e['caches']['v1']), '--third-cache', str(e['caches']['qwen']), '--out', str(bundle_dir)])
    bundle = json.loads((bundle_dir/'bundle.json').read_text())
    assert bundle['feature_kind'] == 'blip_v1_qwen_embedding' and 'clip_large' not in bundle
    assert bundle['qwen_embedding']['embedding_dim'] == 40 and bundle['qwen_embedding']['model_path_env'] == 'QWEN_MODEL_PATH'
    assert bundle['v1_checkpoint']['sha256'] == json.loads((fit_semantic_arm.ROOT/'models/semantic/bundle.json').read_text())['v1_checkpoint']['sha256']
    assert digest(bundle_dir/bundle['classifier']['path']) == bundle['classifier']['sha256']

    # v_semantic as the predictor computes it: scaler(x) @ basis.T, logits preserved.
    rows = read_rows(e['csv'])
    caches = [load_cache(e['caches'][k], e['csv'], rows) for k in ['blip', 'v1', 'qwen']]
    x = feature_matrix(caches[0], caches[1], bundle['feature_kind'], qwen=caches[2])
    saved = joblib.load(bundle_dir/bundle['classifier']['path'])
    projection = np.load(bundle_dir/'semantic_projection.npz')
    semantic = saved['model'][0].transform(x)@projection['basis'].T
    assert semantic.shape == (len(rows), 768)
    np.testing.assert_allclose(semantic@projection['coefficients'].T+projection['intercept'],
                               saved['model'].decision_function(x), atol=1e-6)


def test_caches_and_arms_cannot_be_mixed(experiment, tmp_path):
    e = experiment
    rows = read_rows(e['csv'])
    with pytest.raises(ValueError, match='not a Qwen3-VL-Embedding cache'):
        fit_semantic_arm.features_for('qwen_embedding', rows, e['csv'], e['caches']['blip'], e['caches']['v1'], e['caches']['clip'])
    with pytest.raises(ValueError, match='not a CLIP-L cache'):
        fit_semantic_arm.features_for('clip_large', rows, e['csv'], e['caches']['blip'], e['caches']['v1'], e['caches']['qwen'])
    qwen_dir = fit_arm(e, 'qwen_embedding')
    with pytest.raises(ValueError, match='fitted for qwen_embedding'):
        fit_semantic_arm.main(['evaluate', '--eval-csv', str(e['test']), '--blip', str(e['test_caches']['blip']),
                               '--v1', str(e['test_caches']['v1']), '--out', str(tmp_path/'mixed'),
                               '--arm', 'clip_large', str(qwen_dir), str(e['test_caches']['clip'])])
    with pytest.raises(ValueError, match='baseline'):
        fit_semantic_arm.main(['export', '--fit-dir', str(qwen_dir), '--csv', str(e['csv']), '--blip', str(e['caches']['blip']),
                               '--v1', str(e['caches']['v1']), '--third-cache', str(e['caches']['qwen']),
                               '--out', str(fit_semantic_arm.ROOT/'models/semantic')])


@pytest.mark.parametrize('kind,encoders', [('blip_v1_qwen_embedding', {'clip_large': {}}),
                                           ('blip_v1_clip_large', {'qwen_embedding': {}}),
                                           ('blip_v1_qwen_embedding', {'qwen_embedding': {}, 'clip_large': {}})])
def test_predictor_rejects_kind_bundle_mismatch_before_loading_models(tmp_path, kind, encoders):
    classifier = tmp_path/'classifier.joblib'
    joblib.dump(dict(model=None, task='three_class', kind=kind, threshold=0.5), classifier)
    bundle = tmp_path/'bundle.json'
    bundle.write_text(json.dumps(dict(classifier=dict(path=classifier.name, sha256=digest(classifier)), threshold=0.5, **encoders)))
    with pytest.raises(ValueError, match='disagree'):
        SemanticFusionPredictor(bundle, device='cpu')
