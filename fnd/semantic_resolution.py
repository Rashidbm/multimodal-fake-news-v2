"""Finite, validation-only experiments addressing the remaining semantic errors."""
import argparse
import copy
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .fit_semantic_fusion import feature_matrix, load_cache, metrics
from .train_clip_matcher import digest, read_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/semantic_resolution'
CSV = ROOT/'data/processed/clip_adaptation/v1_blip_train_dev.csv'


def load_development():
    rows = read_rows(CSV)
    if any(r['split'] not in ['train', 'val'] for r in rows):
        raise ValueError('This experiment accepts only train/development rows')
    paths = [ROOT/p for p in ['outputs/clip_adaptation/blip_v1_train_dev.pt',
        'outputs/clip_adaptation/standardized_v1_train_dev.pt',
        'outputs/recall_improvement/clip_large_train_dev.pt']]
    blip, v1, clip = [load_cache(p, CSV, rows) for p in paths]
    x = feature_matrix(blip, v1, 'blip_v1_clip_large', clip)
    bundle = json.loads((ROOT/'outputs/recall_improvement/selected/bundle.json').read_text())
    if digest(bundle['classifier']['path']) != bundle['classifier']['sha256']:
        raise ValueError('Incumbent artifact changed')
    reference = joblib.load(bundle['classifier']['path'])
    return rows, x, clip, reference


def pair_indices(rows):
    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        if row['evaluation_group'] == 'paired_news':
            scenario = int(row['scenario'])
            if scenario in groups[row['caption_id']]:
                raise ValueError('Duplicate caption/scenario')
            groups[row['caption_id']][scenario] = i
    if not groups or any(set(pair) != {1, 4} for pair in groups.values()):
        raise ValueError('Need complete genuine/OOC pairs')
    return np.array([[p[4], p[1]] for p in groups.values()])


def protected_point(rows, scores, reference_recalls, reference_utility):
    scores = np.asarray(scores)
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise ValueError('Bad prediction array')
    groups = [('v1_five_scenarios', s) for s in [1, 2, 3, 4, 5]] + [('paired_news', s) for s in [1, 4]]
    thresholds = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    columns = []
    for group, scenario in groups:
        values = np.sort(scores[[r['evaluation_group'] == group and int(r['scenario']) == scenario for r in rows]])
        if not len(values):
            raise ValueError('Incomplete validation scenarios')
        below = np.searchsorted(values, thresholds, side='left') / len(values)
        columns.append(below if scenario == 4 else 1-below)
    recalls = np.stack(columns, axis=1)
    utility = .25*recalls[:, 3] + .0625*recalls[:, [0, 1, 2, 4]].sum(1) + .25*recalls[:, [5, 6]].sum(1)
    floors = np.array([reference_recalls[f'{g}/s{s}']-.02 for g, s in groups[:5]])
    eligible = (utility >= reference_utility-.005-1e-12) & (recalls[:, :5] >= floors-1e-12).all(1)
    ids = np.flatnonzero(eligible)
    passed = bool(len(ids))
    if not passed:
        ids = np.arange(len(thresholds))
    minimum = recalls.min(1)
    best = max(ids, key=lambda i: (round(float(minimum[i]), 12), round(float(utility[i]), 12), -abs(thresholds[i]-.5)))
    return dict(threshold=float(thresholds[best]), eligible=passed, minimum_recall=float(minimum[best]),
                utility=float(utility[best]), recalls={f'{g}/s{s}': float(v) for (g, s), v in zip(groups, recalls[best])})


def joint_objective(theta, x, y, weights, pairs, inverse_c, rank_weight):
    """Weighted multiclass CE plus OOC-vs-genuine ranking, with analytic gradient."""
    n, d = x.shape
    w, b = theta[:3*d].reshape(3, d), theta[3*d:]
    logits = x @ w.T + b
    probs = softmax(logits, axis=1)
    weight_sum = weights.sum()
    loss = np.dot(weights, logsumexp(logits, axis=1)-logits[np.arange(n), y])/weight_sum
    grad_logits = probs.copy()
    grad_logits[np.arange(n), y] -= 1
    grad_logits *= weights[:, None]/weight_sum
    if rank_weight:
        real, ooc = pairs.T
        margin = logits[:, 1]-logits[:, 0]
        difference = margin[real]-margin[ooc]
        loss += rank_weight*np.logaddexp(0, difference).mean()
        derivative = rank_weight*expit(difference)/len(pairs)
        np.add.at(grad_logits[:, 1], real, derivative)
        np.add.at(grad_logits[:, 0], real, -derivative)
        np.add.at(grad_logits[:, 1], ooc, -derivative)
        np.add.at(grad_logits[:, 0], ooc, derivative)
    reg = inverse_c/weight_sum
    loss += .5*reg*np.sum(w*w)
    grad_w = grad_logits.T @ x + reg*w
    grad_b = grad_logits.sum(0)
    return loss, np.r_[grad_w.ravel(), grad_b]


class ResidualHead(ClassifierMixin, BaseEstimator):
    """A compact frozen-feature residual head, serialized as NumPy parameters."""
    def __init__(self, parameters):
        self.parameters = parameters
        self.classes_ = np.arange(3)

    def fit(self, x, y=None):
        raise RuntimeError('Train this head through the recorded PyTorch experiment')

    def __sklearn_is_fitted__(self):
        return True

    def decision_function(self, x):
        p = self.parameters
        z = np.asarray(x, dtype=np.float32)
        hidden = np.maximum(z @ p['hidden.weight'].T+p['hidden.bias'], 0)
        return z @ p['skip.weight'].T+p['skip.bias'] + hidden @ p['output.weight'].T+p['output.bias']

    def predict_proba(self, x):
        return softmax(self.decision_function(x), axis=1)


class Experiment:
    def __init__(self, name):
        self.rows, self.x, self.clip, self.reference = load_development()
        self.train = np.flatnonzero([r['split'] == 'train' for r in self.rows])
        self.val = np.flatnonzero([r['split'] == 'val' for r in self.rows])
        self.train_rows = [self.rows[i] for i in self.train]
        self.val_rows = [self.rows[i] for i in self.val]
        self.y = np.array([0 if int(r['scenario']) == 4 else 1 if int(r['scenario']) == 1 else 2 for r in self.rows])
        counts = Counter(int(r['scenario']) for r in self.train_rows)
        mass = {4:.5, 1:.25, 2:1/12, 3:1/12, 5:1/12}
        self.weights = np.array([len(self.train)*mass[int(r['scenario'])]/counts[int(r['scenario'])] for r in self.train_rows])
        self.pairs = pair_indices(self.train_rows)
        self.out = OUT/name
        if self.out.exists():
            raise FileExistsError(self.out)
        self.out.mkdir(parents=True)
        self.diagnostic = json.loads((OUT/'incumbent_diagnostic.json').read_text())
        ref_scores = 1-self.reference['model'].predict_proba(self.x[self.val])[:, 0]
        self.reference_utility = metrics(self.val_rows, ref_scores, self.reference['threshold'])[0]['selection_utility']
        self.results = []
        (self.out/'manifest.json').write_text(json.dumps(dict(csv_sha256=digest(CSV),
            source_sha256=digest(__file__), plan_sha256=digest(OUT/'PLAN.md'), test_used=False,
            criterion='Protected seven-group minimum recall; details in PLAN.md',
            reference_utility=self.reference_utility), indent=2))

    def save(self, name, model, note=None):
        scores = 1-model.predict_proba(self.x[self.val])[:, 0]
        point = protected_point(self.val_rows, scores, self.diagnostic['recalls'], self.reference_utility)
        report, predictions = metrics(self.val_rows, scores, point['threshold'])
        result = dict(name=name, **point, metrics=report, note=note)
        self.results.append(result)
        joblib.dump(dict(model=model, task='three_class', kind='blip_v1_clip_large', threshold=point['threshold']), self.out/f'{name}.joblib')
        with (self.out/f'{name}_val.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
        (self.out/'results.json').write_text(json.dumps(self.results, indent=2))
        print(f'{name}: eligible={point["eligible"]} minimum={point["minimum_recall"]:.4f} utility={point["utility"]:.4f} '
              f'paired real/OOC={point["recalls"]["paired_news/s4"]:.4f}/{point["recalls"]["paired_news/s1"]:.4f}', flush=True)


def run_linear(exp):
    scaler, reference_head = exp.reference['model'].steps[0][1], exp.reference['model'].steps[-1][1]
    z = scaler.transform(exp.x).astype(np.float64)
    # Infer offsets from training partners only. At inference, one caption suffices.
    logits = reference_head.decision_function(z[exp.train])
    fake_margin = logsumexp(logits[:, 1:], axis=1)-logits[:, 0]
    real, ooc = exp.pairs.T
    midpoints = .5*(fake_margin[real]+fake_margin[ooc])
    text_slice = slice(2306, 3074)
    for alpha in [10., 100., 1000.]:
        ridge = Ridge(alpha=alpha).fit(z[exp.train][real, text_slice], midpoints)
        for strength in [.5, 1.]:
            head = copy.deepcopy(reference_head)
            head.coef_[1:, text_slice] -= strength*ridge.coef_[None, :]
            head.intercept_[1:] -= strength*ridge.intercept_
            exp.save(f'midpoint_a{alpha:g}_strength{strength:g}', make_pipeline(scaler, head))
    initial = np.r_[reference_head.coef_.ravel(), reference_head.intercept_]
    for c in [.001, .01]:
        for rank in [.25, 1.]:
            result = minimize(joint_objective, initial.copy(), jac=True, method='L-BFGS-B',
                args=(z[exp.train], exp.y[exp.train], exp.weights, exp.pairs, 1/c, rank),
                options=dict(maxiter=500, ftol=1e-10, gtol=1e-6))
            if not result.success:
                raise RuntimeError(f'Joint head did not converge: {result.message}')
            head = copy.deepcopy(reference_head)
            head.coef_ = result.x[:-3].reshape(3, z.shape[1]); head.intercept_ = result.x[-3:]
            exp.save(f'joint_c{c:g}_rank{rank:g}', make_pipeline(scaler, head),
                     dict(iterations=int(result.nit), objective=float(result.fun), optimizer=str(result.message)))


def run_mlp(exp):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    scaler, head = exp.reference['model'].steps[0][1], exp.reference['model'].steps[-1][1]
    z = torch.tensor(scaler.transform(exp.x[exp.train]), dtype=torch.float32)
    y = torch.tensor(exp.y[exp.train], dtype=torch.long)
    weights = torch.tensor(exp.weights, dtype=torch.float32)
    pairs = torch.tensor(exp.pairs)

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.skip = nn.Linear(z.shape[1], 3)
            self.hidden = nn.Linear(z.shape[1], 128)
            self.output = nn.Linear(128, 3)
            with torch.no_grad():
                self.skip.weight.copy_(torch.tensor(head.coef_)); self.skip.bias.copy_(torch.tensor(head.intercept_))
                self.output.weight.zero_(); self.output.bias.zero_()
            self.dropout = nn.Dropout(.3)

        def forward(self, values):
            return self.skip(values)+self.output(self.dropout(torch.relu(self.hidden(values))))

    for decay in [.01, .1]:
        for rank_weight in [0., .5]:
            torch.manual_seed(20260914)
            net = Network()
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=decay)
            for epoch in range(1, 31):
                net.train()
                indices = torch.randperm(len(z))
                losses = []
                for batch in indices.split(512):
                    optimizer.zero_grad()
                    logits = net(z[batch])
                    loss = (nn.functional.cross_entropy(logits, y[batch], reduction='none')*weights[batch]).mean()
                    if rank_weight:
                        selected = pairs[torch.randint(len(pairs), (128,))]
                        pair_logits = net(z[selected.flatten()]).reshape(-1, 2, 3)
                        margin = pair_logits[:, :, 1]-pair_logits[:, :, 0]
                        loss = loss + rank_weight*nn.functional.softplus(margin[:, 0]-margin[:, 1]).mean()
                    loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
                    losses.append(float(loss.detach()))
                if epoch in [5, 15, 30]:
                    net.eval()
                    parameters = {key: value.detach().numpy().copy() for key, value in net.state_dict().items()}
                    model = make_pipeline(scaler, ResidualHead(parameters))
                    exp.save(f'mlp_wd{decay:g}_rank{rank_weight:g}_epoch{epoch}', model,
                             dict(training_loss=float(np.mean(losses)), seed=20260914, hidden=128, dropout=.3))
                elif epoch % 5 == 0:
                    print(f'MLP wd={decay} rank={rank_weight} epoch={epoch} train_loss={np.mean(losses):.4f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['linear', 'mlp'], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        experiment = Experiment(args.arm)
        (run_linear if args.arm == 'linear' else run_mlp)(experiment)


if __name__ == '__main__':
    # Stable module identity for joblib artifacts.
    from .semantic_resolution import main as run
    run()
