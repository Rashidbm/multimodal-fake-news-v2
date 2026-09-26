"""One locked confirmation comparison; no fitting or threshold search."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from .evaluate import score_rows
from .evaluate_alignment import paired_summary, paired_difference
from .fit_semantic_fusion import feature_matrix, load_cache
from .semantic_resolution import ROOT, OUT
from .train_clip_matcher import digest, read_rows


def main():
    lock_path = OUT/'candidate/selection_lock.json'
    lock = json.loads(lock_path.read_text())
    assert lock['test_used_for_selection'] is False
    for key in ['baseline_bundle','candidate_bundle','confirmation_csv','original_evaluation_csv']:
        assert digest(lock[key]) == lock[key+'_sha256']
    assert digest(OUT/'PLAN.md') == lock['plan_sha256']
    destination = OUT/'confirmation_results'
    destination.mkdir()
    models, bundles = {}, {}
    for name, key in [('incumbent','baseline_bundle'),('candidate','candidate_bundle')]:
        bundles[name] = json.loads(Path(lock[key]).read_text())
        spec = bundles[name]['classifier']
        assert digest(spec['path']) == spec['sha256']
        models[name] = joblib.load(spec['path'])
        assert models[name]['threshold'] == bundles[name]['threshold']
    reports, predictions = {}, {}
    with threadpool_limits(limits=2):
        for domain, csv_path, paths in [
            ('fresh',lock['confirmation_csv'],[OUT/f'confirmation_{k}.pt' for k in ['blip','v1','clip']]),
            ('original',lock['original_evaluation_csv'],[ROOT/f'outputs/recall_improvement/{name}.pt'
                for name in ['eval_blip','eval_old_v1','eval_clip_large']])]:
            all_rows = read_rows(csv_path)
            caches = [load_cache(path,csv_path,all_rows) for path in paths]
            for key, cache, development in zip(['blip','v1','clip'],caches,
                [OUT/f'expanded_{k}.pt' for k in ['blip','v1','clip']]):
                import torch
                dev = torch.load(development,map_location='cpu',weights_only=True)
                for field in ['model_name','model_revision','checkpoint_sha256','preprocessing']:
                    assert cache.get(field) == dev.get(field), (key,field)
            indices = np.array([i for i,r in enumerate(all_rows)
                if domain == 'fresh' or r['evaluation_group'] == 'v1_five_scenarios'])
            rows = [all_rows[i] for i in indices]
            x = feature_matrix(caches[0],caches[1],'blip_v1_clip_large',caches[2])[indices]
            reports[domain], predictions[domain] = {}, {}
            for name, saved in models.items():
                model, threshold = saved['model'],saved['threshold']
                probabilities = 1-model.predict_proba(x)[:,0]
                logits = model.decision_function(x)
                rank = logsumexp(logits[:,1:],axis=1)-logits[:,0]
                pred = [dict(sample_id=r['sample_id'],scenario=int(r['scenario']),label_binary=int(r['label_binary']),
                             prob=float(p),ranking_score=float(s)) for r,p,s in zip(rows,probabilities,rank)]
                predictions[domain][name] = pred
                report = dict(threshold=threshold,metrics=score_rows(pred,threshold))
                if domain == 'fresh':
                    report['paired'] = paired_summary(pred,rows,threshold)
                reports[domain][name] = report
                with (destination/f'{domain}_{name}_predictions.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(pred[0]));writer.writeheader();writer.writerows(pred)
            if domain == 'fresh':
                rule = lock['promotion_rule']
                comparison = paired_difference(predictions[domain]['incumbent'],predictions[domain]['candidate'],rows,
                    models['incumbent']['threshold'],models['candidate']['threshold'],
                    repetitions=rule['bootstrap_repetitions'],seed=rule['bootstrap_seed'])
            else:
                previous = {r['sample_id']:float(r['prob']) for r in read_rows(
                    ROOT/'outputs/recall_improvement/final_results/recall_extension_predictions.csv')}
                max_error = max(abs(r['prob']-previous[r['sample_id']]) for r in predictions[domain]['incumbent'])
                assert max_error < 1e-10
    rule = lock['promotion_rule']
    gains = comparison['metrics']
    original_gains = {str(s):reports['original']['candidate']['metrics']['per_scenario'][s]['accuracy']
                     - reports['original']['incumbent']['metrics']['per_scenario'][s]['accuracy'] for s in range(1,6)}
    gates = dict(fresh_balanced_gain_supported=gains['balanced_accuracy']['ci95'][0] > rule['fresh_balanced_gain_ci95_lower_strictly_above'],
        fresh_recalls_protected=all(gains[k]['difference'] >= -rule['maximum_fresh_each_class_drop']-1e-12
                                    for k in ['genuine_recall','ooc_recall']),
        original_scenarios_protected=min(original_gains.values()) >= -rule['maximum_original_scenario_drop']-1e-12)
    promote = all(gates.values())
    report = dict(evaluated_utc=datetime.now(timezone.utc).isoformat(),selection_lock_sha256=digest(lock_path),
        source_sha256=digest(__file__),results=reports,fresh_paired_comparison=comparison,
        original_scenario_gains=original_gains,incumbent_original_reproduction_max_error=max_error,
        promotion_gates=gates,promote_candidate=promote,
        recommended_bundle=lock['candidate_bundle' if promote else 'baseline_bundle'],
        milestone_reached=all(reports['fresh']['candidate']['paired'][k]>=.85 for k in ['genuine_recall','ooc_recall']),
        limitations=['Custom NewsCLIPpings repartition, not official benchmark performance',
                     'Fresh confirmation covers genuine/OOC pairs only; five-scenario test is reused',
                     'No event-disjointness or universal upper-bound claim'])
    (destination/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
