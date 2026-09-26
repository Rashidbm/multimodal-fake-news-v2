"""Lock one development-selected candidate before any new test inference."""
import json
from datetime import datetime, timezone
from pathlib import Path

from fnd.train_clip_matcher import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/semantic_resolution'


def main():
    results = []
    for family, expected in [('linear',10),('mlp',12),('retrieval',3),('data_volume',8)]:
        source = OUT/family/'results.json'
        records = json.loads(source.read_text())
        assert len(records) == expected
        results.extend(dict(r, family=family, artifact=str(OUT/family/(r['name']+'.joblib'))) for r in records)
    incumbent = json.loads((OUT/'incumbent_diagnostic.json').read_text())
    minimum = min(incumbent['recalls'].values())
    eligible = [r for r in results if r['eligible']]
    best = max(eligible, key=lambda r:(round(r['minimum_recall'],12), round(r['utility'],12)))
    assert best['minimum_recall'] >= minimum+.01
    assert best['family'] == 'data_volume', 'Deployment adapter required for another family'
    verification = json.loads((OUT/'data_verification.json').read_text())
    assert len(verification) == 2
    destination = OUT/'candidate'
    destination.mkdir()
    base_path = ROOT/'outputs/recall_improvement/selected/bundle.json'
    bundle = json.loads(base_path.read_text())
    bundle.pop('semantic_projection',None)
    bundle.update(name='Additional unique-pair candidate: '+best['name'],
        classifier=dict(path=best['artifact'],sha256=digest(best['artifact'])), threshold=best['threshold'],
        training_csv=dict(path=str(ROOT/'data/processed/semantic_resolution_train_dev.csv'),
                          sha256=digest(ROOT/'data/processed/semantic_resolution_train_dev.csv')))
    bundle_path = destination/'bundle.json'
    bundle_path.write_text(json.dumps(bundle,indent=2))
    lock = dict(locked_utc=datetime.now(timezone.utc).isoformat(), test_used_for_selection=False,
        candidates_compared=len(results), selected=best, baseline_bundle=str(base_path),
        baseline_bundle_sha256=digest(base_path), candidate_bundle=str(bundle_path), candidate_bundle_sha256=digest(bundle_path),
        confirmation_csv=str(ROOT/'data/processed/semantic_resolution_confirmation.csv'),
        confirmation_csv_sha256=digest(ROOT/'data/processed/semantic_resolution_confirmation.csv'),
        original_evaluation_csv=str(ROOT/'data/processed/recall_evaluation.csv'),
        original_evaluation_csv_sha256=digest(ROOT/'data/processed/recall_evaluation.csv'),
        plan_sha256=digest(OUT/'PLAN.md'), data_verification_sha256=digest(OUT/'data_verification.json'),
        promotion_rule=dict(fresh_balanced_gain_ci95_lower_strictly_above=0,
            maximum_fresh_each_class_drop=.01, maximum_original_scenario_drop=.02,
            bootstrap_repetitions=5000, bootstrap_seed=2026091403))
    (destination/'selection_lock.json').write_text(json.dumps(lock,indent=2))
    (OUT/'all_development_results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:best[k] for k in ['name','family','threshold','minimum_recall','utility','recalls']},indent=2))


if __name__ == '__main__':
    main()
