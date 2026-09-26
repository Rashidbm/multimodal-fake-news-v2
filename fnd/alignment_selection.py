"""Compare genuine recall under fixed validation fake-scenario recall constraints."""
import argparse
import csv
import json
import math
from pathlib import Path

from .evaluate import score_rows


def constrained_score(rows, minimum_recall):
    ceilings=[]
    for scenario, minimum in minimum_recall.items():
        probabilities=sorted(float(r['prob']) for r in rows if int(r['scenario'])==int(scenario))
        if not probabilities or not 0 < minimum <= 1:
            raise ValueError('each constrained scenario needs examples and a recall target in (0,1]')
        required=math.ceil(len(probabilities)*minimum-1e-10)
        ceilings.append(probabilities[len(probabilities)-required])
    threshold=min(ceilings)
    result=score_rows(rows,threshold)
    for scenario, minimum in minimum_recall.items():
        assert result['per_scenario'][int(scenario)]['accuracy']+1e-12>=minimum
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',default='outputs/v1_alignment_round/selection_plan.json')
    ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args(argv)
    minimum=json.loads(Path(args.plan).read_text())['min_recall']
    candidates=[]
    membership=None
    for run in args.runs:
        paths=sorted(Path(run).glob('val_predictions*.csv'))
        for path in paths:
            with open(path,newline='') as f:
                rows=list(csv.DictReader(f))
            identifiers={r['sample_id']:(int(r['label_binary']),int(r['scenario'])) for r in rows}
            if len(identifiers)!=len(rows):
                raise ValueError('duplicate validation prediction IDs')
            if membership is None:
                membership=identifiers
            elif identifiers!=membership:
                raise ValueError('validation membership/labels differ')
            epoch=int(path.stem.rsplit('_',1)[-1]) if '_epoch_' in path.stem else None
            candidates.append({'run':str(run),'epoch':epoch,'validation_predictions':str(path),
                               'checkpoint':str(Path(run)/f'epoch_{epoch}.pt') if epoch else None,
                               'fixed':score_rows(rows),'constrained':constrained_score(rows,minimum)})
    if not candidates:
        raise ValueError('no validation predictions found')
    candidates.sort(key=lambda c:(c['constrained']['real_recall'],c['constrained']['f1_macro'],
                                  c['constrained']['ooc_genuine_auc']),reverse=True)
    Path(args.out).write_text(json.dumps({'minimum_recall':minimum,'selected':candidates[0],
                                         'candidates':candidates},indent=2))
    for c in candidates[:8]:
        m=c['constrained']
        print(c['run'],c['epoch'],f"real={m['real_recall']:.3f} OOC={m['per_scenario'][1]['accuracy']:.3f} macro={m['f1_macro']:.3f} threshold={m['threshold']:.6f}")


if __name__=='__main__':
    main()
