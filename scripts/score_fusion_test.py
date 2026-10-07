"""Score an already-trained fusion checkpoint on a test npz, once, without retraining.

    python scripts/score_fusion_test.py --checkpoint outputs/fusion_compare/qwen_L30_s42/best.pt \
        --test-features features/test_features.npz

Uses fnd.train_fusion's own load/predict/report, so the numbers match what train_fusion.py would
have printed with --test-features. Writes test_summary.json and test_predictions.csv next to the
checkpoint and refuses to overwrite them (test is scored once per checkpoint).
"""
import argparse
import json
from pathlib import Path

import torch

from fnd.models.pipeline_v3 import V3FusionModule
from fnd.train_fusion import CLASSES, digest, load, predict, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--test-features', required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out = Path(args.checkpoint).parent
    if (out/'test_summary.json').exists():
        raise SystemExit(f'{out/"test_summary.json"} exists: this checkpoint was already scored on test')

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = V3FusionModule().to(args.device)
    model.load_state_dict(checkpoint['model'])
    test = load(args.test_features, ['test'])
    probabilities = predict(model, test['x'], args.device)
    summary = dict(checkpoint=str(args.checkpoint), selected_epoch=checkpoint['epoch'], seed=checkpoint.get('seed'),
                   test=report(test['y'], probabilities), test_features=str(args.test_features),
                   test_features_sha256=digest(args.test_features))
    (out/'test_summary.json').write_text(json.dumps(summary, indent=2))
    with open(out/'test_predictions.csv', 'w', encoding='utf-8') as f:
        f.write('sample_id,true_class,predicted_class,' + ','.join(f'p_{c}' for c in CLASSES) + '\n')
        for sample, y, p in zip(test['ids'], test['y'], probabilities):
            f.write(f'{sample},{CLASSES[y]},{CLASSES[p.argmax()]},' + ','.join(f'{v:.6f}' for v in p) + '\n')
    t = summary['test']
    print(f"{out.name}: test acc {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}  "
          f"binary acc {t['binary']['accuracy']:.4f}")


if __name__ == '__main__':
    main()
