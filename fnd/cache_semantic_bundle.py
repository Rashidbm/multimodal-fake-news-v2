"""Export the selected V1 bundle's 768 features in the existing V2 cache format."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import torch
from PIL import Image

from .predict_semantic_fusion import SemanticFusionPredictor
from .train_clip_matcher import digest, read_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args(argv)
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    rows = read_rows(args.csv)
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Expected unique nonempty dataset')
    model = SemanticFusionPredictor(args.bundle, args.device)
    features = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start+args.batch_size]
        images = []
        for row in batch:
            raw = Path(row['image_path']).read_bytes()
            if row.get('image_sha1') and hashlib.sha1(raw).hexdigest() != row['image_sha1']:
                raise ValueError('Image bytes changed since dataset construction')
            with Image.open(io.BytesIO(raw)) as image:
                images.append(image.convert('RGB'))
        result = model.predict(images, [row['text'] for row in batch], return_semantic=True)
        features.append(result['semantic'])
        if start % (args.batch_size*25) == 0:
            print(f'Fused semantic features {min(start+len(batch), len(rows))}/{len(rows)}', flush=True)
    matrix = torch.cat(features)
    if matrix.shape != (len(rows), 768) or not torch.isfinite(matrix).all():
        raise ValueError('Invalid semantic feature shape/values')
    payload = dict(sample_ids=[row['sample_id'] for row in rows], features=matrix,
                   csv_sha256=digest(args.csv), checkpoint=str(Path(args.bundle).resolve()),
                   checkpoint_sha256=digest(args.bundle),
                   feature_kind=model.fusion['kind'],
                   definition='768 selected image/caption bundle features: retained classifier directions plus training-only orthogonal PCA; new downstream heads must be trained for this representation')
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    destination.with_suffix('.json').write_text(json.dumps({k: v for k, v in payload.items() if k not in ['sample_ids', 'features']}, indent=2))
    print(f'COMPLETE {destination}', flush=True)


if __name__ == '__main__':
    main()
