"""Smoke-check the semantic and image branches: each should emit a (1, 768) feature."""
import torch
from PIL import Image

from fnd.forensic.inference import ImageBranch
from fnd.predict_semantic_fusion import SemanticFusionPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image = Image.new('RGB', (256, 256), (120, 80, 200))
caption = 'A dummy caption for a smoke test.'

semantic = SemanticFusionPredictor('models/semantic/bundle.json', device=device)
out = semantic.predict([image], [caption], return_semantic=True)
print('semantic', tuple(out['semantic'].shape), out['predictions'][0])

for name in ['news', 'broad', 'ai']:
    branch = ImageBranch(f'models/image/{name}/bundle.json', device=device)
    out = branch([image])
    print(f'image/{name}', tuple(out['features'].shape), f"p_fake={out['probability'].item():.4f}")
    del branch
    if device == 'cuda':
        torch.cuda.empty_cache()
