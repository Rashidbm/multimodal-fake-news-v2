"""Use a locked binary V1 fusion bundle on an image and its caption."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import BlipForImageTextRetrieval, BlipProcessor, CLIPModel, CLIPTokenizerFast

from .data.torch_dataset import IMAGENET_MEAN, IMAGENET_STD, clip_image_transform
from .fit_semantic_fusion import feature_matrix, probability
from .models.fnd_clip import FNDCLIP, FNDCLIPConfig
from .train import load_tokenizers
from .train_clip_matcher import digest
from .cache_clip_large import embedding
from .cache_qwen_embedding import PROVENANCE_KEYS as QWEN_PROVENANCE_KEYS, encode as encode_qwen, load_embedder as load_qwen_embedder


class SemanticFusionPredictor:
    """Inference receives only PIL images and captions. Metadata/targets are absent."""
    def __init__(self, bundle_path, device='mps'):
        path = Path(bundle_path).resolve()
        self.bundle = json.loads(path.read_text())
        self.device = device

        def artifact(key):
            spec = self.bundle[key]
            location = Path(spec['path'])
            if not location.is_absolute():
                location = path.parent/location
            if digest(location) != spec['sha256']:
                raise ValueError(f'Changed artifact: {key}')
            return location

        self.fusion = joblib.load(artifact('classifier'))
        self.threshold = float(self.bundle['threshold'])
        if self.threshold != float(self.fusion['threshold']):
            raise ValueError('Bundle and classifier thresholds disagree')
        self.semantic_projection = (dict(np.load(artifact('semantic_projection')))
                                    if 'semantic_projection' in self.bundle else None)
        uses_qwen = self.fusion['kind'] == 'blip_v1_qwen_embedding'
        if uses_qwen != ('qwen_embedding' in self.bundle) or (uses_qwen and 'clip_large' in self.bundle):
            raise ValueError('Classifier feature kind and bundle encoders disagree (CLIP-L vs Qwen)')
        self.qwen = None
        if uses_qwen:
            source = self.bundle['qwen_embedding']
            self.qwen = load_qwen_embedder(min_pixels=source['min_pixels'], max_pixels=source['max_pixels'], device=device)
            changed = [k for k in QWEN_PROVENANCE_KEYS if self.qwen.provenance[k] != source[k]]
            if changed:
                raise ValueError(f'Local Qwen model/settings differ from the bundle: {changed}')
        self.blip = None
        if self.fusion['kind'] not in ['v1_only', 'clip_large_only']:
            self.processor = BlipProcessor.from_pretrained(self.bundle['blip_model'], revision=self.bundle['blip_revision'])
            self.blip = BlipForImageTextRetrieval.from_pretrained(self.bundle['blip_model'], revision=self.bundle['blip_revision'])
            if 'blip_checkpoint' in self.bundle:
                checkpoint = torch.load(artifact('blip_checkpoint'), map_location='cpu', weights_only=False)
                self.blip.load_state_dict(checkpoint['model'])
            self.blip.to(device).eval().requires_grad_(False)
            self.blip_transform = transforms.Compose([
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(), transforms.Normalize(self.processor.image_processor.image_mean,
                                                           self.processor.image_processor.image_std)])
        self.clip_large = None
        if 'clip_large' in self.bundle:
            source = self.bundle['clip_large']
            self.clip_large = CLIPModel.from_pretrained(source['name'], revision=source['revision']).to(device).eval().requires_grad_(False)
            self.large_tokenizer = CLIPTokenizerFast.from_pretrained(source['name'], revision=source['revision'])
            self.large_transform = clip_image_transform('official')
        self.v1 = None
        if self.fusion['kind'] not in ['blip', 'blip_with_score', 'clip_large_only']:
            checkpoint = torch.load(artifact('v1_checkpoint'), map_location='cpu', weights_only=False)
            config = FNDCLIPConfig(**checkpoint['config'])
            self.v1 = FNDCLIP(config)
            self.v1.load_state_dict(checkpoint['model'])
            self.v1.to(device).eval().requires_grad_(False)
            self.bert_tokenizer, self.clip_tokenizer = load_tokenizers(config)
            self.resnet_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                       transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
            self.clip_transform = clip_image_transform(config.clip_preprocess)

    @torch.inference_mode()
    def predict(self, images, captions, return_semantic=False):
        if len(images) != len(captions) or not images:
            raise ValueError('Provide one caption for each image')
        images = [image.convert('RGB') for image in images]
        blip, large = None, None
        if self.blip is not None:
            text = self.processor.tokenizer(captions, max_length=77, padding='max_length',
                                            truncation=True, return_tensors='pt')
            output = self.blip(pixel_values=torch.stack([self.blip_transform(image) for image in images]).to(self.device),
                               input_ids=text['input_ids'].to(self.device), attention_mask=text['attention_mask'].to(self.device),
                               use_itm_head=True)
            blip = dict(hidden=output.question_embeds[:, 0].cpu(), logit=(output.itm_score[:, 0]-output.itm_score[:, 1]).cpu())
        if self.clip_large is not None:
            tokens = self.large_tokenizer(captions, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
            large = dict(image=embedding(self.clip_large.get_image_features(
                pixel_values=torch.stack([self.large_transform(im) for im in images]).to(self.device))).cpu(),
                text=embedding(self.clip_large.get_text_features(input_ids=tokens['input_ids'].to(self.device),
                    attention_mask=tokens['attention_mask'].to(self.device))).cpu())
        v1 = None
        if self.v1 is not None:
            bert = self.bert_tokenizer(captions, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
            clip = self.clip_tokenizer(captions, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
            result = self.v1(torch.stack([self.resnet_transform(image) for image in images]).to(self.device),
                bert['input_ids'].to(self.device), bert['attention_mask'].to(self.device),
                torch.stack([self.clip_transform(image) for image in images]).to(self.device),
                clip['input_ids'].to(self.device), clip['attention_mask'].to(self.device))
            v1 = dict(hidden=result['semantic'].cpu(), logit=result['logits'].squeeze(-1).cpu())
        qwen = dict(features=encode_qwen(self.qwen, images, captions)) if self.qwen is not None else None
        features = feature_matrix(blip, v1, self.fusion['kind'], large, qwen=qwen)
        values = probability(self.fusion['model'], features, self.fusion['task'])
        mismatch_values = blip['logit'].sigmoid().tolist() if blip is not None else [None]*len(images)
        predictions = [dict(fake_probability=float(value), prediction='fake' if value >= self.threshold else 'real',
                     threshold=self.threshold, mismatch_score=float(mismatch) if mismatch is not None else None)
                for value, mismatch in zip(values, mismatch_values)]
        if return_semantic:
            if self.semantic_projection is None:
                raise ValueError('Bundle has no trained semantic projection')
            z = self.fusion['model'][0].transform(features).astype(np.float64)
            semantic = torch.from_numpy(z@self.semantic_projection['basis'].T).float()
            return dict(predictions=predictions, semantic=semantic)
        return predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args(argv)
    predictor = SemanticFusionPredictor(args.bundle, args.device)
    with Image.open(args.image) as image:
        result = predictor.predict([image], [args.text])[0]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
