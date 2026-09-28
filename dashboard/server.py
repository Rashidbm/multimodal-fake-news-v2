"""MultiGuard dashboard backend: live semantic, image and text branches with a pluggable fusion model.

Run from the repository root:

    python -m dashboard.server
    python -m dashboard.server --fusion my_package.my_module:load_fusion

Every request runs the three trained branches on the uploaded image and caption:
  semantic  models/semantic/bundle.json   -> v_semantic [1, 768], pair fake score, BLIP mismatch
  image     models/image/<name>/bundle.json -> v_imgfor [1, 768], image fake probability
  text      Qwen3.5-9B layer 30, masked mean, max_length 64 -> v_textfor [1, 4096]
The text settings match the cached features the fusion is trained on (features/v_textfor.json).

The final five-class verdict comes only from a trained fusion model passed with --fusion.
Without one, the response carries the branch results and final_available=false; no
verdict is invented from the branch scores.

A fusion factory is any importable callable `factory(device) -> fusion` where `fusion` has
  fusion.classes  : list of class names, a permutation of CLASSES below
  fusion(semantic, image, text) -> probabilities tensor [1, len(classes)]
with semantic [1, 768], image [1, 768], text [1, 4096] float32 tensors already on `device`.
"""
import argparse
import importlib
import io
import os
import threading
import time
from pathlib import Path

# The dashboard only serves models that are already downloaded. Set before transformers is
# imported, so a terminal without HF_HOME fails at start-up instead of downloading into ~/.cache.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from fnd.forensic.inference import ImageBranch
from fnd.models.text_fluoroscopy import TextFluoroscopy, TextFluoroscopyConfig
from fnd.predict_semantic_fusion import SemanticFusionPredictor

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / 'static'

# Fusion class names (fnd.train_team_fusion order) and their dashboard display:
# (UI label_index, key in the page's probabilities object, verdict, Arabic verdict).
# The probability keys are the semester-1 page's; its "AI-Text" row is this fake/edited-text class.
CLASSES = ['genuine', 'out_of_context', 'real_text_fake_image',
           'fake_or_edited_text_real_image', 'fake_or_edited_text_fake_image']
DISPLAY = {
    'genuine': (0, 'Real', 'Real', 'حقيقي'),
    'out_of_context': (1, 'Out-of-Context', 'Out-of-Context', 'خارج السياق'),
    'real_text_fake_image': (2, 'Manipulated', 'Manipulated', 'معدَّل'),
    'fake_or_edited_text_real_image': (3, 'AI-Text', 'Fake/Edited Text', 'نص مزيَّف أو معدَّل'),
    'fake_or_edited_text_fake_image': (4, 'Fully-Fabricated', 'Fully Fabricated', 'ملفَّق بالكامل'),
}
EXPLANATIONS = {
    'genuine': (
        'The fused model rates this pair as genuine: it found no strong sign of out-of-context use, '
        'image manipulation, or fake or edited text.',
        'يقيّم النموذج المدمج هذا الزوج على أنه حقيقي: لم يرصد مؤشرات قوية على استخدام خارج السياق '
        'أو تعديل في الصورة أو نص مزيَّف أو معدَّل.'),
    'out_of_context': (
        'The text and image each look authentic, but the fused model judges that they do not belong '
        'together, so the image is likely used out of context.',
        'يبدو النص والصورة حقيقيين كلٌّ على حدة، لكن النموذج المدمج يرى أنهما لا ينتميان لبعضهما، '
        'لذا يُرجَّح أن الصورة مستخدمة خارج سياقها.'),
    'real_text_fake_image': (
        'The fused model points to a manipulated or AI-generated image paired with authentic text.',
        'يشير النموذج المدمج إلى صورة معدَّلة أو مولَّدة بالذكاء الاصطناعي مقترنة بنص حقيقي.'),
    'fake_or_edited_text_real_image': (
        'The fused model points to fake or edited text paired with an authentic image. This class '
        'covers rumours and edits, not only AI-written text.',
        'يشير النموذج المدمج إلى نص مزيَّف أو معدَّل مقترن بصورة حقيقية. تشمل هذه الفئة الشائعات '
        'والتعديلات، وليس النصوص المكتوبة بالذكاء الاصطناعي فقط.'),
    'fake_or_edited_text_fake_image': (
        'The fused model points to both fake or edited text and a manipulated or AI-generated image.',
        'يشير النموذج المدمج إلى نص مزيَّف أو معدَّل وصورة معدَّلة أو مولَّدة بالذكاء الاصطناعي معًا.'),
}
PENDING = (
    'Final prediction pending',
    'النتيجة النهائية قيد الانتظار',
    'The scores on the right come from the trained semantic, image and text branches. The final '
    'five-class verdict needs the fused model, which has not been trained yet.',
    'الدرجات المعروضة صادرة عن فروع التحليل الدلالي والصورة والنص المدرَّبة. أما النتيجة النهائية '
    'بفئاتها الخمس فتحتاج إلى النموذج المدمج، ولم يُدرَّب بعد.',
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    two_gpus = torch.cuda.device_count() > 1
    default_branch = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--semantic-bundle', default=str(ROOT/'models/semantic/bundle.json'))
    ap.add_argument('--image-bundle', default=str(ROOT/'models/image/news/bundle.json'),
                    help='news (edited or AI-generated), broad, or ai bundle')
    ap.add_argument('--text-model', default=os.environ.get('MULTIGUARD_TEXT_MODEL', r'D:\models\Qwen3.5-9B'))
    ap.add_argument('--text-layer', type=int, default=30)
    ap.add_argument('--text-max-len', type=int, default=64, help='must match the cached training features')
    ap.add_argument('--no-text', action='store_true', help='skip Qwen (the fusion then cannot run)')
    ap.add_argument('--branch-device', default=default_branch)
    ap.add_argument('--text-device', default='cuda:1' if two_gpus else default_branch)
    ap.add_argument('--fusion', default=None, help='module:factory returning a trained fusion model')
    return ap.parse_args(argv)


def load_fusion(spec, device):
    module_name, _, attr = spec.partition(':')
    if not attr:
        raise ValueError('--fusion must look like package.module:factory')
    fusion = getattr(importlib.import_module(module_name), attr)(device)
    if sorted(fusion.classes) != sorted(CLASSES):
        raise ValueError(f'fusion classes {fusion.classes} are not {CLASSES}')
    return fusion


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()   # one request at a time on the GPUs
        t = time.perf_counter()
        from huggingface_hub import constants
        print(f'model cache {constants.HF_HUB_CACHE} (offline={constants.HF_HUB_OFFLINE})', flush=True)
        print(f'semantic branch on {args.branch_device}', flush=True)
        self.semantic = SemanticFusionPredictor(args.semantic_bundle, device=args.branch_device)
        print(f'image branch {Path(args.image_bundle).parent.name} on {args.branch_device}', flush=True)
        self.image = ImageBranch(args.image_bundle, device=args.branch_device)
        self.text = None
        if not args.no_text:
            print(f'text branch {args.text_model} on {args.text_device}', flush=True)
            cfg = TextFluoroscopyConfig(model_name=args.text_model, layer=args.text_layer,
                                        max_len=args.text_max_len, dtype='auto')
            self.text = TextFluoroscopy(cfg, device=torch.device(args.text_device))
            print(f'  {self.text.describe()}', flush=True)
        self.fusion = None
        if args.fusion:
            if self.text is None:
                raise ValueError('--fusion needs the text branch; drop --no-text')
            self.fusion = load_fusion(args.fusion, args.branch_device)
            print(f'fusion {args.fusion}', flush=True)
        else:
            print('no fusion model: responses carry branch results only', flush=True)
        print(f'ready in {time.perf_counter()-t:.0f}s', flush=True)

    def analyze(self, text, image):
        timings = {}
        with self.lock, torch.inference_mode():
            t = time.perf_counter()
            sem = self.semantic.predict([image], [text], return_semantic=True)
            timings['semantic'] = time.perf_counter() - t

            t = time.perf_counter()
            img = self.image([image])
            timings['image'] = time.perf_counter() - t

            text_info, v_text = None, None
            if self.text is not None:
                t = time.perf_counter()
                v_text, lengths = self.text.encode_texts([text])
                timings['text'] = time.perf_counter() - t
                text_info = dict(tokens=lengths[0], truncated=lengths[0] > self.args.text_max_len,
                                 feature_dim=int(v_text.shape[-1]))

            probs = None
            if self.fusion is not None:
                t = time.perf_counter()
                device = self.args.branch_device
                out = self.fusion(sem['semantic'].to(device).float(), img['features'].to(device).float(),
                                  v_text.to(device).float())
                probs = dict(zip(self.fusion.classes, out[0].float().cpu().tolist()))
                timings['fusion'] = time.perf_counter() - t

        semantic = sem['predictions'][0]
        image_probability = float(img['probability'].item())
        image_threshold = float(self.image.config['threshold'])
        # Module names are the semester-1 page's. No model scores AI authorship or text
        # patterns (the text branch yields features only), so those two stay null. Until a
        # fusion exists, "overall" is the semantic branch's trained pair fake score.
        result = {
            'final_available': probs is not None,
            'modules': {
                'text_ai': None,
                'text_patterns': None,
                'image_manip': round(image_probability, 4),
                'cross_modal': None if semantic['mismatch_score'] is None else round(semantic['mismatch_score'], 4),
                'overall': round(semantic['fake_probability'] if probs is None else 1 - probs['genuine'], 4),
                'overall_source': 'semantic_branch' if probs is None else 'fusion',
            },
            'branches': {
                'semantic': dict(semantic, feature_dim=int(sem['semantic'].shape[-1])),
                'image': dict(probability=image_probability, threshold=image_threshold,
                              prediction='fake' if image_probability >= image_threshold else 'real',
                              target=self.image.config['target'], bundle=Path(self.args.image_bundle).parent.name,
                              feature_dim=int(img['features'].shape[-1])),
                'text': text_info,
            },
            'timings_ms': {k: round(v*1000) for k, v in timings.items()},
        }
        if probs is None:
            title, title_ar, explanation, explanation_ar = PENDING
            result.update(verdict=title, verdict_ar=title_ar, label_index=None, confidence=None,
                          probabilities=None, explanation=explanation, explanation_ar=explanation_ar)
        else:
            winner = max(probs, key=probs.get)
            index, _, name, name_ar = DISPLAY[winner]
            result.update(verdict=name, verdict_ar=name_ar, label_index=index,
                          confidence=round(probs[winner]*100, 1),
                          probabilities={DISPLAY[c][1]: round(p, 4) for c, p in probs.items()},
                          explanation=EXPLANATIONS[winner][0], explanation_ar=EXPLANATIONS[winner][1])
        return result


def create_app(pipeline):
    app = FastAPI(title='MultiGuard')

    @app.post('/api/analyze')
    def analyze(text: str = Form(...), image: UploadFile = File(...)):
        text = text.strip()
        if not text:
            return JSONResponse({'error': 'Please enter article text'}, status_code=400)
        try:
            with Image.open(io.BytesIO(image.file.read())) as raw:
                pil = raw.convert('RGB')
        except (UnidentifiedImageError, OSError):
            return JSONResponse({'error': 'The uploaded file is not a readable image'}, status_code=400)
        try:
            return JSONResponse(pipeline.analyze(text, pil))
        except Exception as e:   # surface the failure to the UI instead of a bare 500
            return JSONResponse({'error': f'{type(e).__name__}: {e}'}, status_code=500)

    @app.get('/api/health')
    def health():
        args = pipeline.args
        return {'status': 'ok', 'final_available': pipeline.fusion is not None, 'fusion': args.fusion,
                'image_bundle': Path(args.image_bundle).parent.name,
                'text_model': None if pipeline.text is None else args.text_model,
                'devices': {'branches': args.branch_device, 'text': args.text_device}}

    app.mount('/', StaticFiles(directory=str(STATIC), html=True), name='static')
    return app


def main(argv=None):
    import uvicorn
    args = parse_args(argv)
    try:
        pipeline = Pipeline(args)
    except OSError as e:
        from huggingface_hub import constants
        raise SystemExit(f'\nA model is missing from {constants.HF_HUB_CACHE}:\n  {str(e).splitlines()[0]}\n'
                         'The dashboard does not download models. Point HF_HOME at the folder that holds '
                         'them (e.g. $env:HF_HOME = "D:\\hf_cache") and start it again.') from None
    app = create_app(pipeline)
    print(f'Open http://{args.host}:{args.port}', flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
