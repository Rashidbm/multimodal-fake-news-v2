"""Write the measured results and the exact image-only training/inference explanation."""
import json
from datetime import datetime, timezone
from pathlib import Path

from .development import OUT, ROOT
from .preprocess import sha256


def pct(value):return f'{100*value:.2f}%'


def main():
    lock=json.loads((OUT/'selection_lock.json').read_text())
    test=json.loads((OUT/'final_test/results.json').read_text())
    verified=json.loads((OUT/'release_verification.json').read_text())
    if test['selection_lock_sha256']!=sha256(OUT/'selection_lock.json'):raise ValueError('Selection lock changed')
    bundles=json.loads((OUT/'release_bundles.json').read_text())
    results=test['results'];lines=['# Image branch: measured results and explanation','',
        'The models, operating thresholds and recommendations below were selected on validation data before the final test images were scored. The selected semantic branch was preserved. This report does not claim an absolute performance ceiling or a completed three-branch fusion model.','',
        '## Selected models on clean final tests','',
        '| Purpose and dataset | Model selected before testing | Authentic recall | Fake-image recall | Balanced accuracy | AP |',
        '|---|---|---:|---:|---:|---:|']
    for purpose,dataset in [('ai','genimage'),('news','news'),('broad','genimage'),('broad','news'),('broad','dgm4')]:
        name=lock['selected'][purpose];m=results[name][dataset+'/clean'][purpose]['overall']
        lines.append(f'| {purpose} / {dataset} | {name} | {pct(m["real_recall"])} | {pct(m["fake_recall"])} | {pct(m["balanced_accuracy"])} | {pct(m["AP"])} |')
    lines+=['','Authentic recall answers “of the authentic images, how many did we keep authentic?” Fake-image recall answers “of the manipulated/generated images, how many did we catch?” Balanced accuracy averages those two recalls, so predicting fake for everything scores 50%. AP and AUC measure ranking and should not be read as accuracy.','',
            '## Specified baselines at the required 0.5 threshold','',
            '| Model, clean GenImage test | Authentic recall | Generated recall | Accuracy | AP | AUC |',
            '|---|---:|---:|---:|---:|---:|']
    for name in ('rgb_guide','dct_guide','rgb_robust'):
        m=results[name]['genimage/clean']['at_half']['overall']
        lines.append(f'| {name} | {pct(m["real_recall"])} | {pct(m["fake_recall"])} | {pct(m["accuracy"])} | {pct(m["AP"])} | {pct(m["AUC"])} |')
    lines+=['','Checkpoint selection record:']
    for name in ('rgb_guide','dct_guide','rgb_robust','rgb_broad'):
        history=json.loads((OUT/name/'history.json').read_text());best=max(history,key=lambda r:r['metrics']['selection_metric_value'])
        lines.append(f'- `{name}`: completed {len(history)} epochs; retained epoch {best["epoch"]}, selected by {best["metrics"]["selection_metric_name"]}.')
    lines+=['','These are results on our fixed 44,000-image GenImage subset, not the full official benchmark. Generator-specific metrics, generator mean/std AP and GAN/diffusion averages are in the JSON results. The two exact-guide model runs are retained even when another candidate performs better.','',
            '## Robustness of the selected broad model','',
            '| Test condition | Authentic recall | Manipulation recall | Balanced accuracy |',
            '|---|---:|---:|---:|']
    name=lock['selected']['broad']
    for case in lock['final_test_cases']:
        m=results[name][case]['broad']['overall']
        lines.append(f'| {case} | {pct(m["real_recall"])} | {pct(m["fake_recall"])} | {pct(m["balanced_accuracy"])} |')
    lines+=['','The broad model uses one global threshold across these conditions; it does not receive the dataset name, manipulation method or test condition as an input. JPEG75 and blur1 are controlled stress tests, not a guarantee against every compression or editing process.','',
            '## Correct labels and actual model inputs','',
            '| Scenario | Image branch target | Reason |',
            '|---|---:|---|',
            '| S1: authentic image with out-of-context text | 0 | The image itself remains authentic. |',
            '| S2: fake text with authentic image | 0 | The caption being fake does not alter the pixels. |',
            '| S3: authentic text with manipulated/generated image | 1 | The image is fake. |',
            '| S4: genuine pair | 0 | The image is authentic. |',
            '| S5: fake text with manipulated/generated image | 1 | The image is fake. |','',
            'Each data-loader item contains (image tensor, target, row index). Only the image tensor enters the encoder. The target is compared with the output afterward in the loss; the index associates predictions with evaluation rows. No caption tokens, scenario, folder name, filename, face box or real/fake ground-truth flag enters the network. A behavioral check verifies that changing a row’s caption, label and scenario leaves its image input identical.','',
            'Using the pair-level fake label as the image label would incorrectly mark 6,589 authentic training-pair images as fake in the current shared package. That is an audit of the label semantics, not proof that every historical image model made that exact mistake.','',
            '## What happens during training','',
            '1. Decode the image and prepare the encoder-specific pixel tensor. Training augmentation modifies this image view while retaining its correct image label.',
            '2. Run the encoder forward. A ResNet returns a 2,048-number image representation; CLIP-L returns a normalized 768-number image representation.',
            '3. A learned classifier produces one logit z. Its sigmoid p = 1 / (1 + exp(-z)) is the model’s fake-image score. It is not a factual certainty or an externally calibrated truth probability.',
            '4. For the CNNs, binary cross-entropy compares z against y=0/1. Backpropagation changes only the parameters enabled for that training phase. The CLIP linear probes fit their scaler and classifier on cached training-image features; their CLIP encoder stays frozen.',
            '5. Validation selects a checkpoint, regularization and/or operating threshold. It never supplies gradients. Final-test images are evaluated only after the selection lock is written.','',
            'For example, suppose the image is manipulated (y=1) and the current model predicts p=0.2. Its unweighted BCE loss is −log(0.2)=1.609 and the logit gradient is p−y=−0.8, pushing the score upward during learning. For the same pixels labeled y=0, the forward score is still 0.2, but the loss becomes −log(0.8)=0.223 and the gradient is +0.2. The label changes the learning signal after prediction; it is not an input that tells the model the answer.','',
            'GenImage contains 32,000 train / 4,000 validation / 8,000 test images, balanced by class within all eight generators. The original GenImage train split supplies our training and internal validation; its original validation split supplies our final test. All eight generators are seen during training, so this is not an unseen-generator experiment.','',
            'The corrected news image dataset contains 17,620 train / 2,342 validation / 1,201 test unique images. The clean DGM4 subset contains 5,995 / 998 / 999. News training has 15,313 authentic versus 2,307 fake images; the image task is therefore not balanced merely because the five pair scenarios were balanced in an earlier dataset. Class-weighted losses address this without pretending repeated augmented views are independent new images.','',
            '## Architectures and trainable parts','',
            '**RGB guide:** 224×224 RGB pixels → ImageNet normalization → author ResNet-50 → global average pool → 2,048 features → learned linear layer → one logit. During training only, the unchanged author Fourier-mask transform is applied with probability 0.5, ratio 0.15, all frequency bands and channels. conv1, bn1, layer1 and layer2 are frozen; layer3, layer4 and the new head are trained. Frozen blocks also retain their BatchNorm running statistics. AdamW starts at 1e-4, weight decay 1e-4, batch size 64, maximum 30 epochs, validation-AP early stopping after five non-improvements.','',
            '**DCT guide:** resize to 224×224 → YCbCr luminance Y → separate orthonormal 8×8-block and 16×16-block DCTs → log(abs(coefficient)+1e-8) → average the two maps → normalize with one scalar mean/std fitted on training pixels → 1×224×224 tensor → ResNet-50 with a new one-channel first convolution → 2,048 features → one logit. There is no per-image min–max normalization. Layers 1/2 are frozen for five epochs; from epoch six all layers are trainable, AdamW resets to 1e-5 and gradient clipping uses norm 1. Missing DCT files cause an error; they never silently become a zero image.','',
            '**CLIP-L/14 image encoder:** official bicubic resize and center crop to 224×224 → 256 patches of 14×14 pixels plus one class token → 24 vision-transformer blocks of width 1,024 → class-token projection to 768 → L2 normalization. We use only its image encoder: no CLIP text encoder, BERT, BLIP or FND text input is used in this branch. The vision weights remain frozen for these probes.','',
            '**Extensions:** RGB robust adds training JPEG recompression. RGB broad starts from that detector and adapts to GenImage, corrected news and DGM4, with equal loss mass per domain/class, JPEG augmentation and occasional blur. It trains the later layers for three epochs and all layers from epoch four; checkpoint selection uses the worst validation-domain AUC. Frozen-feature candidates concatenate CLIP (768) and RGB (2,048), optionally DCT (2,048), then fit a weighted standardized logistic classifier. The final broad feature candidate uses CLIP plus the adapted RGB encoder.','',
            'For a board drawing, use these ResNet tensor shapes (B is batch size): input B×3×224×224 for RGB or B×1×224×224 for DCT → conv1 B×64×112×112 → max pool B×64×56×56 → layer1 B×256×56×56 → layer2 B×512×28×28 → layer3 B×1,024×14×14 → layer4 B×2,048×7×7 → average pool B×2,048 → linear B×1. The four residual stages contain 3, 4, 6 and 3 bottleneck blocks.','',
            '## The exported branch','']
    for purpose,path in bundles.items():
        config=json.loads(Path(path).read_text());candidate=lock['candidates'][lock['selected'][purpose]]
        lines+= [f'- **{purpose}:** `{lock["selected"][purpose]}`, encoders {", ".join(candidate["encoders"])}, raw dimension {config["raw_feature_dimension"]}, threshold {config["threshold"]:.9f}. [Bundle]({path}).']
    lines+=['','Every bundle returns a fake-image score and a 768-number feature vector for later fusion. A 768-dimensional CLIP representation passes through unchanged. Larger representations use a training-only projection: the classifier direction plus 767 residual PCA directions. The first coordinate preserves the detector logit exactly in real arithmetic; export checks numerical equivalence. This projection is an interface for the next integration step, not a trained or evaluated three-branch fusion model.','',
            'The release verifier recomputes pixel inference on stratified training images and checks that the exported probabilities and decisions reproduce the selected models. It also verifies the exported classifier head against all locked test features, without fitting a model or changing a threshold.','',
            'Run image inference from the repository with:', '', '```bash',
            '/Users/rashid/miniconda3/bin/python -m fnd.forensic.inference \\',
            f'  --bundle {bundles["broad"]} \\',
            '  --image /absolute/path/to/image.jpg --device mps', '```','',
            'Use `--device cpu` on a CPU-only machine or `--device cuda` with a compatible NVIDIA installation. The repository, the pinned author ResNet code in `external/FakeImageDetection`, and cached/downloaded CLIP weights are runtime dependencies; the CNN checkpoints and classifier head are included beside each bundle.','',
            '## What worked, what failed, and limits','',
            '- A news-only CLIP image classifier initially achieved 98.70% authentic and 98.79% fake recall on clean validation, but authentic recall fell to 75.97% under blur. Adding blurred training views raised blur authentic recall to 98.43%, with 97.57% fake recall. These are development results, not substitutes for the final-test tables above.',
            '- That news-only classifier detected only 4.4% of DGM4 face edits on the original 1,000-image DGM4 validation reservation. Adding DGM4 examples to the frozen CLIP head reached only 62.3% balanced accuracy there. This motivated the CNN adaptation; a high news score alone was insufficient.',
            '- A metadata-only diagnostic reached 99.19% news validation accuracy using image dimensions, file size and JPEG status. Those fields are forbidden as model inputs. This control shows strong source/preprocessing shortcuts in the dataset; it does not prove the release classifier uses exactly those shortcuts.',
            '- Official CLIP center cropping entirely removes the edited face in 23 of 499 manipulated DGM4 validation images, and leaves under half its area visible in 67. This explains part of the difficulty, not all of it. Whole-image RGB/DCT resizing keeps the frame, but small edits may still lose detail at 224×224.',
            '- DGM4 coverage here includes SimSwap, HFGI and StyleCLIP. InfoSwap images were unavailable locally. All selected metadata matches the author SHA-256 values; all 7,992 image sizes/CRCs match author ZIPs and 16 samples match byte-for-byte.',
            '- The GenImage subset was recovered from pinned original-format archives in a third-party mirror. Exact bytes, decoded pixels, original split provenance, source identity and conservative cross-split perceptual matches were audited. This is a reproducible subset experiment, not a full benchmark reproduction.',
            '- A single training seed, the chosen generators/edit methods and controlled corruptions do not cover future generators, arbitrary Photoshop edits, all sources or all real-world distribution shifts. Pretraining overlap of third-party foundation weights was not exhaustively audited.',
            '- GenImage nature labels are not a guarantee of untouched camera pixels. A manually inspected, confidently misclassified nature validation image is visibly a photographic collage. Benchmark labels were preserved; no validation examples were removed or relabeled because a model got them wrong.',
            '- The project news test split has research history from the semantic branch. It was not used for fitting these image checkpoints or selecting their operating points, but it is not a wholly new project-wide benchmark.',
            '- Changes in labels, training data, architecture and augmentation all contribute; there is no controlled evidence that mapping correction alone caused every improvement. Other branches cannot be assumed to repair these errors without an actual fusion experiment.','',
            '## Evidence and reproduction','',
            f'- [Locked validation choices]({OUT / "selection_lock.json"})',
            f'- [Final results, confusion counts, per-generator/source/edit metrics and recall intervals]({OUT / "final_test/results.json"})',
            f'- [Release verification]({OUT / "release_verification.json"})',
            f'- [Dependency and input provenance]({OUT / "environment.json"})',
            f'- [Development findings and controls]({ROOT / "docs/plans/image_branch_findings_2026_09_14.md"})',
            f'- [Reproduction and handoff instructions]({ROOT / "docs/IMAGE_BRANCH_REPRODUCTION.md"})',
            '- [Specific forensic-image implementation guide](</Users/rashid/Downloads/Forensic_Image_Detector_En 3.pdf>)',
            '- [GenImage author project](https://github.com/GenImage-Dataset/GenImage)',
            '- [Frequency masking author implementation](https://github.com/chandlerbing65nm/FakeImageDetection)',
            '- [Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets](https://arxiv.org/abs/2403.17608)',
            '- [Towards Universal Fake Image Detectors](https://arxiv.org/abs/2302.10174)',
            '- [DGM4 author project](https://github.com/rshaojimmy/MultiModal-DeepFake)','']
    path=ROOT/'docs/IMAGE_BRANCH_RESULTS_2026_09_14.md';path.write_text('\n'.join(lines))
    (OUT/'TASK_COMPLETE.json').write_text(json.dumps(dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),report=str(path),report_sha256=sha256(path),selected=lock['selected'],
        selection_lock_sha256=sha256(OUT/'selection_lock.json'),test_results_sha256=sha256(OUT/'final_test/results.json'),
        verified_release_sha256=sha256(OUT/'release_verification.json')),indent=2))
    print(path,flush=True)


if __name__=='__main__':main()
