r"""Runtime Image Branch v2 (frozen DINOv2-L + shared bottleneck): image -> v_imgfor [B, 768] and P(fake).

    python -m fnd.imagev2.infer --bundle D:\...\production\B1_dinov2_lam0_s13\bundle.json --image photo.jpg --features-out v.pt

The interface mirrors fnd.forensic.inference.ImageBranch (``config['threshold']``, ``config['target']`` and a result dict
with ``features`` [B,768] and ``probability`` [B]) so the dashboard/fusion code can consume it. The dashboard itself is
NOT switched over yet; `v_imgfor` lives in a different feature space than the old branch, so fusion must be retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .canonical import crop224, load_rgb
from .extract import DinoEncoders
from .model import ImageHead


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_bundle(run_dir, out_dir, model_id="facebook/dinov2-large"):
    """Write a self-contained runtime bundle (checkpoint + bundle.json) from a trained main-protocol run directory."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    saved = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    meta = json.loads((run_dir.parent.parent / "features" / "B1_dinov2" / "meta.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=saved["model"]), out_dir / "head.pt")
    bundle = dict(
        format_version=2, name=run_dir.name,
        backbone=dict(model_id=model_id, revision=meta["provenance"]["revision"], layers=meta["provenance"]["layers"], tokens="CLS + mean patch tokens per layer",
                      normalization="ImageNet mean/std", input="shorter side 224 (bicubic) + center crop 224, RGB, metadata ignored", frozen=True),
        head=dict(file="head.pt", sha256=sha256_file(out_dir / "head.pt"), in_dim=meta["dim"], hidden=cfg["hidden"], dropout=cfg["dropout"], out_dim=768),
        threshold=0.5, threshold_note="0.5: the balanced sampler makes the logit roughly calibrated; validation-tuned thresholds were unstable across seeds",
        target="0 authentic image; 1 AI-generated or manipulated image", feature_dimension=768, lambda_aux=cfg["lambda_aux"], seed=cfg["seed"],
        trained_on="Image Branch v2 train split (original project split=train only); no GenImage, DGM4 or external data",
        manifest_sha256=meta["manifest_sha256"],
        validation=dict(worst_domain_balanced_accuracy=metrics["selection"]["best_val_worst_domain_bacc"], best_epoch=metrics["selection"]["best_epoch"]))
    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    return bundle


class ImageBranchV2(nn.Module):
    def __init__(self, bundle, device="cuda:0"):
        super().__init__()
        path = Path(bundle)
        self.config = json.loads(path.read_text())
        self.device = device
        spec = self.config["backbone"]
        self.encoder = DinoEncoders(spec["model_id"], device, spec["layers"], spec["revision"])
        if self.encoder.revision and spec["revision"] and self.encoder.revision != spec["revision"]:
            raise ValueError("Local DINOv2 revision differs from the bundle")
        head_file = path.parent / self.config["head"]["file"]
        if sha256_file(head_file) != self.config["head"]["sha256"]:
            raise ValueError("Head checksum mismatch")
        h = self.config["head"]
        self.head = ImageHead(h["in_dim"], h["hidden"], h["out_dim"], h["dropout"])
        self.head.load_state_dict(torch.load(head_file, map_location="cpu", weights_only=True)["model"])
        self.head.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images):
        crops = [crop224(image.convert("RGB")) for image in images]
        raw = torch.from_numpy(self.encoder.cat(self.encoder.features(crops))).to(self.device)
        out = self.head(raw)
        result = dict(logits=out["logit"], probability=out["logit"].sigmoid(), features=out["v"])
        if self.config["lambda_aux"] > 0:
            result["type_probabilities"] = out["aux"].softmax(-1)      # REAL / AI_GENERATED / MANIPULATED (diagnostic)
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--features-out")
    args = ap.parse_args(argv)
    model = ImageBranchV2(args.bundle, args.device)
    result = model([load_rgb(args.image)])
    probability, threshold = float(result["probability"].item()), model.config["threshold"]
    if args.features_out:
        torch.save(result["features"].cpu(), args.features_out)
    print(json.dumps(dict(probability_fake_image=probability, prediction=int(probability >= threshold), threshold=threshold,
                          target=model.config["target"], feature_dimension=int(result["features"].shape[-1]), input="image pixels only"), indent=2))


if __name__ == "__main__":
    main()
