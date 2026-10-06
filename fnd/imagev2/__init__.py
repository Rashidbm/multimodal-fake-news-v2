"""Image Branch v2 comparison: frozen-feature extraction, shared bottleneck/head, evaluation.

Every arm (A0/A1: current CLIP/RGB/DCT features, B1: DINOv3 features) goes through the same
manifests, canonical image, sampler, bottleneck, losses and metrics. Only the feature
extractor differs. See docs/IMAGE_BRANCH_V2_EXPERIMENT_REPORT.md.
"""
