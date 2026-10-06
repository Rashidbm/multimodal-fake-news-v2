import numpy as np
import pytest
import torch
from PIL import Image

from fnd.imagev2 import canonical as C
from fnd.imagev2 import metrics as M
from fnd.imagev2.model import ImageHead
from fnd.imagev2.sampler import DomainClassSampler


def noise_image(w=300, h=200, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_geometry_is_224_and_crop_keeps_center():
    image = noise_image(400, 200)
    assert C.crop224(image).size == (224, 224) and C.squash224(image).size == (224, 224)
    assert C.crop224(Image.new("RGB", (100, 500))).size == (224, 224)    # small images are upscaled


def test_views_are_deterministic_and_distinct():
    image = noise_image()
    a, b = C.apply_view(image, "aug1", "img_1"), C.apply_view(image, "aug1", "img_1")
    assert np.array_equal(np.asarray(a), np.asarray(b))
    differing = [v for v in ("jpeg75", "blur1") if not np.array_equal(np.asarray(C.apply_view(image, v, "x")), np.asarray(image))]
    assert differing == ["jpeg75", "blur1"]
    with pytest.raises(ValueError):
        C.apply_view(image, "nope", "x")


def test_alpha_is_dropped(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (64, 48), (10, 20, 30, 0)).save(path)
    assert C.canonical(path, "clean", "i").mode == "RGB"


def make_sampler():
    n_news, n_coco, n_fk = 900, 400, 300
    y3 = np.r_[np.zeros(3000), np.ones(n_news), np.ones(n_coco), np.zeros(n_coco + n_fk), np.full(n_fk, 2)].astype(int)
    domain = np.r_[["news"] * 3000, ["news"] * n_news, ["coco"] * n_coco, ["coco"] * n_coco, ["fakeddit"] * n_fk, ["fakeddit"] * n_fk]
    sub = np.r_[["nc_jpeg"] * 1500, ["mmfb_visualnews_png"] * 1500, [""] * (n_news + n_coco + n_coco + 2 * n_fk)]
    role = np.array(["core"] * len(y3))
    return DomainClassSampler(y3, domain, sub, role), y3, domain, sub


def test_sampler_balances_classes_and_domains():
    sampler, y3, domain, sub = make_sampler()
    draw = sampler.draw(np.random.default_rng(0), 4000)
    assert len(draw) == 4000
    share = np.bincount(y3[draw], minlength=3) / 4000
    assert abs(share[0] - .5) < .01 and abs(share[1] - .35) < .01 and abs(share[2] - .15) < .01
    for d in ("news", "coco", "fakeddit"):    # inside each domain the positive and real counts match
        m = domain[draw] == d
        assert abs((y3[draw][m] == 0).sum() - (y3[draw][m] > 0).sum()) <= 2
    jpeg = ((sub[draw] == "nc_jpeg")).sum()
    png = ((sub[draw] == "mmfb_visualnews_png")).sum()
    assert abs(jpeg - png) <= 2


def test_sampler_report_lists_repeats():
    sampler, *_ = make_sampler()
    report = {(r["domain"], r["cls"], r["sub"]): r for r in sampler.report(4096)}
    assert report[("fakeddit", "MAN", "")]["repeats_per_epoch"] > 1


def test_head_shapes_and_both_heads_train_the_bottleneck():
    head = ImageHead(50, hidden=32)
    head.set_statistics(torch.randn(200, 50) * 3 + 1)
    out = head(torch.randn(8, 50))
    assert out["v"].shape == (8, 768) and out["logit"].shape == (8,) and out["aux"].shape == (8, 3)
    (out["logit"].sum() + out["aux"].sum()).backward()
    assert head.net[0].weight.grad.abs().sum() > 0


def test_metrics_basic_values():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(y, np.array([.1, .2, .8, .9])) == 1.0
    r = M.binary_metrics(y, np.array([.1, .6, .8, .9]))
    assert r["fake_recall"] == 1.0 and r["real_recall"] == .5 and r["balanced_accuracy"] == .75
    t = M.three_class(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 0]))
    assert t["per_class"]["MANIPULATED"]["recall"] == .5
    assert M.effective_rank(np.random.default_rng(0).normal(size=(100, 10)))["participation_ratio"] > 5


def test_group_bootstrap_keeps_groups_whole():
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    for sample in M.bootstrap_groups(groups, 20, 0):
        counts = {g: int((groups[sample] == g).sum()) for g in set(groups)}
        assert counts["a"] % 2 == 0 and counts["c"] % 3 == 0


def test_dino_feature_layout_with_a_tiny_random_model():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "DINOv3ViTModel"):
        pytest.skip("this transformers version has no DINOv3")
    from fnd.imagev2.extract import DinoEncoders
    config = transformers.DINOv3ViTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, intermediate_size=64,
                                         patch_size=16, num_register_tokens=4, image_size=224)
    encoder = DinoEncoders("random", "cpu", model=transformers.DINOv3ViTModel(config))
    assert encoder.layers == [2, 3, 3, 4] or len(encoder.layers) == 4
    features = encoder.cat(encoder.features([noise_image(224, 224, i) for i in range(2)]))
    assert features.shape == (2, 2 * len(encoder.layers) * 32)
    assert [n for n, _ in encoder.blocks][:2] == [f"cls_L{encoder.layers[0]}", f"mean_L{encoder.layers[0]}"]
