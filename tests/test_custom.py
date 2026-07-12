# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import DraxBlock, DraxNet


ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "ultralytics" / "cfg" / "models" / "ext"
CFG_AVE = CFG_DIR / "draxnet-ave-yolo26.yml"
CFG_SKNET = CFG_DIR / "draxnet-sknet-yolo26.yml"


def test_draxnet_yolo26_variants_build_and_forward():
    """Build both DraxNet YOLO26 fusion variants and run minimal forward passes."""
    for config, fusion_mode in ((CFG_AVE, "average"), (CFG_SKNET, "sknet")):
        model = YOLO(config)
        outputs = model.model(torch.randn(1, 3, 64, 64))
        backbone = model.model.model[0]

        assert isinstance(backbone, DraxNet)
        assert backbone.fusion_mode == fusion_mode
        assert all(block.drax.fusion_mode == fusion_mode for block in backbone.layer4)
        assert isinstance(outputs, dict)
        assert {"one2many", "one2one"} <= set(outputs)


def test_draxnet_backbone_feature_shapes():
    """Validate the standalone DraxNet backbone P3/P4/P5 feature shapes."""
    backbone = DraxNet(3)
    features = backbone(torch.randn(1, 3, 64, 64))

    assert len(features) == 3
    assert [tuple(x.shape) for x in features] == [(1, 256, 8, 8), (1, 512, 4, 4), (1, 1024, 2, 2)]


def test_drax_sknet_fusion_weights_are_channel_normalized():
    """Verify SKNet-style branch weights form a per-channel convex combination."""
    block = DraxBlock(dim=8, efficient=False, fusion_mode="sknet")
    conv_delta = torch.randn(1, 8, 4, 4)
    attention_delta = torch.randn(1, 8, 4, 4)

    logits = block.fusion_gate(conv_delta + attention_delta)
    weights = logits.reshape(1, 2, 8, 1, 1).softmax(dim=1)

    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]))


def test_drax_average_fusion_has_no_gate_parameters():
    """Keep fixed-average Drax checkpoints free of SKNet gate parameters."""
    block = DraxBlock(dim=8, fusion_mode="average")

    assert block.fusion_gate is None
    assert not any(name.startswith("fusion_gate") for name, _ in block.named_parameters())
