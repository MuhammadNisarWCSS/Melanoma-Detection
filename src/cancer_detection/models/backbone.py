from __future__ import annotations

import timm
import torch.nn as nn


def create_backbone(
    name: str,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> tuple[nn.Module, int]:
    """Instantiate a timm backbone with the classification head removed.

    Args:
        name: Any timm model name (e.g. 'efficientnet_b4', 'resnet50').
        pretrained: Load ImageNet-pretrained weights.
        drop_rate: Dropout rate applied inside the backbone (timm feature).

    Returns:
        (backbone_module, feature_dim) where feature_dim is the number of
        channels output by the final pooling layer (before classification head).
    """
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,  # remove the classification head
        drop_rate=drop_rate,
        global_pool="avg",  # adaptive average pool → flat feature vector
    )
    feature_dim: int = model.num_features
    return model, feature_dim
