from __future__ import annotations

from cancer_detection.models.classifier import MelanomaClassifier

# Maps config backbone names → MelanomaClassifier kwargs overrides.
# All models use the same MelanomaClassifier architecture; only the backbone differs.
_CONFIGS: dict[str, dict] = {
    "efficientnet_b0": {"backbone_name": "efficientnet_b0", "fusion_dropout": 0.3},
    "efficientnet_b2": {"backbone_name": "efficientnet_b2", "fusion_dropout": 0.4},
    "efficientnet_b4": {"backbone_name": "efficientnet_b4", "fusion_dropout": 0.5},
    "resnet50": {"backbone_name": "resnet50", "fusion_dropout": 0.5},
}


def get_model(name: str, pretrained: bool = True, **kwargs: object) -> MelanomaClassifier:
    """Instantiate a MelanomaClassifier by registry name.

    Args:
        name: Key in the registry (e.g. 'efficientnet_b4').
        pretrained: Pass through to backbone creation.
        **kwargs: Override any MelanomaClassifier constructor argument.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in _CONFIGS:
        available = ", ".join(_CONFIGS)
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    config = {**_CONFIGS[name], "pretrained": pretrained, **kwargs}
    return MelanomaClassifier(**config)


def list_models() -> list[str]:
    return list(_CONFIGS.keys())
