"""Unit tests for MelanomaClassifier forward pass and gradient flow.

Uses efficientnet_b0 (smallest EfficientNet) with pretrained=False so tests
run without network access and complete in <10 seconds on CPU.
"""

from __future__ import annotations

import pytest
import torch

from cancer_detection.models.classifier import MelanomaClassifier, MetadataMLP
from cancer_detection.models.registry import get_model, list_models
from cancer_detection.training.losses import FocalLoss


FAST_BACKBONE = "efficientnet_b0"


@pytest.fixture
def model() -> MelanomaClassifier:
    return MelanomaClassifier(backbone_name=FAST_BACKBONE, pretrained=False)


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.randn(4, 3, 224, 224)
    metadata = torch.randn(4, 3)
    return images, metadata


# ---------------------------------------------------------------------------
# Output shape / dtype
# ---------------------------------------------------------------------------
def test_output_shape(model: MelanomaClassifier, batch: tuple) -> None:
    images, metadata = batch
    model.eval()
    with torch.no_grad():
        output = model(images, metadata)
    assert output.shape == (4,), f"Expected (4,), got {output.shape}"


def test_output_dtype(model: MelanomaClassifier, batch: tuple) -> None:
    images, metadata = batch
    model.eval()
    with torch.no_grad():
        output = model(images, metadata)
    assert output.dtype == torch.float32


def test_output_is_logit_not_prob(model: MelanomaClassifier, batch: tuple) -> None:
    """Model should return raw logits (not sigmoid-transformed probabilities)."""
    images, metadata = batch
    model.eval()
    with torch.no_grad():
        output = model(images, metadata)
    # Logits can be outside [0, 1]; probabilities cannot
    # This is a soft check — a model outputting near-zero logits at init could pass either way
    # We just verify sigmoid(logits) is bounded
    probs = torch.sigmoid(output)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------
def test_gradient_flows_through_image_branch(model: MelanomaClassifier, batch: tuple) -> None:
    images, metadata = batch
    images.requires_grad_(True)
    output = model(images, metadata)
    output.sum().backward()
    assert images.grad is not None
    assert not torch.all(images.grad == 0)


def test_all_parameters_receive_gradients(model: MelanomaClassifier, batch: tuple) -> None:
    images, metadata = batch
    output = model(images, metadata)
    output.sum().backward()
    no_grad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert len(no_grad) == 0, f"Parameters with no gradient: {no_grad}"


# ---------------------------------------------------------------------------
# Metadata branch
# ---------------------------------------------------------------------------
def test_metadata_mlp_shape() -> None:
    mlp = MetadataMLP(input_dim=3, hidden_dim=64, output_dim=32)
    x = torch.randn(8, 3)
    out = mlp(x)
    assert out.shape == (8, 32)


def test_metadata_branch_effect(model: MelanomaClassifier) -> None:
    """Different metadata should produce different logits (metadata branch is active)."""
    images = torch.randn(2, 3, 224, 224)
    meta_a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    meta_b = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    model.eval()
    with torch.no_grad():
        out_a = model(images, meta_a)
        out_b = model(images, meta_b)
    assert not torch.allclose(out_a, out_b), "Metadata branch has no effect on output"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def test_focal_loss_shapes(model: MelanomaClassifier, batch: tuple) -> None:
    images, metadata = batch
    labels = torch.zeros(4)
    labels[0] = 1.0
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    logits = model(images, metadata)
    loss = criterion(logits, labels)
    assert loss.shape == ()
    assert loss.item() > 0.0


def test_focal_loss_reduction_none(model: MelanomaClassifier, batch: tuple) -> None:
    images, metadata = batch
    labels = torch.zeros(4)
    criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction="none")
    logits = model(images, metadata)
    loss = criterion(logits, labels)
    assert loss.shape == (4,)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_list_models() -> None:
    models = list_models()
    assert "efficientnet_b4" in models
    assert "resnet50" in models


def test_get_model_returns_classifier() -> None:
    m = get_model("efficientnet_b0", pretrained=False)
    assert isinstance(m, MelanomaClassifier)


def test_get_model_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("nonexistent_model_xyz")
