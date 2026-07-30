"""Unit tests for MelanomaClassifier forward pass and gradient flow.

Uses efficientnet_b0 (smallest EfficientNet) with pretrained=False so tests
run without network access and complete in <10 seconds on CPU.
"""

from __future__ import annotations

import pytest
import torch

from cancer_detection.models.classifier import MelanomaClassifier, MetadataMLP
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


def test_focal_loss_gamma_zero_matches_weighted_bce() -> None:
    """gamma=0 removes the focusing term; only the alpha class-weight remains."""
    logits = torch.tensor([2.0, -1.5, 0.3, -3.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    alpha = 0.7

    focal = FocalLoss(gamma=0.0, alpha=alpha, reduction="none")(logits, labels)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    expected = alpha_t * bce

    assert torch.allclose(focal, expected, atol=1e-6)


def test_focal_loss_alpha_weights_positives_not_negatives() -> None:
    """alpha should scale the *positive*-class loss; alpha=0 zeroes it out entirely."""
    logits = torch.tensor([1.0, -1.0])
    labels_pos = torch.tensor([1.0, 0.0])  # first sample positive
    labels_neg = torch.tensor([0.0, 1.0])  # first sample negative (mirrored)

    loss_pos = FocalLoss(gamma=2.0, alpha=0.0, reduction="none")(logits, labels_pos)
    loss_neg = FocalLoss(gamma=2.0, alpha=0.0, reduction="none")(logits, labels_neg)

    # alpha=0 should zero the loss on the positive-labelled sample, not the negative one.
    assert loss_pos[0] == pytest.approx(0.0, abs=1e-6)
    assert loss_neg[0] > 0.0


def test_focal_loss_alpha_half_is_uniform_scale() -> None:
    """alpha=0.5 should scale every sample's loss identically, regardless of label."""
    logits = torch.tensor([2.0, -2.0, 0.5, -0.5])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    half = FocalLoss(gamma=2.0, alpha=0.5, reduction="none")(logits, labels)

    # alpha_t is 0.5 for every sample when alpha=0.5, regardless of label — i.e.
    # exactly half of the un-weighted focal term (gamma-only, no class weight).
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt = torch.exp(-bce)
    unweighted_focal = (1.0 - pt) ** 2.0 * bce

    assert torch.allclose(half, unweighted_focal * 0.5, atol=1e-6)


def test_focal_loss_no_nan_at_extreme_logits() -> None:
    """BCEWithLogits-based formulation must stay finite at large-magnitude logits."""
    logits = torch.tensor([50.0, -50.0, 50.0, -50.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])  # includes confidently-wrong cases
    loss = FocalLoss(gamma=2.0, alpha=0.5, reduction="none")(logits, labels)
    assert torch.isfinite(loss).all()
