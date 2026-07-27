from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from cancer_detection.models.classifier import MelanomaClassifier


class _ImageOnlyWrapper(nn.Module):
    """Adapts MelanomaClassifier for GradCAM by fixing the metadata tensor.

    GradCAM computes gradients w.r.t. the input image only. Since our model
    takes (image, metadata), we freeze metadata as a buffer and expose a
    single-input forward, making the full model compatible with the GradCAM API.
    """

    def __init__(self, model: MelanomaClassifier, meta: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("meta", meta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand stored metadata to match the batch dimension of x
        meta = self.meta.expand(x.shape[0], -1)  # type: ignore[union-attr]
        logit = self.model(x, meta)
        return logit.unsqueeze(-1)  # (B, 1) for BinaryClassifierOutputTarget


def _get_target_layer(model: MelanomaClassifier) -> nn.Module:
    """Return the last spatial layer of the backbone before global pooling.

    Prefer ``bn2`` (post-``conv_head`` BatchNorm) on EfficientNet: GradCAM-style
    methods that average gradients over space are easily dominated by high-|A|
    border cells in ``conv_head`` (systematic top-right hotspots). ``bn2``
    activations are more class-calibrated. ResNet uses ``layer4[-1]``.
    """
    backbone = model.backbone
    if hasattr(backbone, "bn2"):
        return backbone.bn2  # EfficientNet (timm) — after conv_head 1×1
    if hasattr(backbone, "conv_head"):
        return backbone.conv_head
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]  # ResNet (timm)
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]
    raise ValueError(
        f"Cannot determine GradCAM target layer for backbone type {type(backbone).__name__}. "
        "Add a branch for your architecture in explainability/gradcam.py."
    )


class GradCAMWrapper:
    """High-level GradCAM interface for MelanomaClassifier.

    Wraps pytorch-grad-cam to handle the multimodal architecture: GradCAM
    is applied to the image branch only, with metadata held constant.

    Usage:
        wrapper = GradCAMWrapper(model)
        heatmap = wrapper.generate_heatmap(image_tensor, metadata_tensor, original_image_np)
        b64_str  = GradCAMWrapper.heatmap_to_base64(heatmap)
    """

    def __init__(self, model: MelanomaClassifier) -> None:
        self.model = model
        self._target_layer = _get_target_layer(model)

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        metadata_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_category: int = 1,
        image_size: int = 384,
    ) -> np.ndarray:
        """Compute GradCAM heatmap and overlay it on the original image.

        Args:
            image_tensor:    Normalised image tensor, shape (C, H, W). Not batched.
            metadata_tensor: Encoded metadata tensor, shape (3,). Not batched.
            original_image:  Raw uint8 RGB numpy array before normalisation.
            target_category: Predicted class to explain — 1=malignant, 0=benign.
                             Must match the model's decision. Targeting the opposite
                             class (always malignant) makes GradCAM's ReLU suppress
                             the lesion and leave only corner/artifact noise.
            image_size:      Resize original_image to this before overlay.

        Returns:
            RGB uint8 numpy array (image_size × image_size × 3) with heatmap overlay.
        """
        meta_1d = metadata_tensor.unsqueeze(0) if metadata_tensor.dim() == 1 else metadata_tensor
        wrapper = _ImageOnlyWrapper(self.model, meta_1d)
        # Keep wrapper buffers (metadata) on the same device as the image input.
        input_batch = image_tensor.unsqueeze(0)
        wrapper = wrapper.to(input_batch.device)

        # HiResCAM (not vanilla GradCAM): GradCAM globally averages gradients, so
        # high-|activation| border cells dominate → systematic top-right hotspots
        # even for malignant predictions. HiResCAM keeps spatial gradients
        # (element-wise A⊙ReLU(∂y/∂A)), which suppresses those border artifacts.
        # BinaryClassifierOutputTarget: category 1 → +logit, category 0 → -logit
        # so we explain the *predicted* class (ReLU would otherwise zero lesion
        # evidence on benign calls).
        with torch.enable_grad():
            with HiResCAM(model=wrapper, target_layers=[self._target_layer]) as cam:
                targets = [BinaryClassifierOutputTarget(target_category)]
                grayscale_cam = cam(
                    input_tensor=input_batch,
                    targets=targets,
                )
        grayscale_cam = grayscale_cam[0]  # (H_cam, W_cam)

        # Resize original image for overlay; normalise to [0, 1]
        img_resized = cv2.resize(original_image, (image_size, image_size))
        img_float = img_resized.astype(np.float32) / 255.0
        if img_float.shape[:2] != grayscale_cam.shape:
            grayscale_cam = cv2.resize(grayscale_cam, (image_size, image_size))

        overlay: np.ndarray = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        return overlay

    @staticmethod
    def heatmap_to_base64(heatmap: np.ndarray) -> str:
        """Encode a uint8 RGB numpy array as a base64 PNG string."""
        from PIL import Image

        pil_img = Image.fromarray(heatmap)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
