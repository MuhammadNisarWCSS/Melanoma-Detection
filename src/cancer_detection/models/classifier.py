from __future__ import annotations

import torch
import torch.nn as nn

from cancer_detection.models.backbone import create_backbone


class MetadataMLP(nn.Module):
    """Small MLP that encodes 3 tabular patient features into a dense vector.

    Architecture: Linear → BN → ReLU → Dropout → Linear → ReLU
    BatchNorm before ReLU stabilizes training when metadata features have
    different scales (normalized age, binary sex, ordinal site).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MelanomaClassifier(nn.Module):
    """Multimodal melanoma classifier.

    Image branch:    EfficientNet-B4 → AdaptiveAvgPool → 1792-d feature vector
    Metadata branch: MetadataMLP(3 → 64 → 32-d)
    Fusion head:     cat(1792+32) → Linear(512) → ReLU → Dropout → Linear(1)

    Returns raw logits (no sigmoid). Use torch.sigmoid() for probabilities.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        backbone_dropout: float = 0.0,
        meta_hidden_dim: int = 64,
        meta_output_dim: int = 32,
        meta_dropout: float = 0.3,
        fusion_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone, img_feature_dim = create_backbone(
            backbone_name, pretrained=pretrained, drop_rate=backbone_dropout
        )
        self.meta_branch = MetadataMLP(
            input_dim=3,
            hidden_dim=meta_hidden_dim,
            output_dim=meta_output_dim,
            dropout=meta_dropout,
        )
        fusion_in = img_feature_dim + meta_output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(512, 1),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:    (B, 3, H, W) normalized image tensor
            metadata: (B, 3) encoded patient metadata tensor

        Returns:
            (B,) raw logits
        """
        img_feat = self.backbone(image)           # (B, img_feature_dim)
        meta_feat = self.meta_branch(metadata)    # (B, meta_output_dim)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.fusion_head(fused).squeeze(1)  # (B,)
