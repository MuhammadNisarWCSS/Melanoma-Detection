from __future__ import annotations

import numpy as np
import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from cancer_detection.evaluation.metrics import compute_metrics
from cancer_detection.models.classifier import MelanomaClassifier
from cancer_detection.training.losses import FocalLoss
from cancer_detection.utils.logger import get_logger

logger = get_logger(__name__)


class MelanomaLitModule(LightningModule):
    """PyTorch Lightning module for melanoma classification.

    Wraps MelanomaClassifier with:
    - FocalLoss training objective
    - AdamW optimizer + CosineAnnealingLR scheduler
    - Epoch-level AUROC and F1 tracking via torchmetrics
    - Full clinical metric suite on the test split, evaluated at ``test_threshold``
    - Structured logging to the active Lightning logger (MLflow)

    save_hyperparameters() ensures the full config is stored in the checkpoint,
    enabling exact reproducibility from any saved ckpt file.
    """

    def __init__(self, model_cfg: DictConfig, training_cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = MelanomaClassifier(
            backbone_name=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            backbone_dropout=model_cfg.backbone_dropout,
            meta_hidden_dim=model_cfg.meta_hidden_dim,
            meta_output_dim=model_cfg.meta_output_dim,
            meta_dropout=model_cfg.meta_dropout,
            fusion_dropout=model_cfg.fusion_dropout,
        )

        self.criterion = FocalLoss(
            gamma=training_cfg.focal_gamma,
            alpha=training_cfg.focal_alpha,
        )

        # Separate metric objects for train/val to avoid state bleed
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()

        # Sensitivity, specificity and pAUC need the whole probability vector rather
        # than a running torchmetrics state, so the test loop buffers predictions and
        # reduces them once at epoch end. Overwrite test_threshold with the value from
        # threshold calibration before calling Trainer.test(). The reduced vectors stay
        # on the module afterwards so callers can re-sweep thresholds without re-running
        # inference.
        self.test_threshold = 0.5
        self.test_metrics: dict[str, float] = {}
        self.test_probs: np.ndarray | None = None
        self.test_labels: np.ndarray | None = None
        self._test_prob_buffer: list[torch.Tensor] = []
        self._test_label_buffer: list[torch.Tensor] = []

        self._lr = float(training_cfg.lr)
        self._weight_decay = float(training_cfg.weight_decay)
        self._epochs = int(training_cfg.epochs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.model(image, metadata)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, metadata, labels = batch
        logits = self(images, metadata)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)

        self.train_auroc.update(probs, labels.int())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/auroc", self.train_auroc.compute(), prog_bar=True)
        self.train_auroc.reset()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, metadata, labels = batch
        logits = self(images, metadata)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)

        self.val_auroc.update(probs, labels.int())
        self.val_f1.update(probs, labels.int())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        auroc = self.val_auroc.compute()
        f1 = self.val_f1.compute()
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/f1", f1, prog_bar=True)
        self.val_auroc.reset()
        self.val_f1.reset()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def on_test_epoch_start(self) -> None:
        self._test_prob_buffer.clear()
        self._test_label_buffer.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        images, metadata, labels = batch
        logits = self(images, metadata)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)

        # float() because "16-mixed" precision yields half-precision probabilities,
        # which numpy/sklearn handle poorly downstream.
        self._test_prob_buffer.append(probs.detach().float().cpu())
        self._test_label_buffer.append(labels.detach().float().cpu())
        self.log("test/loss", loss, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.test_probs = torch.cat(self._test_prob_buffer).numpy()
        self.test_labels = torch.cat(self._test_label_buffer).numpy().astype(int)
        self._test_prob_buffer.clear()
        self._test_label_buffer.clear()

        y_prob, y_true = self.test_probs, self.test_labels

        # AUROC and pAUC are undefined for a single-class split, which happens with
        # the tiny synthetic fixtures used by the smoke tests.
        if len(set(y_true.tolist())) < 2:
            logger.warning("Test split has only one class — skipping metric computation")
            return

        self.test_metrics = compute_metrics(y_true, y_prob, threshold=self.test_threshold)
        for name, value in self.test_metrics.items():
            self.log(f"test/{name}", float(value))
        logger.info("Test metrics", **self.test_metrics)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self) -> dict:  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self._epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/auroc",
            },
        }
