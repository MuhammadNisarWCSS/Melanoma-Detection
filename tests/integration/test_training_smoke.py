"""Integration smoke test: run 2 batches of training without GPU.

Uses fast_dev_run=True (Lightning built-in) with the tiny efficientnet_b0
backbone on 64×64 synthetic images. No real ISIC data needed.
Completes in < 60 seconds on CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from cancer_detection.data.datamodule import ISICDataModule
from cancer_detection.training.lit_module import MelanomaLitModule


def _make_synthetic_data(tmp_path: Path, n: int = 20) -> tuple[pd.DataFrame, Path]:
    """Create synthetic images + CSV in tmp_path."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        name = f"img_{i:04d}"
        arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{name}.jpg")
        rows.append(
            {
                "image_name": name,
                "target": 1 if i < 2 else 0,
                "age_approx": float(rng.integers(20, 80)),
                "sex": "male" if i % 2 == 0 else "female",
                "anatom_site_general_challenge": "torso",
            }
        )
    df = pd.DataFrame(rows)
    csv_dir = tmp_path / "processed"
    csv_dir.mkdir()
    df.to_csv(csv_dir / "train.csv", index=False)
    df.to_csv(csv_dir / "val.csv", index=False)
    df.to_csv(csv_dir / "test.csv", index=False)
    return df, img_dir


def test_training_smoke_run(tmp_path: Path) -> None:
    """Assert that 2 batches of train + val complete without exceptions."""
    _, img_dir = _make_synthetic_data(tmp_path)
    processed_dir = tmp_path / "processed"

    data_cfg = OmegaConf.create(
        {
            "raw_dir": str(tmp_path),
            "processed_dir": str(processed_dir),
            "image_dir": str(img_dir),
            "image_size": 64,
            "num_workers": 0,  # no multiprocessing in CI
            "val_split": 0.2,
            "test_split": 0.1,
            "random_state": 42,
        }
    )
    training_cfg = OmegaConf.create(
        {
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 2,
            "precision": "32",
            "early_stopping_patience": 2,
            "focal_gamma": 2.0,
            "focal_alpha": 0.5,
            "positive_sample_rate": 0.15,
            "seed": 42,
            "experiment_name": "smoke",
            "mlflow_uri": str(tmp_path / "mlruns"),
        }
    )
    model_cfg = OmegaConf.create(
        {
            "backbone": "efficientnet_b0",
            "pretrained": False,
            "backbone_dropout": 0.0,
            "meta_hidden_dim": 16,
            "meta_output_dim": 8,
            "meta_dropout": 0.1,
            "fusion_dropout": 0.2,
        }
    )

    from lightning.pytorch import Trainer

    datamodule = ISICDataModule(data_cfg, training_cfg)
    lit_module = MelanomaLitModule(model_cfg, training_cfg)

    trainer = Trainer(
        max_epochs=1,
        fast_dev_run=2,  # run exactly 2 train batches + 2 val batches then stop
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    # Should not raise
    trainer.fit(lit_module, datamodule)

    # fast_dev_run runs step-level hooks (training_step logs "train/loss" on_step=True)
    # but skips epoch-end hooks, so only the step-logged key is guaranteed present.
    assert "train/loss_step" in trainer.callback_metrics
