# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multimodal melanoma classifier on ISIC 2020 dermoscopy images: EfficientNet-B4 image branch + patient-metadata MLP → fusion head → single logit. Trained with PyTorch Lightning + Hydra, tracked in MLflow, served by FastAPI, consumed by a React/Vite dashboard, deployed to EC2 via ECR/GitHub Actions.

## Commands

```bash
pip install -e ".[dev]"                       # editable install with dev extras (Python 3.11)

# Data (one-time, after manual Kaggle download into data/raw/)
python scripts/prepare_data.py                # patient-grouped train/val/test CSVs (--overwrite to regenerate)
python scripts/resize_images.py --size 448    # build data/processed/jpeg_448 training cache (required; configs point here)

# Train / evaluate
python scripts/train.py training=fast_dev     # 2-batch CPU smoke test, <60s — run this first
python scripts/train.py                       # full run (GPU)
python scripts/train.py model=efficientnet_b2 training.lr=1e-4   # Hydra overrides
python scripts/train.py -m model=efficientnet_b2,efficientnet_b4 training.lr=1e-3,5e-4  # multirun sweep
python scripts/evaluate.py [--ckpt PATH] [--threshold 0.5] [--save-predictions]
python scripts/diagnose.py --model <ckpt>     # train/test/degraded/web probability drift audit
python scripts/republish_checkpoint.py --ckpt <ckpt>  # re-log a best checkpoint as the served MLflow model

# Serve
uvicorn cancer_detection.serving.api:app --host 0.0.0.0 --port 8000 --reload
cd frontend && npm install && npm run dev     # :3000; npm run build = tsc && vite build
docker compose --project-directory . -f docker/docker-compose.yml up --build   # frontend+api+mlflow

# Quality gates (mirror .github/workflows/ci.yml)
ruff check src/cancer_detection tests scripts
ruff format --check src/cancer_detection tests scripts   # CI fails on unformatted code
mypy src/cancer_detection                                # non-blocking in CI
pytest tests/unit -v --cov=src/cancer_detection --cov-report=term-missing
pytest tests/integration -v
pytest tests/unit/test_ood.py::test_name -v              # single test
```

Tests use only synthetic fixtures (`tests/conftest.py`) — no ISIC download or trained model needed; `tests/integration/test_api.py` patches `cancer_detection.serving.api.predictor`.

## Architecture

### Config flow (Hydra)
`configs/config.yaml` composes `data/isic` + `model/efficientnet_b4` + `training/default`. Nothing is hardcoded — `scripts/train.py` is `@hydra.main`-decorated and everything downstream (`ISICDataModule`, `MelanomaLitModule`) takes `DictConfig` slices. `scripts/evaluate.py` re-composes the same tree via `initialize_config_dir` and forwards `--override` strings.

### Data path
`data/raw/train.csv` → `prepare_data.py` (`StratifiedGroupKFold` on `patient_id`) → `data/processed/{train,val,test}.csv` → `ISICDataset` reads `<image_name>.jpg` from `data.image_dir` (the resized cache, **not** `data/raw`) → `ISICDataModule` applies `WeightedRandomSampler` on train only.

### Invariants that carry real cost if broken
- **Patient-disjoint splits.** `prepare_data.assert_patient_disjoint` runs at split time and `tests/unit/test_patient_leakage.py` fails CI if any `patient_id` appears in two splits. An earlier image-level split leaked 1,656/1,657 test images and inflated test AUROC to 0.9355.
- **Deploy the best checkpoint, not the final one.** `train.py` reloads `trainer.checkpoint_callback.best_model_path` into a fresh `MelanomaLitModule` before `mlflow.pytorch.log_model`; Lightning leaves the last (most overfit) epoch in memory after `fit()`.
- **Train/serve geometry parity.** Val/TTA/serving use `SmallestMaxSize` + `CenterCrop` (`_to_square` in `data/transforms.py`), never `A.Resize` — ISIC images are 3:2 and a direct resize squashes them relative to training's `RandomResizedCrop`.
- **Deterministic TTA.** `get_tta_transforms()` returns the 8 dihedral symmetries built from fixed `np.rot90` lambdas; `A.RandomRotate90(p=1.0)` still samples `k` and made identical uploads return different probabilities. Index 0 is the identity, so `Predictor` reuses that pass for GradCAM.
- **Threshold calibration matches the serving pipeline.** `training/threshold.py` calibrates on the deployable weights through the same 8-pass TTA average, targeting sensitivity ≥ 0.80, and writes `artifacts/threshold.json` (also logged to the MLflow run).
- **`focal_alpha: 0.5`, not the RetinaNet 0.25** — the sampler already balances batches; 0.25 re-downweights positives.

### MLflow contract between training and serving
Training logs the model artifact under the name `model` and registers it as `melanoma-classifier` (alias `@champion`), plus metric `val/auroc`. `serving/model_uri.resolve_model_uri()` picks, in order: `MODEL_URI` env → the FINISHED run with highest `val/auroc` that has a `model` artifact. Changing either the artifact name or the metric key breaks model resolution silently. Lightning checkpoints stay local under `1/<run_id>/checkpoints/epoch={n}-auroc={val/auroc}.ckpt`; `evaluate.py` parses that filename to find the best one.

### Serving
`api.py` loads the predictor in a background thread from `lifespan` (a blocking load made Docker healthchecks fail) and degrades gracefully: `/health` stays 200 with `model_loaded: false`, prediction endpoints return 503. The app sets `root_path="/api"` because nginx proxies `/api` → FastAPI and `/mlflow` → MLflow so the browser only needs port 3000. `Predictor` loads once at startup and owns TTA, threshold, GradCAM, and the Mahalanobis OOD detector (`serving/ood.py`, cached to `artifacts/ood_detector.npz`, fitted from training embeddings if absent). `GradCAMWrapper` uses HiResCAM on backbone `bn2` and wraps the multimodal model with fixed metadata so gradients flow through the image only.

Serving env vars: `MODEL_URI`, `MLFLOW_TRACKING_URI`, `THRESHOLD_PATH`, `TEST_METRICS_PATH`, `DEVICE`, `TTA_N_PASSES`, `OOD_CACHE_PATH`, `OOD_TRAIN_CSV`, `OOD_IMAGE_DIR`, `OOD_N_SAMPLES`.

## Gotchas

- **The EC2 IP is duplicated in ~8 places.** `DEFAULT_TRACKING_URI` in `serving/model_uri.py`, `mlflow_uri` in both `configs/training/*.yaml`, `frontend/src/api/client.ts`, `frontend/src/components/Navbar.tsx`, `scripts/republish_checkpoint.py`, `scripts/prepare_mlflow_seed.py`, docker/deploy files. There is no Elastic IP, so all of them move together.
- Training defaults to the **hosted** MLflow, not localhost. Override with `MLFLOW_TRACKING_URI` rather than editing configs.
- `data.num_workers: 0` is deliberate — Windows spawn deadlocked while pickling the dataset. Raise only on Linux.
- `train.py`/`evaluate.py` reconfigure stdout/stderr to UTF-8 before importing MLflow (CP1252 consoles crash on MLflow's emoji output). Keep those blocks above the third-party imports; the `# noqa: E402` markers exist for the same reason.
- `train.py` registers an `atexit` guardian that marks the run FINISHED/FAILED, falling back to a direct SQLite write against `mlflow.db` if the HTTP client is unreachable — runs must never be left `RUNNING`.
- `data/`, `mlflow.db`, `mlartifacts/`, `1/`, `logs/` are local state, not sources of truth; MLflow history lives in the EC2 `mlflow-data` Docker volume.
