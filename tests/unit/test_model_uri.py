"""Unit tests for serving model URI resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cancer_detection.serving.model_uri import resolve_model_uri


def _run(run_id: str, val_auroc: float | None) -> SimpleNamespace:
    metrics = {} if val_auroc is None else {"val/auroc": val_auroc}
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(metrics=metrics),
    )


def test_explicit_argument_wins() -> None:
    assert resolve_model_uri("runs:/abc/model") == "runs:/abc/model"


def test_env_model_uri_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_URI", "models:/melanoma-classifier/3")
    assert resolve_model_uri() == "models:/melanoma-classifier/3"


def test_selects_highest_val_auroc_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_URI", raising=False)
    client = MagicMock()
    client.search_experiments.return_value = [
        SimpleNamespace(experiment_id="1", lifecycle_stage="active")
    ]
    # Simulated server-side order_by metrics.`val/auroc` DESC
    client.search_runs.return_value = [
        _run("best", 0.91),
        _run("mid", 0.85),
        _run("low", 0.70),
        _run("no_metric", None),
    ]
    client.list_artifacts.return_value = [SimpleNamespace(path="model")]

    with (
        patch(
            "cancer_detection.serving.model_uri.ensure_tracking_uri",
            return_value="http://localhost:5000",
        ),
        patch("cancer_detection.serving.model_uri.MlflowClient", return_value=client),
    ):
        uri = resolve_model_uri()

    assert uri == "runs:/best/model"


def test_client_side_sort_when_order_by_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MODEL_URI", raising=False)
    client = MagicMock()
    client.search_experiments.return_value = [
        SimpleNamespace(experiment_id="1", lifecycle_stage="active")
    ]
    unordered = [_run("low", 0.70), _run("best", 0.91), _run("mid", 0.85)]
    client.search_runs.side_effect = [Exception("order_by unsupported"), unordered]
    client.list_artifacts.return_value = [SimpleNamespace(path="model")]

    with (
        patch(
            "cancer_detection.serving.model_uri.ensure_tracking_uri",
            return_value="http://localhost:5000",
        ),
        patch("cancer_detection.serving.model_uri.MlflowClient", return_value=client),
    ):
        uri = resolve_model_uri()

    assert uri == "runs:/best/model"


def test_skips_runs_without_model_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_URI", raising=False)
    client = MagicMock()
    client.search_experiments.return_value = [
        SimpleNamespace(experiment_id="1", lifecycle_stage="active")
    ]
    client.search_runs.return_value = [
        _run("no_model", 0.99),
        _run("has_model", 0.80),
    ]

    def list_artifacts(run_id: str, path: str | None = None):
        if path == "model":
            return (
                [SimpleNamespace(path="model/MLmodel")]
                if run_id == "has_model"
                else []
            )
        if run_id == "has_model":
            return [SimpleNamespace(path="model")]
        return []

    client.list_artifacts.side_effect = list_artifacts

    with (
        patch(
            "cancer_detection.serving.model_uri.ensure_tracking_uri",
            return_value="http://localhost:5000",
        ),
        patch("cancer_detection.serving.model_uri.MlflowClient", return_value=client),
    ):
        uri = resolve_model_uri()

    assert uri == "runs:/has_model/model"


def test_detects_model_when_absent_from_root_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLflow 3 may omit logged models from the root artifact listing."""
    monkeypatch.delenv("MODEL_URI", raising=False)
    client = MagicMock()
    client.search_experiments.return_value = [
        SimpleNamespace(experiment_id="1", lifecycle_stage="active")
    ]
    client.search_runs.return_value = [_run("best", 0.91)]

    def list_artifacts(run_id: str, path: str | None = None):
        if path == "model":
            return [SimpleNamespace(path="model/MLmodel")]
        return [SimpleNamespace(path="threshold.json")]

    client.list_artifacts.side_effect = list_artifacts

    with (
        patch(
            "cancer_detection.serving.model_uri.ensure_tracking_uri",
            return_value="http://localhost:5000",
        ),
        patch("cancer_detection.serving.model_uri.MlflowClient", return_value=client),
    ):
        uri = resolve_model_uri()

    assert uri == "runs:/best/model"


def test_raises_when_nothing_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_URI", raising=False)
    client = MagicMock()
    client.search_experiments.return_value = []

    with (
        patch(
            "cancer_detection.serving.model_uri.ensure_tracking_uri",
            return_value="http://localhost:5000",
        ),
        patch("cancer_detection.serving.model_uri.MlflowClient", return_value=client),
        pytest.raises(RuntimeError, match="No MLflow run found"),
    ):
        resolve_model_uri()
