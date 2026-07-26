"""Integration tests for the FastAPI serving layer.

The predictor (MLflow model) is mocked so no actual model weights are needed.
Tests verify the full HTTP layer: routing, input validation, response schema.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cancer_detection.serving.api import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def jpeg_bytes() -> bytes:
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


_MOCK_RESULT = {
    "probability": 0.35,
    "label": 0,
    "label_str": "benign",
    "confidence": 0.30,
    "tta_std": 0.04,
    "threshold_used": 0.5,
    "gradcam_heatmap_b64": "iVBORw0KGgo=",
}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
def test_health_no_model(client: TestClient) -> None:
    """Health check returns 200 even when model is not loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


# ---------------------------------------------------------------------------
# Predict endpoint — with mocked predictor
# ---------------------------------------------------------------------------
def test_predict_returns_200(client: TestClient, jpeg_bytes: bytes) -> None:
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = _MOCK_RESULT.copy()

    with patch("cancer_detection.serving.api.predictor", mock_predictor):
        response = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
            data={
                "age_approx": "45.0",
                "sex": "male",
                "anatom_site": "torso",
                "return_gradcam": "true",
            },
        )

    assert response.status_code == 200


def test_predict_response_schema(client: TestClient, jpeg_bytes: bytes) -> None:
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = _MOCK_RESULT.copy()

    with patch("cancer_detection.serving.api.predictor", mock_predictor):
        response = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
            data={"age_approx": "50.0", "sex": "female", "anatom_site": "head/neck"},
        )

    data = response.json()
    assert "probability" in data
    assert "label" in data
    assert "label_str" in data
    assert "confidence" in data
    assert "tta_std" in data
    assert "threshold_used" in data
    assert data["label"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_no_model_returns_503(client: TestClient, jpeg_bytes: bytes) -> None:
    """When predictor is None (model not loaded), API should return 503."""
    with patch("cancer_detection.serving.api.predictor", None):
        response = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg_bytes, "image/jpeg")},
            data={"age_approx": "50.0"},
        )
    assert response.status_code == 503


def test_predict_invalid_image_returns_400(client: TestClient) -> None:
    """Sending non-image bytes should trigger a 400 error."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = _MOCK_RESULT.copy()

    with patch("cancer_detection.serving.api.predictor", mock_predictor):
        response = client.post(
            "/predict",
            files={"image": ("bad.jpg", b"not an image at all", "image/jpeg")},
            data={"age_approx": "50.0"},
        )
    assert response.status_code == 400


def test_batch_predict_returns_list(client: TestClient, jpeg_bytes: bytes) -> None:
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = {**_MOCK_RESULT, "gradcam_heatmap_b64": None}

    with patch("cancer_detection.serving.api.predictor", mock_predictor):
        response = client.post(
            "/predict/batch",
            files=[
                ("images", ("img1.jpg", jpeg_bytes, "image/jpeg")),
                ("images", ("img2.jpg", jpeg_bytes, "image/jpeg")),
            ],
            data={"age_approx": "50.0", "sex": "male", "anatom_site": "torso"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert data["count"] == 2
