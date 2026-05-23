import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def client():
    with patch("src.serving.api._model") as mock_model, \
         patch("src.serving.api._preprocessor") as mock_prep, \
         patch("src.serving.api._config", {
             "model": {"type": "classification", "algorithm": "random_forest"}
         }), \
         patch("src.serving.api._model_loaded_at", 1000.0):

        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_prep.transform.return_value = np.array([[1, 2, 3]])

        from src.serving.api import app
        yield TestClient(app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] in ["ok", "degraded"]


def test_predict(client):
    res = client.post("/predict", json={
        "features": {"age": 30, "income": 50000}
    })
    assert res.status_code == 200
    assert "prediction" in res.json()


def test_batch_predict(client):
    res = client.post("/predict/batch", json={
        "records": [
            {"age": 30, "income": 50000},
            {"age": 45, "income": 80000}
        ]
    })
    assert res.status_code == 200
    assert res.json()["count"] == 2


def test_model_info(client):
    res = client.get("/model/info")
    assert res.status_code == 200
    assert "model_type" in res.json()