import pytest
from fastapi.testclient import TestClient

from app.main import app


VALID_CUSTOMER = {
    "age": 42,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 2500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["metadata_loaded"] is True


def test_predict(client):
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["yes", "no"]
    assert 0 <= data["probability_yes"] <= 1
    assert 0 <= data["prediction_probability"] <= 1
    assert data["model_version"] == "1.0.0"


def test_predict_batch(client):
    second_customer = VALID_CUSTOMER.copy()

    second_customer.update(
        {
            "age": 61,
            "job": "retired",
            "education": "secondary",
            "balance": 5000,
            "housing": "no",
            "day": 10,
            "month": "oct",
        }
    )

    response = client.post(
        "/predict-batch",
        json=[
            VALID_CUSTOMER,
            second_customer,
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2
    assert data[0]["prediction"] in ["yes", "no"]
    assert data[1]["prediction"] in ["yes", "no"]


def test_invalid_input_returns_422(client):
    invalid_customer = {
        "age": 150,
        "job": "astronaut",
    }

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422