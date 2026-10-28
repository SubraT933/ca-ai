import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "BMI & Weight Category Classification API" in data["service"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_normal_weight():
    payload = {
        "Age": 25,
        "Gender": "Male",
        "Height": 175.0,
        "Weight": 80.0,
        "BMI": 25.3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Normal Weight"
    assert data["bmi"] == 25.3
    assert 0.0 <= data["confidence"] <= 1.0
    assert "Normal Weight" in data["probabilities"]

def test_predict_underweight():
    payload = {
        "Age": 15,
        "Gender": "Male",
        "Height": 175.0,
        "Weight": 45.0,
        "BMI": 18.7
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Underweight"
    assert data["bmi"] == 18.7
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_obese():
    payload = {
        "Age": 45,
        "Gender": "Male",
        "Height": 190.0,
        "Weight": 100.0,
        "BMI": 31.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Obese"
    assert data["bmi"] == 31.2
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_auto_calculate_bmi():
    # Test omission of BMI field triggers auto calculation: 80 / (1.75^2) = 26.1
    payload = {
        "Age": 29,
        "Gender": "Male",
        "Height": 175.0,
        "Weight": 80.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["bmi"] == 26.1
    assert data["prediction"] in ["Normal Weight", "Overweight", "Obese", "Underweight"]

def test_predict_invalid_gender():
    payload = {
        "Age": 25,
        "Gender": "Other",
        "Height": 175.0,
        "Weight": 80.0,
        "BMI": 25.3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_age():
    payload = {
        "Age": -5,
        "Gender": "Male",
        "Height": 175.0,
        "Weight": 80.0,
        "BMI": 25.3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
