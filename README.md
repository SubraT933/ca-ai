# BMI & Weight Category Classification API

[![CI Pipeline](https://github.com/SubraT933/ca-ai/actions/workflows/ci.yaml/badge.svg)](https://github.com/SubraT933/ca-ai/actions/workflows/ci.yaml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SubraT933/ca-ai/blob/main/bmi_weight_classification.ipynb)

A production-ready Machine Learning system that classifies individuals into health weight categories (`Underweight`, `Normal Weight`, `Overweight`, `Obese`) based on physical measurements (Age, Gender, Height, Weight, and BMI).

---

## 1. Machine Learning Case Study Overview

- **Problem Type**: Multiclass Classification
- **Dataset**: Body Mass Index (BMI) & Physical Measurement Dataset (`bmi_dataset.csv`)
- **Features**: `Age`, `Gender`, `Height` (cm), `Weight` (kg), `BMI`
- **Target Label**: `Underweight`, `Normal Weight`, `Overweight`, `Obese`
- **Model**: Scikit-Learn Pipeline (`ColumnTransformer` with `OneHotEncoder` + `RandomForestClassifier`)
- **Evaluation Accuracy**: 95.45% on holdout test set

---

## 2. Google Colab / Jupyter Notebook

The full interactive workflow is implemented in [`bmi_weight_classification.ipynb`](bmi_weight_classification.ipynb) (and mirrored in `notebooks/`):

1. **One-click Colab Execution**: Click the badge above or open `bmi_weight_classification.ipynb` directly in [Google Colab](https://colab.research.google.com/github/SubraT933/ca-ai/blob/main/bmi_weight_classification.ipynb).
2. The notebook automatically falls back to loading `bmi_dataset.csv` directly from GitHub if not already present in the local runtime.
3. Steps included:
   - Data loading & exploratory data analysis (distributions, statistics)
   - Preprocessing with `ColumnTransformer` & stratified 80/20 train/test split
   - Model training with `RandomForestClassifier`
   - Evaluation with Classification Report and Confusion Matrix
   - Exporting the trained pipeline artifact to `model.joblib`
   - Sample inference testing

---

## 3. Local Setup & Training

### Prerequisites
- Python 3.10+
- Pip

### Installation
```bash
pip install -r requirements.txt
```

### Train Model via CLI
```bash
python train.py
```
This generates and saves the production pipeline to `model.joblib`.

---

## 4. FastAPI Service

### Start Local API Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger Documentation will be available at: **http://localhost:8000/docs**

### Endpoints
- `GET /`: Service information and status.
- `GET /health`: Health check verifying model availability.
- `POST /predict`: Predict weight category from physical parameters.

#### Sample Request (`POST /predict`):
```json
{
  "Age": 25,
  "Gender": "Male",
  "Height": 175.0,
  "Weight": 80.0,
  "BMI": 25.3
}
```

#### Sample Response:
```json
{
  "prediction": "Normal Weight",
  "bmi": 25.3,
  "confidence": 0.95,
  "probabilities": {
    "Normal Weight": 0.95,
    "Obese": 0.0,
    "Overweight": 0.05,
    "Underweight": 0.0
  }
}
```

---

## 5. Automated Testing

Execute the unit test suite covering health checks, prediction outputs, auto-BMI calculation, and schema validation:
```bash
python -m pytest -v
```

---

## 6. Docker Containerization

### Build Docker Image
```bash
docker build -t bmi-classification-api:latest .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name bmi-api bmi-classification-api:latest
```
Access endpoints at `http://localhost:8000`.

---

## 7. CI/CD Pipeline (GitHub Actions)

Continuous Integration is configured in [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml):
- Checks out code on every push and pull request to `main`.
- Sets up Python 3.11 with pip caching.
- Installs runtime dependencies.
- Executes automated test suite with `pytest`.
- Builds Docker image to guarantee production deployability.
