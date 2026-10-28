from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
import joblib
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI(
    title="BMI & Weight Category Classification API",
    description="Production Machine Learning API to classify individual weight status into Underweight, Normal Weight, Overweight, or Obese",
    version="1.0.0"
)

# Model artifact path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "model.joblib"

model = None

def get_model():
    """Lazy loader to ensure model is loaded in both production and test environments."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            return None
    return model

# Pre-load model on startup if file is available
if os.path.exists(MODEL_PATH):
    get_model()

# Pydantic schema for input validation
class BMIInput(BaseModel):
    Age: int = Field(..., gt=0, lt=150, examples=[25], description="Age in years")
    Gender: str = Field(..., examples=["Male"], description="Gender ('Male' or 'Female')")
    Height: float = Field(..., gt=50, lt=300, examples=[175.0], description="Height in centimeters (cm)")
    Weight: float = Field(..., gt=5, lt=400, examples=[80.0], description="Weight in kilograms (kg)")
    BMI: Optional[float] = Field(None, examples=[25.3], description="Optional BMI; auto-calculated if not provided")

# Pydantic schema for output response
class BMIResponse(BaseModel):
    prediction: str = Field(..., description="Weight category prediction")
    bmi: float = Field(..., description="Body Mass Index used for classification")
    confidence: float = Field(..., description="Confidence probability for the predicted category")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each weight category")

@app.get("/")
def read_root():
    return {
        "service": "BMI & Weight Category Classification API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    m = get_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=BMIResponse)
def predict(data: BMIInput):
    m = get_model()
    if m is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable")

    # Normalize gender
    gender_norm = data.Gender.strip().capitalize()
    if gender_norm not in ["Male", "Female"]:
        raise HTTPException(status_code=422, detail="Gender must be 'Male' or 'Female'")

    # Calculate BMI if not provided
    calculated_bmi = data.BMI
    if calculated_bmi is None or calculated_bmi <= 0:
        height_m = data.Height / 100.0
        calculated_bmi = round(data.Weight / (height_m ** 2), 1)

    input_df = pd.DataFrame([{
        "Age": data.Age,
        "Gender": gender_norm,
        "Height": data.Height,
        "Weight": data.Weight,
        "BMI": calculated_bmi
    }])

    prediction = str(m.predict(input_df)[0])
    probabilities = m.predict_proba(input_df)[0]
    classes = m.classes_

    prob_dict = {
        cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)
    }
    confidence = prob_dict.get(prediction, round(float(max(probabilities)), 4))

    return BMIResponse(
        prediction=prediction,
        bmi=calculated_bmi,
        confidence=confidence,
        probabilities=prob_dict
    )
