"""
Stand-alone model training script for BMI and Weight Category Classification.
Trains a Scikit-Learn Pipeline and exports model.joblib.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_and_export():
    dataset_path = "bmi_dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "https://raw.githubusercontent.com/SubraT933/ca-ai/main/bmi_dataset.csv"

    print(f"Loading data from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    features = ["Age", "Gender", "Height", "Weight", "BMI"]
    X = df[features]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Gender"]),
            ("num", "passthrough", ["Age", "Height", "Weight", "BMI"])
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training Random Forest Classifier Pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    output_path = "model.joblib"
    joblib.dump(pipeline, output_path)
    print(f"Model successfully exported to {output_path}")

if __name__ == "__main__":
    train_and_export()
