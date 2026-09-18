from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel


app = FastAPI(
    title="Customer Churn Intelligence API",
    description="API for predicting customer churn",
    version="1.0.0"
)


# Load trained model and preprocessing pipeline
model = joblib.load(
    "models/churn_logistic_model.joblib"
)

preprocessor = joblib.load(
    "models/preprocessor.joblib"
)


@app.get("/")
def home():
    from pydantic import BaseModel


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict_churn(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    processed_data = preprocessor.transform(data)

    probability = model.predict_proba(processed_data)[0][1]

    prediction = int(probability >= 0.5)

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction
    }