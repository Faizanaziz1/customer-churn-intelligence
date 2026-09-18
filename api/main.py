from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import numpy as np
import shap


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


# Load training data to create SHAP background data
background_df = pd.read_csv(
    "data/raw/telco_customer_churn.csv"
)

background_df["TotalCharges"] = background_df["TotalCharges"].str.strip()

background_df["TotalCharges"] = pd.to_numeric(
    background_df["TotalCharges"],
    errors="coerce"
)

background_df["TotalCharges"] = background_df["TotalCharges"].fillna(0)


background_X = background_df.drop(
    columns=["Churn", "customerID"]
)


background_processed = preprocessor.transform(
    background_X
)


# Create SHAP explainer
explainer = shap.LinearExplainer(
    model,
    background_processed
)


feature_names = preprocessor.get_feature_names_out()


@app.get("/")
def home():
    return {
        "message": "Customer Churn Intelligence API is running"
    }


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
def predict(customer: CustomerData):

    data = pd.DataFrame(
        [customer.model_dump()]
    )

    processed_data = preprocessor.transform(
        data
    )

    probability = model.predict_proba(
        processed_data
    )[0][1]

    prediction = int(
        probability >= 0.5
    )


    if probability >= 0.70:
        risk_level = "High"
        recommendation = (
            "Consider proactive customer retention."
        )

    elif probability >= 0.40:
        risk_level = "Medium"
        recommendation = (
            "Monitor customer and consider targeted engagement."
        )

    else:
        risk_level = "Low"
        recommendation = (
            "No immediate retention action required."
        )


    return {
        "churn_probability": round(
            float(probability), 4
        ),
        "churn_prediction": prediction,
        "risk_level": risk_level,
        "recommendation": recommendation
    }
@app.post("/explain")
def explain_customer(customer: CustomerData):

    data = pd.DataFrame(
        [customer.model_dump()]
    )

    processed_data = preprocessor.transform(data)

    probability = model.predict_proba(
        processed_data
    )[0][1]

    shap_result = explainer(
        processed_data
    )

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_result.values[0]
    })

    contributions["impact"] = np.where(
        contributions["shap_value"] > 0,
        "Increases churn risk",
        "Decreases churn risk"
    )

    contributions["absolute_impact"] = (
        contributions["shap_value"].abs()
    )

    top_factors = contributions.sort_values(
        "absolute_impact",
        ascending=False
    ).head(10)

    return {
        "churn_probability": round(
            float(probability), 4
        ),
        "top_factors": top_factors[
            ["feature", "shap_value", "impact"]
        ].to_dict(orient="records")
    }