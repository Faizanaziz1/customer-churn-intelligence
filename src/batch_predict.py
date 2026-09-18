import joblib
import pandas as pd

# Load data
df = pd.read_csv("data/raw/telco_customer_churn.csv")

# Clean TotalCharges
df["TotalCharges"] = df["TotalCharges"].str.strip()
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Keep actual churn for comparison
actual_churn = df["Churn"]

# Prepare features
X = df.drop(columns=["Churn", "customerID"])

# Load trained model and preprocessor
model = joblib.load("models/churn_logistic_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

# Transform data
X_processed = preprocessor.transform(X)

# Generate predictions
probabilities = model.predict_proba(X_processed)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Create results
results = pd.DataFrame({
    "customer_id": df["customerID"],
    "actual_churn": actual_churn,
    "churn_probability": probabilities,
    "predicted_churn": predictions
})

# Assign risk level
results["risk_level"] = results["churn_probability"].apply(
    lambda x: (
        "High" if x >= 0.70
        else "Medium" if x >= 0.40
        else "Low"
    )
)

# Save predictions
results.to_csv(
    "data/processed/churn_predictions.csv",
    index=False
)

print("Batch prediction completed successfully!")
print(f"Customers scored: {len(results)}")
print(results["risk_level"].value_counts())