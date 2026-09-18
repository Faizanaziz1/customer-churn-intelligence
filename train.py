import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv("data/raw/telco_customer_churn.csv")


# --------------------------------------------------
# 2. Clean data
# --------------------------------------------------

df["TotalCharges"] = df["TotalCharges"].str.strip()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)


# --------------------------------------------------
# 3. Encode target
# --------------------------------------------------

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# 4. Create X and y
# --------------------------------------------------

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]


# --------------------------------------------------
# 5. Define features
# --------------------------------------------------

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    col for col in X.columns
    if col not in numerical_features
]


# --------------------------------------------------
# 6. Create preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 7. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 8. Preprocess data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)


# --------------------------------------------------
# 9. Train model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_processed,
    y_train
)


# --------------------------------------------------
# 10. Save model and preprocessor
# --------------------------------------------------

joblib.dump(
    model,
    "models/churn_logistic_model.joblib"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.joblib"
)

print("Model training completed successfully!")
print("Model saved to models/churn_logistic_model.joblib")
print("Preprocessor saved to models/preprocessor.joblib")