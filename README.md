# Customer Churn Intelligence Platform

An end-to-end customer churn prediction and analytics project built using Python, PostgreSQL, Machine Learning, SHAP, FastAPI, and Power BI.

## Project Overview

This project analyzes customer behavior, predicts churn probability, explains machine learning predictions using SHAP, provides REST APIs using FastAPI, and presents insights through an interactive Power BI dashboard.

## Key Features

- Customer churn analysis
- Machine learning churn prediction
- SHAP-based explainability
- PostgreSQL database integration
- FastAPI prediction API
- Batch prediction pipeline
- Interactive Power BI dashboard

## Dataset

This project uses the IBM Telco Customer Churn dataset.

- 7,043 customer records
- Customer demographic information
- Service subscription information
- Contract and billing information
- Historical churn labels

### Target Variable

`Churn`

- `Yes` — Customer churned
- `No` — Customer did not churn

## Machine Learning

The following classification models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

### Logistic Regression Performance

| Metric | Score |
|---|---:|
| Accuracy | 80.70% |
| Precision | 66.04% |
| Recall | 56.15% |
| F1 Score | 60.69% |
| ROC-AUC | 84.22% |

Logistic Regression was used for the prediction API and batch prediction pipeline.

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to understand which features contribute to individual churn predictions.

SHAP values explain model behavior and should not be interpreted as causal effects.

## FastAPI

The project provides REST APIs for customer churn prediction and model explanation.

### Endpoints

**Health Check**

`GET /`

**Churn Prediction**

`POST /predict`

Returns:

- Churn probability
- Churn prediction
- Risk level
- Retention recommendation

**Prediction Explanation**

`POST /explain`

Returns:

- Churn probability
- Top contributing features
- SHAP values
- Feature impact

Interactive API documentation is available through FastAPI Swagger UI.

## Batch Prediction

The batch prediction pipeline scores all 7,043 customers using the trained Logistic Regression model.

Output:

`data/processed/churn_predictions.csv`

The output contains:

- `customer_id`
- `actual_churn`
- `churn_probability`
- `predicted_churn`
- `risk_level`

## Power BI Dashboard

An interactive Power BI dashboard was developed to visualize customer churn and machine learning risk insights.

### Dashboard Includes

- Total Customers
- Churn Rate
- Churned Customers
- Average Monthly Charges
- High Risk Customers
- Churn Rate by Contract Type
- Churn Rate by Payment Method
- Churn Rate by Internet Service
- Churn Rate by Customer Tenure
- Average Monthly Charges by Churn Status
- Customer Churn Distribution
- Predicted Customer Risk Distribution
- Contract and Internet Service filters

The dashboard combines historical customer data with machine-learning-based churn risk predictions.

## Technologies Used

### Programming
- Python
- SQL

### Data Science & Machine Learning
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP

### Database
- PostgreSQL
- SQLAlchemy

### API
- FastAPI
- Uvicorn
- Pydantic

### Visualization
- Power BI
- Matplotlib
- Seaborn

### Tools
- Jupyter Notebook
- VS Code
- Git
- GitHub

## Project Structure

```text
customer-churn-intelligence/
├── api/
│   └── main.py
├── dashboard/
│   └── customer_churn_dashboard.pbix
├── data/
│   ├── processed/
│   │   └── churn_predictions.csv
│   └── raw/
├── models/
│   ├── churn_logistic_model.joblib
│   ├── feature_names.joblib
│   └── preprocessor.joblib
├── notebooks/
├── src/
│   ├── batch_predict.py
│   └── database.py
├── tests/
├── train.py
├── requirements.txt
├── README.md
└── .gitignore

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Faizanaziz1/customer-churn-intelligence.git
cd customer-churn-intelligence

2. Create Virtual Environment
python -m venv .venv

3. Activate Virtual Environment


.venv\Scripts\activate.bat

4. Install Dependencies
pip install -r requirements.txt

5. Configure PostgreSQL

Create a PostgreSQL database named:

churn_db

Create a .env file in the project root:

DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/churn_db

6. Train the Model
python train.py

7. Generate Batch Predictions
python src/batch_predict.py

8. Run the FastAPI Application
uvicorn api.main:app --reload

Open:

http://127.0.0.1:8000/docs

to access the interactive Swagger documentation.

## Key Skills Demonstrated

- Data Cleaning and Exploratory Data Analysis
- Feature Engineering
- Classification Modeling
- Model Evaluation
- Class Imbalance Handling
- Explainable AI with SHAP
- PostgreSQL Database Integration
- REST API Development
- Batch Machine Learning Inference
- Power BI Dashboard Development
- Git and GitHub Version Control

## Author

**Faizan Aziz**

Data Science / Machine Learning

## Dashboard Preview

![Customer Churn Intelligence Dashboard](dashboard/powerbi_dashboard.png)