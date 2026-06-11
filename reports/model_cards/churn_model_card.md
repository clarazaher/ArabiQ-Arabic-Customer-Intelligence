# Model Card: Customer Churn Prediction Model

## 1. Model Details

- **Project**: ArabiQ — Arabic Customer Intelligence Platform
- **Model task**: Binary classification
- **Target variable**: Customer churn
- **Selected model**: Tuned XGBoost Classifier
- **Developer**: Clara Amir Hakim Zaher
- **Organization context**: Advansys Intelligent Solutions — Project 2
- **Version**: 1.0
- **Date**: June 2026

## 2. Intended Use

The model predicts whether a customer is likely to churn, meaning whether they may leave the company.

The intended business use is to help customer retention teams identify at-risk customers early and take proactive actions, such as offering support, reviewing pricing concerns, or improving service experience.

This model is intended for:

- customer churn risk prediction,
- customer retention prioritization,
- business analysis of churn drivers,
- internal decision support.

This model should **not** be used for:

- credit approval,
- loan decisions,
- denying service,
- legally sensitive classification,
- fully automated decisions without human review.

## 3. Dataset

The model was trained using the IBM Telco Customer Churn dataset.

The dataset contains:

- **Rows**: 7,043 customers
- **Original columns**: 33
- **Target classes**:
  - No churn: 5,174 customers
  - Churn: 1,869 customers
- **Approximate churn rate**: 26.5%

This means the dataset is moderately imbalanced, so accuracy alone is not a reliable metric. AUC-ROC, F1 score, precision, and recall were used instead.

## 4. Data Preparation

The raw dataset was processed using a feature engineering pipeline.

The pipeline:

1. Removed leakage columns.
2. Removed direct identifiers.
3. Removed demographic and geographic columns for ethical deployment.
4. Converted the target into a binary column.
5. Created new business-friendly features.

### Removed leakage columns

The following columns were removed because they may reveal information about churn that would not be available at prediction time:

- `Churn Score`
- `CLTV`
- `Churn Reason`
- `Churn Value`

### Removed identifier, demographic, and geographic columns

The following columns were excluded to reduce privacy, fairness, and ethical risks:

- `CustomerID`
- `City`
- `Zip Code`
- `Lat Long`
- `Latitude`
- `Longitude`
- `Gender`
- `Senior Citizen`

### Engineered features

The project created several new predictive features, including:

- `tenure_segment`
- `charge_per_month`
- `num_services`
- `paperless`
- `contract_risk`
- `high_value_at_risk`

## 5. Model Training

Several models were trained and compared:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Neural Network

XGBoost was selected because it performed best overall and achieved strong churn recall, which is important for customer retention use cases.

Optuna was used to tune the XGBoost hyperparameters.

## 6. Performance

### Original XGBoost

| Metric | Value |
|---|---:|
| Test AUC | 0.8536 |
| F1 Score | 0.6337 |
| Precision | 0.5226 |
| Recall | 0.8048 |

### Tuned XGBoost

| Metric | Value |
|---|---:|
| Best Cross-Validation AUC | 0.8656 |
| Test AUC | 0.8535 |
| F1 Score | 0.6240 |
| Precision | 0.5033 |
| Recall | 0.8209 |

## 7. Model Selection Reasoning

The tuned XGBoost model achieved nearly the same test AUC as the original XGBoost model, while improving recall from 0.8048 to 0.8209.

For a churn retention use case, recall is especially important because missing a customer who is likely to churn can be costly. A higher recall means the model catches more actual churn customers, even if it also creates more false positives.

Therefore, the tuned XGBoost model is a strong candidate for business use when the goal is proactive retention outreach.

## 8. Explainability

SHAP was used to explain the model predictions.

SHAP helps answer two important questions:

1. Which features matter most overall?
2. Why did the model predict churn for a specific customer?

The project generated:

- SHAP summary bar chart
- SHAP beeswarm chart
- SHAP waterfall chart for a churned customer
- SHAP waterfall chart for a non-churned customer

These explainability outputs make the model more transparent and easier to discuss with business stakeholders.

## 9. Key Business Insights

Based on the EDA and SHAP analysis, the main churn-related factors are expected to include:

1. Contract type
2. Customer tenure
3. Monthly charges
4. Number of active services
5. Paperless billing and payment behavior

Customers on short-term contracts with high monthly charges and fewer services are likely to represent a higher churn-risk group.

## 10. Limitations

This model has several limitations:

- The dataset is fictional and based on a telecom use case.
- The dataset may not reflect current customer behavior.
- External factors such as competitor offers, economic changes, customer complaints, and social media sentiment are not included.
- The model predicts churn risk, but it does not prove why a customer will churn.
- The model should be validated on real company data before production use.

## 11. Ethical Considerations

This model should be used as a decision-support tool, not as a fully automated decision-maker.

Important ethical considerations:

- The model should not be used to deny services to customers.
- The model should not be used for credit or legal decisions.
- Human review should be included before taking business action.
- Sensitive demographic and geographic columns were excluded from training.
- The model should be monitored for unfair patterns before deployment.

## 12. Reproducibility

To reproduce the Phase 1 churn model pipeline:

1. Place the raw dataset at:

   `data/raw/telco_churn_ibm.csv`

2. Run feature engineering:

   `python scripts/ml/feature_engineering.py`

3. Run preprocessing:

   `python scripts/ml/preprocess.py`

4. Train models:

   `python scripts/ml/train_models.py`

5. Tune XGBoost:

   `python scripts/ml/tune_xgboost.py`

6. Run SHAP analysis:

   `notebooks/02_shap_analysis.ipynb`

Raw data, processed data, and trained model files are intentionally excluded from GitHub using `.gitignore`.
