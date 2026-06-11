## Current Project Status

ArabiQ currently includes three completed AI components:

### Phase 1 — Customer Churn Prediction

A machine learning pipeline was built to predict customer churn using the IBM Telco Customer Churn dataset. The phase includes data cleaning, feature engineering, preprocessing, model training, Optuna tuning, SHAP explainability, and a churn model card.

Best selected model:

- Tuned XGBoost
- Test AUC: 0.8535
- Recall: 0.8209

### Phase 2 — Arabic Sentiment Analysis

An Arabic NLP pipeline was built using the Arabic Sentiment Twitter Corpus. The phase includes dataset preparation, Arabic text preprocessing, TF-IDF baseline modeling, fast model comparison, Logistic Regression tuning, and an Arabic sentiment model card.

Best selected model:

- Tuned TF-IDF + Logistic Regression
- Macro F1 used as the main evaluation metric

### Phase 3 — Arabic/English Banking RAG Assistant

A bilingual retrieval-based banking assistant was built using a synthetic Arabic and English banking policy corpus. The system uses multilingual sentence embeddings, ChromaDB, terminal query testing, and a Streamlit interface.

The assistant supports questions such as:

- What documents are needed to open a savings account?
- How long does it take to get a replacement debit card?
- متى تصل بطاقة الخصم البديلة؟
- هل الموافقة على القرض الشخصي مضمونة؟
# ArabiQ — Arabic Customer Intelligence Platform

ArabiQ is an end-to-end AI and machine learning project developed as part of **Project 2 — Advansys Intelligent Solutions**.

The project combines customer churn prediction, Arabic NLP, retrieval-augmented generation, explainable AI, Streamlit application development, Docker, and Azure deployment.

## Project Goal

The goal of ArabiQ is to build an Arabic customer intelligence platform that can:

1. Predict customer churn using machine learning.
2. Explain churn predictions using SHAP.
3. Analyze Arabic customer sentiment using transformer-based NLP.
4. Answer Arabic and English banking questions using a RAG assistant.
5. Deploy the final assistant as a cloud-ready web application.

## Current Progress

### Phase 1 — Customer Churn Prediction

Completed:

* Exploratory Data Analysis on the IBM Telco Customer Churn dataset.
* Feature engineering pipeline.
* Preprocessing pipeline using scikit-learn.
* Model comparison across multiple ML algorithms.
* XGBoost hyperparameter tuning using Optuna.
* SHAP explainability analysis.
* Professional model card documentation.

## Dataset

Phase 1 uses the IBM Telco Customer Churn dataset.

The raw dataset is not included in this repository because raw data files are intentionally ignored using `.gitignore`.

Expected local path:

```text
data/raw/telco_churn_ibm.csv
```

Dataset summary:

* Rows: 7,043 customers
* Original columns: 33
* No churn customers: 5,174
* Churn customers: 1,869
* Approximate churn rate: 26.5%

Because the dataset is moderately imbalanced, accuracy alone is not enough to evaluate the model. The project uses AUC-ROC, F1 score, precision, and recall.

## Machine Learning Models Trained

The following models were trained and compared:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* Neural Network

## Best Model Performance

### Original XGBoost

| Metric    |  Value |
| --------- | -----: |
| Test AUC  | 0.8536 |
| F1 Score  | 0.6337 |
| Precision | 0.5226 |
| Recall    | 0.8048 |

### Tuned XGBoost

| Metric                    |  Value |
| ------------------------- | -----: |
| Best Cross-Validation AUC | 0.8656 |
| Test AUC                  | 0.8535 |
| F1 Score                  | 0.6240 |
| Precision                 | 0.5033 |
| Recall                    | 0.8209 |

The tuned XGBoost model achieved nearly the same test AUC as the original XGBoost model while improving recall. For a churn retention use case, recall is especially important because it helps identify more customers who are likely to leave.

## Explainability

SHAP was used to explain the churn prediction model.

Generated explainability outputs include:

* SHAP summary bar chart
* SHAP beeswarm chart
* SHAP waterfall chart for a churned customer
* SHAP waterfall chart for a non-churned customer

These outputs help explain both global feature importance and individual customer predictions.

## Feature Engineering

The feature engineering pipeline removes leakage columns, direct identifiers, and sensitive or high-risk demographic/geographic columns.

Removed leakage columns include:

* `Churn Score`
* `CLTV`
* `Churn Reason`
* `Churn Value`

Engineered features include:

* `tenure_segment`
* `charge_per_month`
* `num_services`
* `paperless`
* `contract_risk`
* `high_value_at_risk`

## Project Structure

```text
ArabiQ-Arabic-Customer-Intelligence/
│
├── app/
├── data/
│   ├── raw/
│   ├── processed/
│   └── rag_corpus/
│
├── docs/
├── logs/
├── models/
│   ├── ml/
│   ├── nlp/
│   └── rag/
│
├── notebooks/
│   ├── 01_churn_eda.ipynb
│   └── 02_shap_analysis.ipynb
│
├── reports/
│   ├── charts/
│   ├── model_cards/
│   ├── model_comparison.csv
│   ├── classification_reports.txt
│   ├── xgb_tuned_results.csv
│   └── xgb_tuned_best_params.json
│
├── scripts/
│   ├── ml/
│   ├── nlp/
│   ├── rag/
│   └── deploy/
│
├── requirements.txt
└── README.md
```

## How to Reproduce Phase 1

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the churn dataset locally at:

```text
data/raw/telco_churn_ibm.csv
```

Run feature engineering:

```bash
python scripts/ml/feature_engineering.py
```

Run preprocessing:

```bash
python scripts/ml/preprocess.py
```

Train models:

```bash
python scripts/ml/train_models.py
```

Tune XGBoost:

```bash
python scripts/ml/tune_xgboost.py
```

Run SHAP analysis:

```text
notebooks/02_shap_analysis.ipynb
```

## Security and GitHub Safety

The following files are intentionally not committed:

* `.env`
* API keys
* passwords
* virtual environments
* raw datasets
* processed datasets
* trained model files
* large vector databases

These files are excluded using `.gitignore`.

## Upcoming Phases

### Phase 2 — Arabic NLP

Fine-tune CAMeLBERT for Arabic sentiment analysis.

### Phase 3 — RAG Assistant

Build an Arabic-English document question-answering assistant using embeddings, ChromaDB, and an LLM.

### Phase 4 — Deployment

Containerize the Streamlit app with Docker and deploy it to Azure.

## Author

**Clara Amir Hakim Zaher**
AI Engineering Intern — Advansys Intelligent Solutions
B.Sc. Computer Science, Data Science Major

