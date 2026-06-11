from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_RESULTS = PROJECT_ROOT / "reports/arabic_sentiment_baseline_results.csv"
FAST_RESULTS = PROJECT_ROOT / "reports/arabic_sentiment_fast_model_comparison.csv"
TUNED_RESULTS = PROJECT_ROOT / "reports/arabic_sentiment_tuned_logreg_results.csv"
TUNED_REPORT = PROJECT_ROOT / "reports/arabic_sentiment_tuned_logreg_classification_report.txt"

OUTPUT_PATH = PROJECT_ROOT / "reports/model_cards/arabic_sentiment_model_card.md"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(BASELINE_RESULTS)
    fast_df = pd.read_csv(FAST_RESULTS)
    tuned_df = pd.read_csv(TUNED_RESULTS)

    baseline = baseline_df.iloc[0]
    best_fast = fast_df.sort_values(by="Macro F1", ascending=False).iloc[0]
    best_tuned = tuned_df.sort_values(by="Macro F1", ascending=False).iloc[0]

    tuned_report_text = ""
    if TUNED_REPORT.exists():
        tuned_report_text = TUNED_REPORT.read_text(encoding="utf-8")

    content = f"""# Arabic Sentiment Analysis Model Card

## 1. Model Overview

This phase belongs to **ArabiQ — Arabic Customer Intelligence Platform**.

The goal is to classify Arabic social media text into two sentiment classes:

- Negative
- Positive

This phase uses Arabic text preprocessing, TF-IDF feature extraction, fast model comparison, and Logistic Regression tuning.

## 2. Dataset

The model uses the Arabic Sentiment Twitter Corpus.

After preprocessing, the dataset contains around 35k Arabic tweets.

The preprocessing pipeline handles:

- URLs
- User mentions
- Hashtag symbols
- Emojis and non-Arabic symbols
- Arabic diacritics
- Different Alef and Yeh forms
- Extra whitespace

## 3. Baseline Model

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|
| {baseline["Model"]} | {baseline["Accuracy"]} | {baseline["Macro F1"]} | {baseline["Macro Precision"]} | {baseline["Macro Recall"]} |

## 4. Fast Model Comparison

The following models were compared:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Complement Naive Bayes

Best fast model:

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|
| {best_fast["Model"]} | {best_fast["Accuracy"]} | {best_fast["Macro F1"]} | {best_fast["Macro Precision"]} | {best_fast["Macro Recall"]} |

## 5. Tuned Model

Best tuned configuration:

| Model | TF-IDF Config | C | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---|---:|---:|---:|---:|---:|
| {best_tuned["Model"]} | {best_tuned["TF-IDF Config"]} | {best_tuned["C"]} | {best_tuned["Accuracy"]} | {best_tuned["Macro F1"]} | {best_tuned["Macro Precision"]} | {best_tuned["Macro Recall"]} |

## 6. Selected Model

The selected model is the tuned TF-IDF + Logistic Regression model because it achieved the best Macro F1 score among the tested fast Arabic sentiment models.

Macro F1 was used because we care about balanced performance across both positive and negative sentiment classes.

## 7. Business Use

This model can support Arabic customer intelligence tasks such as:

- Detecting negative customer feedback
- Monitoring Arabic customer satisfaction
- Prioritizing customer service responses
- Summarizing public sentiment about products or services

## 8. Limitations

This model has some important limitations:

- It only predicts positive or negative sentiment.
- It does not detect neutral sentiment.
- It may struggle with sarcasm, dialect-heavy Arabic, and mixed Arabic-English text.
- It was trained on Twitter-style text, so performance may differ on formal reviews, emails, or banking documents.
- It is not as context-aware as transformer models such as CAMeLBERT.

## 9. Ethical Considerations

This model should support human decision-making, not replace it. Sentiment predictions can be wrong, especially for sarcasm or dialects, so human review is recommended for important customer decisions.

## 10. Reproducibility

Run these scripts in order:

1. python scripts/nlp/prepare_arabic_dataset.py
2. python scripts/nlp/arabic_preprocess.py
3. python scripts/nlp/baseline_tfidf.py
4. python scripts/nlp/compare_fast_sentiment_models.py
5. python scripts/nlp/tune_tfidf_logreg.py
6. python scripts/nlp/create_sentiment_model_card.py

## 11. Tuned Classification Report

{tuned_report_text}
"""

    OUTPUT_PATH.write_text(content, encoding="utf-8")

    print("Arabic sentiment model card created successfully.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
