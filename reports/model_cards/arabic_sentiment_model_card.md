# Arabic Sentiment Analysis Model Card

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
| TF-IDF + Logistic Regression | 0.6821 | 0.681 | 0.6822 | 0.681 |

## 4. Fast Model Comparison

The following models were compared:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Complement Naive Bayes

Best fast model:

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.6821 | 0.681 | 0.6822 | 0.681 |

## 5. Tuned Model

Best tuned configuration:

| Model | TF-IDF Config | C | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF + Tuned Logistic Regression | char_wb_3_5_80k | 1.0 | 0.6821 | 0.6808 | 0.6825 | 0.6808 |

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

Best TF-IDF + Logistic Regression configuration
================================================================================
{'Model': 'TF-IDF + Tuned Logistic Regression', 'TF-IDF Config': 'char_wb_3_5_80k', 'Analyzer': 'char_wb', 'Ngram Range': '(3, 5)', 'Max Features': 80000, 'C': 1.0, 'Accuracy': 0.6821, 'Macro F1': 0.6808, 'Macro Precision': 0.6825, 'Macro Recall': 0.6808, 'Vectorization Time Seconds': 1.56, 'Train Time Seconds': 0.27}

Classification report:
              precision    recall  f1-score   support

    negative       0.68      0.73      0.70      3677
    positive       0.69      0.64      0.66      3470

    accuracy                           0.68      7147
   macro avg       0.68      0.68      0.68      7147
weighted avg       0.68      0.68      0.68      7147

