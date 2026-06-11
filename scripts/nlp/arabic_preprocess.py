"""
Arabic text preprocessing for ArabiQ sentiment analysis.

This script:
1. Loads the prepared Arabic sentiment dataset.
2. Cleans Arabic tweet text.
3. Creates a binary label column.
4. Saves a cleaned local dataset to data/processed/arabic_sentiment_clean.csv.

Important:
- The processed CSV is ignored by Git because data/processed/ is in .gitignore.
- The code file itself is safe to commit.
"""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/arabic_sentiment/data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data/processed/arabic_sentiment_clean.csv"


def clean_arabic_text(text: str) -> str:
    """
    Clean and normalize Arabic text for sentiment analysis.

    Parameters
    ----------
    text : str
        Raw tweet text.

    Returns
    -------
    str
        Cleaned Arabic text.
    """

    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove user mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag text but remove the # symbol
    text = text.replace("#", " ")

    # Remove Arabic diacritics/tashkeel
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize Alef forms to ا
    text = re.sub(r"[إأآا]", "ا", text)

    # Normalize Yeh forms
    text = re.sub(r"[يى]", "ي", text)

    # Normalize Teh Marbuta to Heh
    text = re.sub(r"ة", "ه", text)

    # Remove tatweel/kashida
    text = re.sub(r"ـ", "", text)

    # Remove emojis, English letters, numbers, punctuation, and symbols
    # Keep Arabic letters and spaces only
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)

    # Remove extra repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main() -> None:
    """Run Arabic preprocessing pipeline."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw Arabic sentiment file not found at {RAW_DATA_PATH}. "
            "Run scripts/nlp/prepare_arabic_dataset.py first."
        )

    print("Loading Arabic sentiment dataset from:")
    print(RAW_DATA_PATH)

    df = pd.read_csv(RAW_DATA_PATH)
    print("Raw dataset shape:", df.shape)

    required_columns = {"text", "sentiment"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["clean_text"] = df["text"].apply(clean_arabic_text)

    # Remove empty cleaned texts
    before = len(df)
    df = df[df["clean_text"] != ""].copy()
    after = len(df)

    print(f"Removed empty cleaned rows: {before - after}")

    # Encode labels: positive = 1, negative = 0
    sentiment_map = {
        "negative": 0,
        "positive": 1,
    }

    df["label"] = df["sentiment"].map(sentiment_map)

    if df["label"].isna().any():
        bad_values = df.loc[df["label"].isna(), "sentiment"].unique()
        raise ValueError(f"Unexpected sentiment labels found: {bad_values}")

    df["label"] = df["label"].astype(int)

    output_columns = ["text", "clean_text", "sentiment", "label", "source_file"]
    output_columns = [col for col in output_columns if col in df.columns]

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8")

    print("\nArabic preprocessing completed successfully.")
    print("Processed dataset shape:", df[output_columns].shape)
    print("Saved to:", PROCESSED_DATA_PATH)

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nCleaning examples:")
    examples = df[["text", "clean_text", "sentiment", "label"]].head(5)
    print(examples.to_string(index=False))


if __name__ == "__main__":
    main()
