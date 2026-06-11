"""
Prepare the Arabic Sentiment Twitter Corpus for Phase 2.

This script:
1. Reads the raw positive/negative TSV files.
2. Adds sentiment labels based on the filename.
3. Combines everything into one dataset.
4. Saves the result locally as data/raw/arabic_sentiment/data.csv.

Note:
The output CSV is inside data/raw/, so it is ignored by GitHub.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw/arabic_sentiment"
OUTPUT_PATH = RAW_DIR / "data.csv"


def read_tsv_file(path: Path, sentiment: str) -> pd.DataFrame:
    """Read a TSV file and attach a sentiment label."""

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    df["sentiment"] = sentiment
    df["source_file"] = path.name

    return df


def main() -> None:
    """Combine all Arabic sentiment TSV files into one CSV."""

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw Arabic sentiment folder not found: {RAW_DIR}")

    tsv_files = sorted(RAW_DIR.glob("*.tsv"))

    if not tsv_files:
        raise FileNotFoundError(
            f"No TSV files found in {RAW_DIR}. "
            "Make sure the Arabic sentiment dataset was copied correctly."
        )

    print("Found TSV files:")
    for file in tsv_files:
        print("-", file.name)

    datasets = []

    for file in tsv_files:
        filename_lower = file.name.lower()

        if "positive" in filename_lower:
            sentiment = "positive"
        elif "negative" in filename_lower:
            sentiment = "negative"
        else:
            print(f"Skipping file because sentiment is unclear: {file.name}")
            continue

        df_part = read_tsv_file(file, sentiment)
        datasets.append(df_part)

    if not datasets:
        raise ValueError("No valid positive/negative TSV files were loaded.")

    df = pd.concat(datasets, ignore_index=True)

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nArabic sentiment dataset prepared successfully.")
    print("Saved to:", OUTPUT_PATH)
    print("Shape:", df.shape)

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
