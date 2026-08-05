from pathlib import Path

import pandas as pd

# ----------------------------

# Load Dataset

# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

fake = pd.read_csv(DATA_DIR / "Fake.csv")

true = pd.read_csv(DATA_DIR / "True.csv")

# ----------------------------

# Create Labels

# ----------------------------

fake["label"] = 0

true["label"] = 1

# ----------------------------

# Merge Dataset

# ----------------------------

news = pd.concat([fake, true], ignore_index=True)

# ----------------------------

# Shuffle Dataset

# ----------------------------

news = news.sample(frac=1, random_state=42)

# ----------------------------

# Reset Index

# ----------------------------

news.reset_index(drop=True, inplace=True)

print("="*60)

print("FINAL DATASET")

print("="*60)

print(news.head())

print()

print("Shape :", news.shape)

print()

print(news["label"].value_counts())