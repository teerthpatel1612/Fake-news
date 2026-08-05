from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

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

# ====================================================
# TEXT CLEANING
# ====================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = " ".join(text.split())

    # Remove stopwords
    words = []

    for word in text.split():
        if word not in ENGLISH_STOP_WORDS:
            words.append(word)

    return " ".join(words)


print("\nCleaning Text...")

news["clean_text"] = news["text"].apply(clean_text)
print("\nConverting text into vectors...")

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(news["clean_text"])

y = news["label"]

print("Done!")

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nLabels Shape:")
print(y.shape)

print("Done!")

print("=" * 60)
print("CLEANED ARTICLE")
print("=" * 60)

print(news["clean_text"].iloc[0][:1000])

# ==================================================
# Train/Test Split
# ==================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Done!")

# ==================================================
# Train Model
# ==================================================

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Training Complete!")

# ==================================================
# Prediction
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# Evaluation
# ==================================================

print("\nAccuracy:")

accuracy = accuracy_score(y_test, y_pred)

print(f"{accuracy*100:.2f}%")

print("\nClassification Report")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, y_pred))

# ==================================================
# Test Your Own News
# ==================================================

print("\n" + "=" * 60)
print("TEST YOUR OWN NEWS")
print("=" * 60)

user_news = input("\nPaste a news article:\n\n")

cleaned = clean_text(user_news)

vector = tfidf.transform([cleaned])

prediction = model.predict(vector)

if prediction[0] == 0:
    print("\nPrediction: FAKE NEWS")
else:
    print("\nPrediction: REAL NEWS")