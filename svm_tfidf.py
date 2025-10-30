import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# Load csvs
df = pd.read_csv("data/Ling.csv")
test_df = pd.read_csv("data/sample.csv", encoding="cp1252")


import nltk
import string
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Clean data
def clean_text(text):
  #Lowercase
  text = text.lower()

  # Remove punctuation
  text = text.translate(str.maketrans('', '', string.punctuation))

  # Tokenize
  words = text.split()

  # Remove stopwords
  words = [word for word in words if word not in stop_words]

  # Join back to string
  return ' '.join(words)

# Clean body text
df["body"] = df["body"].apply(clean_text)
test_df["body"] = test_df["body"].apply(clean_text)

# Label mapping test data
label_mapping = {"legitimate": 0, "phishing": 1}

# Apply mapping to test labels
test_df["label"] = test_df["label"].replace(label_mapping)

# Save cleaned data to CSV
df.to_csv("cleaned_train_data.csv", index=False)
test_df.to_csv("cleaned_test_data.csv", index=False)


from sklearn.utils import resample

# label dataframe labels
legitimate_df = df[df["label"] == 0]
phishing_df = df[df["label"] == 1]

# Undersample legitimate class
legitimate_undersample_df = resample(legitimate_df,
                                     replace=False,
                                     n_samples=len(phishing_df),
                                     random_state=42)

# Concatenate the resampled legitimate and original phishing
df_balanced = pd.concat([legitimate_undersample_df, phishing_df])

# Shuffle the data
df = df_balanced.sample(frac=1, random_state=42)

# Check the distribution of classes in the balanced dataset
print(df["label"].value_counts())

# Training dataset
X_train = df["body"]
y_train = df["label"]

# Test dataset
X_test = test_df["body"]
y_test = test_df["label"]

from sklearn.model_selection import GridSearchCV

# Build SVM Model
SVM = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2))),("SVM", LinearSVC(class_weight="balanced"))])

# Set up hyperparameter grid to search
param_grid = {
    "tfidf__max_df": [0.85],
    "tfidf__min_df": [3],
    "tfidf__max_features": [5000],
    "SVM__C": [0.1, 1]
}

# Setup GridSearchCV
grid_search = GridSearchCV(SVM, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit model
grid_search.fit(X_train, y_train)


print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# y_prediction for SVM model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))

# Check the SVM model metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision:  {precision:.3f}")
print(f"Recall:  {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")