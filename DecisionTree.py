import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier  # Changed from LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- 1. LOAD AND PREPARE TRAINING DATA ---
# This section is identical to your original script.
training_files = [
    'data/Phishing_validation_emails(1).csv',
    'data/Phishing_Email.csv'
]

try:
    # Read and combine the training files
    list_of_dfs = [pd.read_csv(file, encoding='latin-1') for file in training_files]
    df_train = pd.concat(list_of_dfs, ignore_index=True)

    print(f"--- Loaded and combined {len(training_files)} files for TRAINING ---")

    # Clean the training data
    df_train.dropna(subset=['Email Text', 'Email Type'], inplace=True)
    df_train.drop_duplicates(subset=['Email Text'], inplace=True)

    # Balance the training data by downsampling the majority class
    safe_email = df_train[df_train["Email Type"] == 'Safe Email']
    phishing_email = df_train[df_train["Email Type"] == 'Phishing Email']

    if len(safe_email) > len(phishing_email):
        safe_email = safe_email.sample(n=len(phishing_email), random_state=42)
    else:
        phishing_email = phishing_email.sample(n=len(safe_email), random_state=42)

    # Final balanced training dataset
    balanced_train_data = pd.concat([safe_email, phishing_email], ignore_index=True)
    print(f"Total balanced samples for training: {len(balanced_train_data)}\n")

    # --- 2. LOAD AND PREPARE DEDICATED TEST DATA ---
    # This section is also identical.
    test_file = 'data/sample with lables.csv'
    df_test = pd.read_csv(test_file, encoding='latin-1')

    # Combine 'subject' and 'body' to match the training data format
    df_test['Email Text'] = df_test['subject'].fillna('') + ' ' + df_test['body'].fillna('')
    df_test.dropna(subset=['Email Text', 'label'], inplace=True)
    print(f"--- Loaded {len(df_test)} samples from '{test_file}' for TESTING ---\n")

    # --- 3. FEATURE EXTRACTION (VECTORIZATION) ---
    # The vectorization process remains the same.
    # Prepare the training data
    X_train = balanced_train_data['Email Text']
    y_train = balanced_train_data['Email Type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)

    # Prepare the test data
    X_test = df_test['Email Text']
    y_test = df_test['label'].apply(lambda x: 1 if x == 'phishing' else 0)

    # Initialize the vectorizer and fit it ONLY on the training data
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    # Use the SAME vectorizer to transform the test data
    X_test_vec = vectorizer.transform(X_test)

    print("--- Text data has been vectorized ---")

    # --- 4. TRAIN THE MODEL ---
    # ** KEY CHANGE **: Swapped LogisticRegression for DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42) # Using random_state for reproducibility
    print("--- Training the Decision Tree model on the full training dataset... ---")
    model.fit(X_train_vec, y_train)
    print("--- Model training complete! ---\n")

    # --- 5. EVALUATE THE MODEL ON THE NEW TEST DATA ---
    # The evaluation process is identical.
    print("--- Evaluating model on the dedicated test file... ---")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on '{test_file}': {accuracy:.4f}\n")

    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print()

except FileNotFoundError as e:
    print(f"Error: A file was not found. Please check the path: {e.filename}")
except Exception as e:
    print(f"An error occurred: {e}")