import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 1. LOAD DATA ---
file_path = 'data/Phishing_validation_emails(1).csv'
try:
    df = pd.read_csv(file_path)
    print("--- File loaded successfully ---\n")

    # --- 2. CLEAN AND BALANCE DATA (from Snippet 1) ---
    # Remove any rows with missing text or types
    df = df.dropna(subset=['Email Text', 'Email Type'])

    # Separate safe and phishing emails for balancing
    safe_email = df[df["Email Type"] == 'Safe Email']
    phishing_email = df[df["Email Type"] == 'Phishing Email']

    # Balance the dataset by downsampling the larger class to match the smaller one
    if len(safe_email) > len(phishing_email):
        print(f"Imbalanced data detected: {len(safe_email)} safe emails, {len(phishing_email)} phishing emails.")
        safe_email = safe_email.sample(n=len(phishing_email), random_state=42)
        print("Downsampling 'Safe Email' class to balance the dataset.\n")
    elif len(phishing_email) > len(safe_email):
        print(f"Imbalanced data detected: {len(phishing_email)} phishing emails, {len(safe_email)} safe emails.")
        phishing_email = phishing_email.sample(n=len(safe_email), random_state=42)
        print("Downsampling 'Phishing Email' class to balance the dataset.\n")
    
    # Combine the balanced sets into a single dataframe
    balanced_data = pd.concat([safe_email, phishing_email], ignore_index=True)
    print(f"--- Dataset is now balanced. Total samples: {len(balanced_data)} ---\n")

    # --- 3. PREPARE DATA FOR MODELING ---
    # Vectorize the email text from the BALANCED data
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(balanced_data['Email Text'])
    
    # Prepare the labels using the consistent logic (from Snippet 2)
    # 1 = Phishing, 0 = Safe
    y = balanced_data['Email Type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)

    # --- 4. SPLIT DATA ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 5. TRAIN THE LOGISTIC REGRESSION MODEL ---
    model = LogisticRegression()
    print("--- Training the model... ---")
    model.fit(X_train, y_train)
    print("--- Model training complete! ---\n")

    # --- 6. EVALUATE THE MODEL ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Data: {accuracy:.4f}\n")

    # ----------------------------------------------------------------------
    # --- 7. TEST THE MODEL WITH A NEW EMAIL ---
    # ----------------------------------------------------------------------
    print("--- Ready to predict a new email ---")

    # Put the email you want to test inside the triple quotes
    new_email = """
    your package is bein processed and canot be delieverd due to the incorect infomation. please urgently respond to this Email andupdate your infomation within the next 24 hours to confirm delivery..
    """

    # Transform the new email using the SAME vectorizer
    new_email_transformed = vectorizer.transform([new_email])

    # Make a prediction
    prediction = model.predict(new_email_transformed)

    # Print the result in a user-friendly way
    print(f"Email Text: \n{new_email}")
    if prediction[0] == 1:
        print("\nPrediction: This is a Phishing Email 🎣")
    else:
        print("\nPrediction: This is a Safe Email ✅")

except FileNotFoundError:
    print(f"Error: The file was not found at the path: '{file_path}'")
except Exception as e:
    print(f"An error occurred: {e}")