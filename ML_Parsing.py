import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --- 1. LOAD AND PARSE DATA ---
file_path = 'data/Phishing_validation_emails(1).csv'
try:
    df = pd.read_csv(file_path)
    print("--- File loaded and data parsed ---\n")

    # Remove empty rows
    df = df.dropna()

    # Separate ham and phishing emails
    ham_email = df[df["label"] == 1]
    phishing_email = df[df["label"] == 0]

    # Balance the dataset
    phishing_email = phishing_email.sample(ham_email.shape[0])
    
    # Combine ham and phishing emails into a single dataframe
    Data = pd.concat([ham_email, phishing_email], ignore_index = True)

    # Vectorize the email text
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(Data['Email Text'])
    

    # Prepare the labels (the target variable)
    y = Data["label"]

    # --- 2. SPLIT DATA ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. TRAIN THE LOGISTIC REGRESSION MODEL ---
    model = LogisticRegression()
    print("--- Training the model... ---")
    model.fit(X_train, y_train)
    print("--- Model training complete! ---\n")

    # --- 4. EVALUATE THE MODEL (Optional but good practice) ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Data: {accuracy:.4f}\n")

    # ----------------------------------------------------------------------
    # --- 5. TEST THE MODEL WITH A NEW EMAIL ---
    # ----------------------------------------------------------------------
    print("--- Ready to predict a new email ---")

    # Put the email you want to test inside the triple quotes
    new_email = """
    Hi team, just a reminder that our project meeting will take place tomorrow at 10am in room 301. Please bring your progress reports..
    """

    # Transform the new email using the SAME vectorizer ....
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