import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#  1. LOAD AND COMBINE DATASETS 
# List of file paths for all your datasets
file_paths = [
    'data/Phishing_validation_emails(1).csv',  
    'data/Phishing_Email.csv'                       
]

try:
    # Read each file and store it in a list of DataFrames
    list_of_dfs = [pd.read_csv(file) for file in file_paths]
    
    # Combine all DataFrames in the list into one
    df = pd.concat(list_of_dfs, ignore_index=True)
    
    print(f"--- Loaded and combined {len(file_paths)} files successfully ---")
    print(f"Original total emails: {len(df)}")

    # 2. CLEAN AND PREPARE DATA 
    # Remove any rows with missing text or types
    df.dropna(subset=['Email Text', 'Email Type'], inplace=True)
    
    # Remove duplicate emails based on the 'Email Text' column
    df.drop_duplicates(subset=['Email Text'], inplace=True)
    print(f"Total unique emails after cleaning: {len(df)}\n")

    # 3. BALANCE THE DATASET 
    safe_email = df[df["Email Type"] == 'Safe Email']
    phishing_email = df[df["Email Type"] == 'Phishing Email']

    print(f"Imbalanced data detected: {len(safe_email)} safe emails, {len(phishing_email)} phishing emails.")
    
    # Balance the dataset by downsampling the larger class
    if len(safe_email) > len(phishing_email):
        safe_email = safe_email.sample(n=len(phishing_email), random_state=42)
        print("Downsampling 'Safe Email' class to balance the dataset.\n")
    elif len(phishing_email) > len(safe_email):
        phishing_email = phishing_email.sample(n=len(safe_email), random_state=42)
        print("Downsampling 'Phishing Email' class to balance the dataset.\n")
    
    # Combine the balanced sets into a final dataframe for training
    balanced_data = pd.concat([safe_email, phishing_email], ignore_index=True)
    print(f"--- Dataset is now balanced. Total samples for training: {len(balanced_data)} ---\n")

    # 4. PREPARE DATA FOR MODELING 
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(balanced_data['Email Text'])
    
    # 1 = Phishing, 0 = Safe
    y = balanced_data['Email Type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)

    # 5. SPLIT DATA 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. TRAIN THE MODEL 
    model = LogisticRegression(max_iter=1000) # Increased max_iter for larger dataset
    print("--- Training the model... ---")
    model.fit(X_train, y_train)
    print("--- Model training complete! ---\n")

    # 7. EVALUATE THE MODEL
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Data: {accuracy:.4f}\n")

    # 8. TEST THE MODEL WITH A NEW EMAIL 

    print("--- Ready to predict a new email ---")

    new_email = """
    URGENT: Your account has been suspended due to suspicious activity. Click here to verify your details immediately to avoid permanent closure.
    """

    new_email_transformed = vectorizer.transform([new_email])
    prediction = model.predict(new_email_transformed)

    print(f"Email Text: \n{new_email}")
    if prediction[0] == 1:
        print("\nPrediction: This is a Phishing Email ")
    else:
        print("\nPrediction: This is a Safe Email ")

except FileNotFoundError as e:
    print(f"Error: The file was not found. Please check the path: {e.filename}")
except Exception as e:
    print(f"An error occurred: {e}")