import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re


# Function to extract features from the email
def extract_features(text):
    words = text.split()
    num_words = len(words)
    num_links = len(re.findall(r'http?://[^\s]+', text))
    num_capitalized_words = sum(1 for word in words if word.isupper())
    spam_words = ['buy', 'free', 'click', 'Join', 'offer', 'exclusive']
    num_spam_words = sum(text.count(word) for word in spam_words)
    return [num_words, num_links, num_capitalized_words, num_spam_words]


# Load data
df = pd.read_csv('spam-data.csv')

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Read the email from text file
with open('email.txt', 'r') as file:
    email_content = file.read()

# Extract features
features = extract_features(email_content)
print(features)
features_df = pd.DataFrame([features], columns=X.columns)

# Predict
prediction = model.predict(features_df)
prediction_proba = model.predict_proba(features_df)

# Print the prediction
print("The email is classified as:", "Spam" if prediction[0] == 1 else "Not Spam")
print(f"Probability [Not Spam, Spam]: {prediction_proba[0]}")
