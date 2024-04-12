import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

# Load data from spam-data.csv
df = pd.read_csv('spam-data.csv')

# Splitting the data into features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting the test set results and calculating accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


# Function to parse email features and return as DataFrame with column names
def parse_email_features(email_text):
    words = email_text.split()
    number_of_words = len(words)
    number_of_links = sum(1 for word in words if word.startswith('http://') or word.startswith('https://'))
    number_of_capitalized_words = sum(1 for word in words if word.isupper())
    spam_words = ['buy', 'free', 'click', 'Join', 'offer', 'exclusive']
    number_of_spam_words = sum(1 for word in words if word.lower() in spam_words)

    return pd.DataFrame([[number_of_words, number_of_links, number_of_capitalized_words, number_of_spam_words]],
                        columns=['Number of Words', 'Number of Links', 'Number of Capitalized Words',
                                 'Number of Spam Words'])


# Read emails from emails.txt and check for spam
with open('emails.txt', 'r') as file:
    email_content = file.read()

# Splitting the emails using the specified delimiter
emails = email_content.split('----------------')
emails = [email.strip() for email in emails if email.strip()]

for email in emails:
    features_df = parse_email_features(email)
    is_spam = model.predict(features_df)[0]
    print(f"Email: {email[:50]}... if long\nIs spam: {'Yes' if is_spam else 'No'}\n")

# Feature importance analysis
feature_importance = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
print("Feature Importances:\n", feature_importance.sort_values(by='Coefficient', ascending=False))
