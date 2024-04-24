import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("Darknet.csv")

# Drop source and destination IP addresses
df = df.drop(["Src IP", "Dst IP", "Flow ID", "Timestamp", "Label2"], axis=1)

# Remove rows with missing values
df = df.dropna()

# Remove rows with infinity or extremely large values
df = df[~df.isin([float('inf'), float('-inf')]).any(axis=1)]
df = df[(df.select_dtypes(include=['number']) < 1e15).all(axis=1)]  # Remove rows with values larger than 1e15

# Encode the target variable
label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

# Split the dataset into features and target variable
X = df.drop("Label1", axis=1)
y = df["Label1"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Write the accuracy to a file
with open("accuracy2.txt", "w") as f:
    f.write("Model Accuracy: {:.2f}%".format(accuracy * 100))
