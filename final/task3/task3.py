import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate synthetic network traffic data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Create DataFrame
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
data['target'] = y

# Display the first few rows of the dataset
print(data.head())

# Calculate correlation matrix
correlation_matrix = data.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Network Traffic Features')
plt.show()