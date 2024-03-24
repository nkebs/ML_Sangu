import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

np.random.seed(0)
data = np.random.rand(200, 4)
np.savetxt("network_data.csv", data, delimiter=",")

data = pd.read_csv("network_data.csv", header=None)
correlation_matrix = data.corr(method='pearson')

feature_names = [f'Name_{i+1}' for i in range(4)]
correlation_matrix.columns = feature_names
correlation_matrix.index = feature_names

correlation_matrix.to_csv("correlation_matrix.csv")

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')

for i in range(len(correlation_matrix.index)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, round(correlation_matrix.iloc[i, j], 2),
                 ha='center', va='center', color='black')

plt.colorbar()
plt.title('Correlation Matrix')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
plt.savefig("correlation_matrix.pdf")

# Find the highest correlation
max_corr = -1
highest_corr_features = None

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > max_corr:
            max_corr = abs(correlation_matrix.iloc[i, j])
            highest_corr_features = (correlation_matrix.columns[i], correlation_matrix.columns[j])

# Create a PDF file and write the highest correlation features
with open("highest_correlation.pdf", "wb") as file:
    c = canvas.Canvas(file, pagesize=letter)
    c.drawString(100, 750, f"The two features with the highest correlation ({max_corr}):")
    c.drawString(100, 730, f"{highest_corr_features[0]} and {highest_corr_features[1]}")
    c.save()