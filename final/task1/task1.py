import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np

# Load CIFAR-10 dataset
(train_images, train_labels), (_, _) = cifar10.load_data()

# Filter out normal (automobile) and malware images
normal_images = train_images[train_labels.flatten() == 1]
malware_images = train_images[train_labels.flatten() != 1]

# Label normal images as 0 and malware images as 1
normal_labels = np.zeros(len(normal_images))
malware_labels = np.ones(len(malware_images))

# Concatenate normal and malware images and labels
X = np.concatenate([normal_images, malware_images])
y = np.concatenate([normal_labels, malware_labels])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape and normalize the data
X_train = X_train.reshape((-1, 32 * 32 * 3)) / 255.0
X_test = X_test.reshape((-1, 32 * 32 * 3)) / 255.0

# Model Architecture
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(32 * 32 * 3,)),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
