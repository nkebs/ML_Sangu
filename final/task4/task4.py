import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load CIFAR-10 dataset
(train_images, train_labels), (_, _) = cifar10.load_data()

# Filter out only the images labeled as cars (label 1 in CIFAR-10)
car_images = train_images[train_labels.flatten() == 1]

# You should replace these paths with the actual paths to your gun images
gun_paths = ["gun1.jpeg,"
             "gun2.jpeg,"
             "gun3.jpeg,"
             "gun4.jpeg"]

gun_images = []
for path in gun_paths:
    img = tf.keras.preprocessing.image.load_img(path, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    gun_images.append(img_array)

# Labels: 0 for cars, 1 for guns
car_labels = np.zeros(len(car_images))
gun_labels = np.ones(len(gun_images))

# Concatenate car and gun images and labels
X = np.concatenate([car_images, gun_images])
y = np.concatenate([car_labels, gun_labels])

# Shuffle the data
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# Split the data into training and testing sets
split = int(0.8 * len(X))
train_images, test_images = X[:split], X[split:]
train_labels, test_labels = y[:split], y[split:]

# Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax') # Output layer with 2 units for car and gun
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(train_images, train_labels, epochs=10)

# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Prediction
predictions = model.predict(test_images)