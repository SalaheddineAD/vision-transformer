

import tensorflow as tf
import numpy as np
import cv2
import random
from tensorflow.keras import layers
from tensorflow_addons.metrics import F1Score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.preprocessing.image import img_to_array

import tensorflow as tf
import numpy as np
import cv2

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess images
new_shape = (32, 32)
train_images = np.array([cv2.resize(img, new_shape) for img in train_images])
test_images = np.array([cv2.resize(img, new_shape) for img in test_images])

train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images[..., tf.newaxis].astype("float32")
test_images = test_images[..., tf.newaxis].astype("float32")

# Build the ViT model
input_shape = (*new_shape, 1)
patch_size = 4
num_heads = 2
num_patches = (new_shape[0] // patch_size) * (new_shape[1] // patch_size)
projection_dim = 32
num_transformer_layers = 2
mlp_head_units = [64, 32]

inputs = tf.keras.layers.Input(shape=input_shape)
x = tf.keras.layers.Conv2D(3, (3, 3), padding="same")(inputs)  # Convert to 3 channels
x = tf.keras.layers.Rescaling(1.0 / 255)(x)  # Rescale
x = tf.keras.layers.Conv2D(32, (patch_size, patch_size), strides=patch_size)(x)
x = tf.keras.layers.Reshape((num_patches, 32))(x)

for _ in range(num_transformer_layers):
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    x2 = tf.keras.layers.Add()([attention_output, x])
    x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = tf.keras.layers.Conv1D(64, 2, padding="same", activation="relu")(x3)
    x3 = tf.keras.layers.Dropout(0.1)(x3)
    x3 = tf.keras.layers.Conv1D(32, 2, padding="same")(x3)
    x = tf.keras.layers.Add()([x3, x2])

x = tf.keras.layers.GlobalAveragePooling1D()(x)
for dim in mlp_head_units:
    x = tf.keras.layers.Dense(dim, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training the model
model.fit(train_images, train_labels, epochs=1, batch_size=32)

# Evaluate and report as before...

# Applying transformations to test images
# Applying transformations to test images
# Applying transformations to test images
# Applying transformations to test images
transformed_test_images = []
for img in test_images:
    transformed_img = random_transform(img)
    transformed_test_images.append(transformed_img)

# Checking for shape consistency
first_shape = transformed_test_images[0].shape
for i, img in enumerate(transformed_test_images):
    if img.shape != first_shape:
        print(f"Inconsistent shape at index {i}: expected {first_shape}, but got {img.shape}")

# Add more diagnostic code if needed

# Evaluating the model
test_loss, test_acc = model.evaluate(transformed_test_images, test_labels)

# Getting Precision, Recall, and F1 Score
y_pred = model.predict(transformed_test_images)
y_pred_labels = np.argmax(y_pred, axis=1)

precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, y_pred_labels, average='macro')

print(f"Test Accuracy: {test_acc}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

