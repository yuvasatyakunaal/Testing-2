import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, datasets
import matplotlib.pyplot as plt

# 1. Load and prepare data
(x_train, _), (x_test, _) = datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.
x_test = x_test.reshape(-1, 784).astype('float32') / 255.

# 2. Add noise to test images
noisy_test = x_test + 0.5 * np.random.normal(size=x_test.shape)
noisy_test = np.clip(noisy_test, 0, 1)

# 3. Simple autoencoder model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse')

# 4. Train (using noisy training data)
noisy_train = x_train + 0.5 * np.random.normal(size=x_train.shape)
noisy_train = np.clip(noisy_train, 0, 1)
model.fit(noisy_train, x_train, epochs=5, batch_size=256)

# 5. Get denoised images
denoised = model.predict(noisy_test)

# 6. Show samples
plt.figure(figsize=(10, 4))
for i in range(5):  # show first 5 samples
    # Original
    plt.subplot(3, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Noisy
    plt.subplot(3, 5, i+6)
    plt.imshow(noisy_test[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    # Denoised
    plt.subplot(3, 5, i+11)
    plt.imshow(denoised[i].reshape(28, 28), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

plt.tight_layout()
plt.show()