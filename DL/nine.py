import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -----------------------------
# AlexNet Model (Sequential)
# -----------------------------
print("\nBuilding AlexNet Model...")
alexnet_model = Sequential([
    Conv2D(96, (11, 11), strides=4, activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D((3, 3), strides=2),
    
    Conv2D(256, (5, 5), activation='relu', padding='same'),
    MaxPooling2D((3, 3), strides=2),
    
    Conv2D(384, (3, 3), activation='relu', padding='same'),
    Conv2D(384, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    
    Flatten()
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

alexnet_model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

print("\nTraining AlexNet...")
alexnet_model.fit(x_train, y_train,
                  epochs=10,
                  batch_size=64,
                  validation_data=(x_test, y_test),
                  verbose=2)

train_loss, train_acc = alexnet_model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = alexnet_model.evaluate(x_test, y_test, verbose=0)
print(f"\nAlexNet Training Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
print(f"AlexNet Testing Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# -----------------------------
# VGG16 Model (Sequential with Transfer Learning)
# -----------------------------
print("\nLoading Pretrained VGG16 Model...")
vgg_base = VGG16(weights='imagenet',
                include_top=False,
                input_shape=(32, 32, 3))

vgg_base.trainable = False

vgg16_model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

vgg16_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

print("\nTraining VGG16 Model...")
vgg16_model.fit(x_train, y_train,
                epochs=10,
                batch_size=64,
                validation_split=0.2,
                verbose=2)

train_loss, train_acc = vgg16_model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = vgg16_model.evaluate(x_test, y_test, verbose=0)

print(f"\nVGG16 Training Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
print(f"VGG16 Testing Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
