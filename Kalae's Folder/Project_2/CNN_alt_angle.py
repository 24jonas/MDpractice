# === Imports ===
import keras
import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical      

# Constants
img_size = 128
num_classes = 360
num_samples = 1000
batch_size = 32
epochs = 130

angle_resolution = 10  # degrees per class
# Classes
class ForwardPass:
    def __init__ (self, W1, b1, W2, b2, W3, b3, input_data):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3
        self.input_data = input_data

    def forward(self):
        Z1 = np.dot(self.W1, self.input_data) + self.b1
        A1 = np.maximum(Z1,0)  # ReLU activation
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = np.maximum(Z2, 0)  # ReLU activation
        Z3 = np.dot(self.W3, A2) + self.b3
        A3 = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)
        return Z1, A1, Z2, A2, Z3, A3

class BackwardPass:
    def __init__(self, W1, b1, W2, b2, W3, b3, input_data, Y):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3
        self.input_data = input_data
        self.Y = Y

    def backward(self, Z1, A1, Z2, A2, Z3, A3):
        m = self.input_data.shape[1]
        one_hot_Y = np.zeros((self.Y.size, np.max(self.Y) + 1))
        one_hot_Y[np.arange(self.Y.size), self.Y] = 1
        one_hot_Y = one_hot_Y.T

        dZ3 = A3 - one_hot_Y
        dW3 = 1 / m * np.dot(dZ3, A2.T)
        db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

        dZ2 = np.dot(self.W3.T, dZ3) * (Z2 > 0)
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * (Z1 > 0)
        dW1 = 1 / m * np.dot(dZ1, self.input_data.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3



#Step 1: Load & Rotate Dataset 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("/Users/kalaeabrams/Desktop/Images", transform=transform)

for i in range(5):
    image, _ = dataset[i]
    angle = np.random.randint(0, 360)
    rotated = TF.rotate(image, angle)
    plt.imshow(rotated.squeeze(), cmap="gray")
    plt.title(f"Angle: {angle}")
    plt.show()

def build_dataset(dataset, num_samples=1000):
    X = []
    Y = []
    for i in range(num_samples):
        image, _ = dataset[i % len(dataset)]
        angle = np.random.randint(1, 360)
        rotated = TF.rotate(image, angle)
        X.append(rotated.squeeze().numpy())  # shape: (128, 128)
        label = angle #// angle_resolution    # <-- Fix here
        Y.append(label)
    X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0  # Normalize
    Y = np.array(Y)
    return X, Y

print("Loading and rotating images...")
X, Y = build_dataset(dataset, num_samples=num_samples)
Y_cat = to_categorical(Y, num_classes=num_classes)

print("X shape:", X.shape)       # (num_samples, 128, 128, 1)
print("Y_cat shape:", Y_cat.shape)  # (num_samples, num_classes)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(img_size, img_size, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),   
    keras.layers.Dense(num_classes, activation='softmax')  # Output layer for 360 classes
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training model...")
model.fit(X, Y_cat, epochs=epochs, batch_size=batch_size, validation_split=0.01)


for i in range(32):
    image = X[i].reshape(1, img_size, img_size, 1)  # Add batch dimension
    prediction = model.predict(image, verbose=0)    # Predict softmax output
    predicted_class = np.argmax(prediction)
    predicted_angle = predicted_class               # Assuming 1 class = 1 degree

    actual_class = np.argmax(Y_cat[i])              # True label as class
    actual_angle = actual_class                     # Same assumption

    print(f"Example {i+1}: Predicted Angle = {predicted_angle}°, Actual Angle = {actual_angle}°")


"""


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Loading and rotating images...")
X, Y = build_dataset(dataset, num_samples=num_samples)
Y_cat = to_categorical(Y, num_classes=num_classes)

print("X shape:", X.shape)  # Should be (num_samples, img_size, img_size, 1)
print("Y_cat shape:", Y_cat.shape)  # Should be (num_samples, num_classes)

print("Training model...")



history = model.fit(X, Y_cat, epochs=epochs, batch_size=batch_size, validation_split=0.01)

# Visualize training progress
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

print("Evaluating on rotated test images...")
errors = []
pred_angle = []
angles = []
for i in range(100):
    image, _ = dataset[i]
    angle = np.random.randint(0, 360)
    rotated = TF.rotate(image, angle)
    rotated_np = rotated.squeeze().numpy().reshape(1, img_size, img_size, 1) / 255.0
    pred = model.predict(rotated_np, verbose=0)
    predicted_angle = np.argmax(pred)
    error = abs(predicted_angle - angle)
    pred_angle.append(predicted_angle)
    angles.append(int(angle))
    print(f"Image {i+1}: Predicted = {predicted_angle}°, Actual = {angle}°, Error = {error}°")
    errors.append(error)

average_error = sum(errors) / len(errors)
print(f" Average prediction error on test set: {average_error:.2f}°")

percentage_accuracy = 1 - sum(errors) / sum(angles) * 100
print(f" Average prediction accuracy: {percentage_accuracy:.2f}°")


"""
#Time it takes to run the code
"""
#Use this to understand how long the code takes to run
before_time = datetime.now().hour *3600+  datetime.now().minute * 60 + datetime.now().second
after_time = datetime.now().hour *3600+  datetime.now().minute * 60 + datetime.now().second

print(f"Time taken: {(after_time - before_time)} seconds")

"""