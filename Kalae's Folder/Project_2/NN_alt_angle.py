from turtle import color
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import keras 


# Z is standard notation for the input to the activation function
#Constants 
epochs = 500
alpha = 0.001
batch_size = 32
###############
#Arrays of angles
angles = np.arange(0, 360, 1)  # Angles from 0 to 359
error = []
#################
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((128, 128)),               # Resize to 128x128
    transforms.ToTensor()                        # Convert to PyTorch tensor
])

dataset = datasets.ImageFolder("/Users/kalaeabrams/Desktop/Images", transform=transform)


#################
#Functions
def deriv_reLU(Z):
    return Z > 0

def Softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def plot_sample(X,Y, index):
    plt.imshow(X[:, index].reshape(128, 128), cmap='gray')
    plt.title(f"Angle: {angles[Y[index]]} degrees")
    plt.axis('off')
    plt.show()
#################
def rotate_image(image, angle):
    """
    Rotate the image by a given angle.
    :param image: Input image as a PyTorch tensor.
    :param angle: Angle in degrees to rotate the image.
    :return: Rotated image as a PyTorch tensor.
    """
    return torchvision.transforms.functional.rotate(image, angle)

"""
for i in range(4):
    A = np.random.randint(0, 359)
    print(f"Rotating image {i} by {A} degrees")
    current_image = dataset[i][0]  # PyTorch tensor in [C, H, W] format
    rotated_image = torchvision.transforms.functional.rotate(current_image, A)

# Convert to [H, W, C] format for Matplotlib
    rotated_image_np = rotated_image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(rotated_image_np)
    plt.show()

"""
# lines 71-97 copilot generated because it stated that torchvision didn't work like that.
def preprocess_dataset(dataset, max_samples=None):
    X = []
    Y = []
    count = 0
    for image in dataset:
        A = np.random.randint(0, 359)
        current_image = dataset[count][0]  # PyTorch tensor in [C, H, W] format
        rotated_image = torchvision.transforms.functional.rotate(current_image, A)
        rotated_image_np = rotated_image.permute(1, 2, 0).numpy()  # shape: (16384,)
        X.append(rotated_image_np)
        Y.append(A)
        count += 1
        if max_samples and count >= max_samples:
            break
    X = np.array(X).T / 255.0  # shape: (16384, num_samples)
    Y = np.array(Y)
    return X, Y

X_all, Y_all = preprocess_dataset(dataset)

# Shuffle and split into train/dev
m = X_all.shape[1]
split = batch_size

X_dev = X_all[:, :split]
Y_dev = Y_all[:split]

X_train = X_all[:, split:]
Y_train = Y_all[split:]
m_train = X_train.shape[1]

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(128,128,1)),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(360, activation='softmax')  # Output layer for 360 classes
])
model.compile(optimizer= 'SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ann = model.fit(X_train.T, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_dev.T, Y_dev))

"""
Training = models.Sequential([
    models.Linear(16384, 128),  # Input layer
    models.ReLU(),
    models.Linear(128, 64),
    models.ReLU(),
    models.Linear(64, 360),  # Output layer (360 classes)
    models.Softmax(dim=1)
])
"""