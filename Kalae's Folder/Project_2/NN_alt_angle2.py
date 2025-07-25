import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

"""
NN_alt_angle2.py will be a modified version of NN_alt_angle.py. 
with 3 sets of weights and biases.
"""

# Z is standard notation for the input to the activation function
#Constants 
epochs = 50
alpha = 0.01
batch_size = 32
#################
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("/Users/kalaeabrams/Desktop/Images", transform=transform)


#################
for i in range(4):
    A = np.random.randint(1, 360)
    print(f"Rotating image {i} by {A} degrees")
    current_image = dataset[i][0]  # PyTorch tensor in [C, H, W] format
    rotated_image = torchvision.transforms.functional.rotate(current_image, A)

# Convert to [H, W, C] format for Matplotlib
    rotated_image_np = rotated_image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(rotated_image_np)
    plt.show()

#Loading in images from dataset. 

data = np.array(dataset)
m, n = data.shape
np.random.shuffle(data)
#shuffle for splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1,n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255 #Assuming the activation is between 0-255 for pixel
A, m_train = X_train.shape


def parameters(): ########################### supposed to be rand instead of randn (critical)
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) -0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5


    return W1, b1, W2, b2, W3, b3




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
    

for i in range(epochs):
    print(f"Epoch {i+1}/{epochs}")
    W1, b1, W2, b2, W3, b3 = parameters()
    
    for j in range(0, m_train, batch_size):
        X_batch = X_train[:, j:j+batch_size]
        Y_batch = Y_train[j:j+batch_size]

        forward_pass = ForwardPass(W1, b1, W2, b2, W3, b3, X_batch)
        Z1, A1, Z2, A2, Z3, A3 = forward_pass.forward()

        backward_pass = BackwardPass(W1, b1, W2, b2, W3, b3, X_batch, Y_batch)
        dW1, db1, dW2, db2, dW3, db3 = backward_pass.backward(Z1, A1, Z2, A2, Z3, A3)

        # Update parameters
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W3 -= alpha * dW3
        b3 -= alpha * db3

print(f"Training complete for this epoch: {i+1}")