from turtle import color
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision


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

#################
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





#Loading in images from dataset.

##data = np.array(dataset)
#m, n = data.shape
#np.random.shuffle(data)
#shuffle for splitting into dev and training sets

#data_dev = dataset[0:batch_size].T 
#Y_dev = data_dev[0]
#X_dev = data_dev[1,n]
#X_dev = X_dev / 255 #Normalizing the data


"""
data_train = dataset[batch_size:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255 #Assuming the activation is between 0-255 for pixel
A, m_train = X_train.shape
"""
# lines 71-97 copilot generated because it stated that torchvision didn't work like that.
def preprocess_dataset(dataset, max_samples=None):
    X = []
    Y = []
    count = 0
    for image, label in dataset:
        image_np = image.squeeze().numpy().flatten()  # shape: (16384,)
        X.append(image_np)
        Y.append(label)
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


def parameters(): ########################### supposed to be rand instead of randn (critical)
    W1 = np.random.rand(10,128*128) - 0.5
    b1 = np.random.rand(10,1) -0.5
    W2 = np.random.rand(360,10) - 0.5
    b2 = np.random.rand(360,1) - 0.5


    return W1, b1, W2, b2


class ForwardPass:
    def __init__ (self, W1, b1, W2, b2, input_data):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.input_data = input_data

    # For debugging purposes
    def print_shapes(self):
        print("W1 shape:", self.W1.shape)
        print("b1 shape:", self.b1.shape)
        print("W2 shape:", self.W2.shape)
        print("b2 shape:", self.b2.shape)
        print("A1 shape:", A1.shape)

    def forward(self):
        Z1 = np.dot(self.W1, self.input_data) + self.b1
        A1 = np.maximum(Z1,0)  # ReLU activation
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = np.maximum(Z2, 0)  # ReLU activation
        return Z1, A1, Z2, A2

   
    

class BackwardPass:
    def __init__(self,j, W1, b1, W2, b2, Z1, A1, Z2, A2, angle, m_train, input_data):
        self.j = j
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.m_train = m_train
        self.angle = angle
        self.Z1 = Z1
        self.A1 = A1
        self.Z2 = Z2
        self.A2 = A2
        self.alpha = 0.01
        self.input_data = input_data

    def back_propagation(self): ############## m_train is the size of the training batch

        self.dZ2 = 2 * (self.A2 - self.angle)
        self.dW2 = (1 / self.m_train) * self.dZ2.dot(self.A1.T)
        self.db2 = (1 / self.m_train) * np.sum(self.dZ2, axis=1, keepdims=True)
        self.dZ1 = self.W2.T.dot(self.dZ2) * deriv_reLU(self.Z1)
        self.dW1 = (1/self.m_train) * self.dZ1.dot(self.input_data.T)
        self.db1 =  (1/self.m_train) * np.sum(self.dZ1, axis=1, keepdims=True)



    def update_parameters(self):
        self.W1 -= self.alpha * self.dW1
        self.b1 -= self.alpha * self.db1
        self.W2 -= self.alpha * self.dW2
        self.b2 -= self.alpha * self.db2
        return self.W1, self.b1, self.W2, self.b2
    
    def error(self):
        if self.j % 100 == 0:  
            print(f"The total error is", {np.abs(self.A2 - self.angle)})
        return 0

for i in range(epochs):
    if i == 0:
        W1, b1, W2, b2 = parameters()
        print("starting...")
    for j in range(batch_size):
        '''
        X_batch = X_train[:, j:j+batch_size]
        Y_batch = Y_train[j:j+batch_size]
        '''
        # Randomly rotate the image
        A = np.random.randint(0, 359)

        rotated_image = torchvision.transforms.functional.rotate(dataset[j][0], A)


        rotated_np = rotated_image.squeeze().numpy()  # [128,128]
        X_input = rotated_np.flatten().reshape(-1, 1) / 255.0  # [128*128, 1]
        """
        current_image = current_image.Grayscale()  # Convert to grayscale if needed
        rotated_image_np = rotated_image.permute(1, 2, 0).numpy()
        """

        forward_pass = ForwardPass(W1, b1, W2, b2, X_input)
        Z1, A1, Z2, A2 = forward_pass.forward()


        backward_pass = BackwardPass(j, W1, b1, W2, b2, Z1, A1, Z2, A2, A, m_train, X_input)
        backward_pass.back_propagation()
        W1, b1, W2, b2 = backward_pass.update_parameters()
       # backward_pass = BackwardPass(W1, b1, W2, b2, X_batch, Y_batch)
       # W1, b1, W2, b2 = backward_pass.backward(Z1, A1, Z2, A2)

    if i %10 == 0:
        error_now = []
        for i in range(100):

           

            image_tensor, _ = dataset[i]
            angle = np.random.randint(0, 360)
            rotated_tensor = torchvision.transforms.functional.rotate(image_tensor, angle)
            rotated_np = rotated_tensor.permute(1, 2, 0).numpy()
            X = rotated_np.flatten().reshape(-1, 1) / 255.0
            forward = ForwardPass(W1, b1, W2, b2, X)
            _, _, _, A2 = forward.forward()
            predicted_angle = np.argmax(A2)
            error_now.append((abs(angle-predicted_angle)))
        average_error = sum(error_now) / len(error_now)
        print(average_error )

print(f"Training complete for this epoch: {i+1}")

print("Testing predictions...")
for i in range(100):
    image_tensor, _ = dataset[i]
    angle = np.random.randint(0, 360)
    rotated_tensor = torchvision.transforms.functional.rotate(image_tensor, angle)
    rotated_np = rotated_tensor.permute(1, 2, 0).numpy()
    X = rotated_np.flatten().reshape(-1, 1) / 255.0

    forward = ForwardPass(W1, b1, W2, b2, X)
    _, _, _, A2 = forward.forward()
    predicted_angle = np.argmax(A2)

    print(f"Test {i+1}: Predicted angle = {predicted_angle}, Actual angle = {angle}")
    
    error.append((abs(angle-predicted_angle)))

    print(f"error,{ error[i]}" )

"""
    # Show image
    plt.imshow(rotated_np)
    plt.title(f"Predicted: {predicted_angle}° | Actual: {angle}°")
    plt.axis('off')
    plt.show()
"""
average_error = sum(error) / len(error)
print(f"\nAverage error across test images: {average_error:.2f}°")


"""

for i in range(4):
    print(f"Testing image {i}")
    current_image = dataset[i][0]  # PyTorch tensor in [C, H, W] format
    # Randomly rotate the image
    A = np.random.randint(0, 359) 
    rotated_image = torchvision.transforms.functional.rotate(current_image, A)
    current_image = dataset[A][0] 
    rotated_image_np = rotated_image.permute(1, 2, 0).numpy()
    forward_pass = ForwardPass(W1, b1, W2, b2, rotated_image_np)
    Z1, A1, Z2, A2 = forward_pass.forward()

    prediction = np.argmax(backward_pass.A2, axis=0)
    print(f"Predicted angle: {prediction}")
    print(f"Actual angle: {A}")

    """