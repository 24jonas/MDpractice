import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
DB = pd.read_csv(url)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
Data = pd.read_csv(url)

# Drop missing values to avoid errors
PREDICTORS = ["mpg"]
TARGET = "displacement"
Data = Data.dropna(subset=PREDICTORS + [TARGET])

# Scale the input feature (but not the target)
scaler = StandardScaler()
Data[PREDICTORS] = scaler.fit_transform(Data[PREDICTORS])

# Split the dataset: 70% train, 15% validation, 15% test
n = len(Data)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train = Data.iloc[:train_end]
valid = Data.iloc[train_end:val_end]
test = Data.iloc[val_end:]

# Convert each split into NumPy arrays (X, y)
train_x, train_y = train[PREDICTORS].to_numpy(), train[[TARGET]].to_numpy()
valid_x, valid_y = valid[PREDICTORS].to_numpy(), valid[[TARGET]].to_numpy()
test_x, test_y = test[PREDICTORS].to_numpy(), test[[TARGET]].to_numpy()

# Utility functions
def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def mse_gradient(actual, predicted):
    return 2 * (predicted - actual)

# Neural network layer initialization
def initialized_layers(layer_sizes):
    layers = []
    for i in range(1, len(layer_sizes)):
        weight = np.random.rand(layer_sizes[i-1], layer_sizes[i]) / 5 - 0.1
        bias = np.ones((1, layer_sizes[i]))
        layers.append([weight, bias])
    return layers

# Forward pass
def forward_pass(batch, layers):
    hiddens = [batch.copy()]
    for i in range(len(layers)):
        weight, bias = layers[i]
        batch = np.matmul(batch, weight) + bias
        if i < len(layers) - 1:
            batch = np.maximum(batch, 0)  # ReLU
        hiddens.append(batch.copy())
    return batch, hiddens

# Backward pass
def backward_pass(layers, hiddens, grad, lr):
    for i in range(len(layers) - 1, -1, -1):
        if i != len(layers) - 1:
            grad = grad * np.heaviside(hiddens[i + 1], 0)  # ReLU derivative

        w_grad = hiddens[i].T @ grad
        b_grad = np.mean(grad, axis=0, keepdims=True)

        layers[i][0] -= w_grad * lr
        layers[i][1] -= b_grad * lr

        grad = grad @ layers[i][0].T

# Plotting initial correlation
plt.scatter(Data['mpg'], Data['displacement'])
plt.xlabel('mpg')
plt.ylabel('displacement')
plt.title('mpg vs displacement')
corr = Data['mpg'].corr(Data['displacement'])
print("Correlation coefficient (mpg vs displacement):", corr)

# Plot simple linear guess
prediction = lambda x, w1 = -12.2, b = 510: w1 * x + b
plt.plot([Data['mpg'].min(), Data['mpg'].max()],
         [prediction(Data['mpg'].min()), prediction(Data['mpg'].max())],
         'green')
plt.show()

# Optional: look at binned behavior
mpg_bins = pd.cut(Data["mpg"], 25)
ratios = Data["displacement"] - 510 / Data['mpg']
binned_ratio = ratios.groupby(mpg_bins).mean()
binned_mpg = Data["mpg"].groupby(mpg_bins).mean()

plt.scatter(binned_mpg, binned_ratio)
plt.xlabel('mpg_bins')
plt.ylabel('binned_ratio')
plt.title('mpg vs ratio')
plt.show()

# Training Hyperparameters
epochs = 10
batch_size = 8
lr = 0.0001

# Define and initialize layers: [input, hidden1, hidden2, output]
layer_config = [1, 10, 10, 1]
layers = initialized_layers(layer_config)

# Training loop
for epoch in range(epochs):
    epoch_loss = 0

    for i in range(0, train_x.shape[0], batch_size):
        x_batch = train_x[i:i + batch_size]
        y_batch = train_y[i:i + batch_size]

        pred, hidden = forward_pass(x_batch, layers)

        loss_grad = mse_gradient(y_batch, pred)
        epoch_loss += mse(y_batch, pred)

        backward_pass(layers, hidden, loss_grad, lr)

    avg_loss = epoch_loss / (train_x.shape[0] // batch_size)
    print(f"Epoch {epoch + 1} Train MSE: {avg_loss:.4f}")
