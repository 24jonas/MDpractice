import math
from pyexpat import model
from tkinter import _test 
import numpy as np
import pandas as pd

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
DB = pd.read_csv(url)
test_size = 3000
lr =0.001

#Learning Function

#Sigmoid 
#Learning algorthm
def activation_fun(x):
    return 1 /(1+np.exp(-x))

def dx_activation_fun(x):
    s = activation_fun(x)
    return s *(1-s)

#Finding out how wrong the neural network 
def loss_fun(y_pred,y_true):
    return np.mean ((y_pred-y_true)**2)

def dx_loss_fun(y_pred,y_true):
    return 2* (y_pred-y_true)

# The Neural Network. 

class Neural_Network:
    def __init__(self,init_arguments):
        super(Neural_Network, self).__init__

        #Number of input features
        self.input_dim = init_arguments["input_dim"]
        #Output used for backwards propagation
        self.output_dim = init_arguments["output_dim"]
        #Neuron within the hidden layers
        self.num_hidden_layers = init_arguments["num_hidden_layers"]
        #same thing
        self.size_hidden_layers = init_arguments["size_hidden_layers"]
        #Itialization for the weights being random.
        self.seed = init_arguments["seed"]
        #the root mean square of the output and the correct answers
        #show how wrong the network is.
        self.loss_function = init_arguments["loss_function"]
        #The derivative of the loss function giving 
        #insight on how to change the weights going back.
        self.dx_loss_function = init_arguments["dx_loss_function"]
        # ex: reLU, Sigmoid.
        self.activation_fun = init_arguments["activation_fun"]
        # derivative of: reLU, Sigmoid.
        self.dx_activation_fun = init_arguments["dx_activation_fun"]

        np.random.seed(self.seed)

        #Weights and biases

        # First hidden layer weights and biases
        self.W_start = np.random.randn(self.input_dim, self.size_hidden_layers[0])
        self.B_start = np.random.randn(1, self.size_hidden_layers[0])

        # Hidden layers weights and biases
        self.W = [np.random.randn(self.size_hidden_layers[i], self.size_hidden_layers[i + 1])
          for i in range(self.num_hidden_layers - 1)]
        self.B = [np.random.randn(1, self.size_hidden_layers[i + 1])
          for i in range(self.num_hidden_layers - 1)]

        # Output layer weights and biases
        self.W_end = np.random.randn(self.size_hidden_layers[-1], self.output_dim)
        self.B_end = np.random.randn(1, self.output_dim)


        self.z = [None] * self.num_hidden_layers
        self.a = [None] * self.num_hidden_layers
        self.z_start = None
        self.a_start = None 
        self.z_end = None

def backward(self, x, y, y_pred):
    # Ensure input, target, and prediction are single samples with appropriate shapes
    #Creates 2D arrays with batch size 1
    x = x.reshape(1, -1)
    y = y.reshape(1, 1)
    y_pred = y_pred.reshape(1, 1)

    # Gradient of loss w.r.t. prediction
    d_loss = self.dx_loss_fun(y_pred, y)

    # Gradient of loss w.r.t. weighted input of output layer
    #Cross Entropy of the function.
    d_z_end = d_loss * self.dx_activation_fun(self.z_end)

    # Gradients for output layer weights and biases
    self.dW_end = np.matmul(self.a[self.num_hidden_layers - 1].T, d_z_end)
    self.dB_end = np.sum(d_z_end, axis=0, keepdims=True)

    # Gradient of loss w.r.t. activation of last hidden layer
    # .matmul does matrix multiplication
    d_a = np.matmul(d_z_end, self.W_end.T)

    # Gradients for hidden layers (backwards)
    self.dW = [None] * self.num_hidden_layers
    self.dB = [None] * self.num_hidden_layers

    for i in range(self.num_hidden_layers - 1, -1, -1):
        # Gradient of loss w.r.t. weighted input of current hidden layer
        d_z = d_a * self.dx_activation_fun(self.z[i])

        # Gradient of loss w.r.t. weights and biases of current hidden layer
        if i == 0:
            self.dW[i] = np.matmul(self.a_start.T, d_z)
        else:
            self.dW[i] = np.matmul(self.a[i-1].T, d_z)

        self.dB[i] = np.sum(d_z, axis=0, keepdims=True)

        # Gradient of loss w.r.t. activation of previous layer (for next iteration)
        if i > 0:
            d_a = np.matmul(d_z, self.W[i].T)
        else:
            # d_a for the input layer (gradient w.r.t. a_start)
            d_a_input = np.matmul(d_z, self.W[i].T)

    # Gradients for input layer
    # Use the calculated gradient w.r.t. a_start
    d_z_start = d_a_input * self.dx_activation_fun(self.z_start)
    self.dW_start = np.matmul(x.T, d_z_start)
    self.dB_start = np.sum(d_z_start, axis=0, keepdims=True)

    def update_weights(self,lr):
        self.W_start -= lr * self.dW_start
        self.B_start -= lr *self.dB_start
        for i in range(self.num_hidden_layers):
            self.W[i] -= lr *self.dW[i]
            self.B[i] -= lr *self.dB[i]
        self.W_end -= lr * self.dW_end
        self.B_end -= lr * self.dB_end


    total_loss = 0 

    for sample in range(test_size):
        out = model.forward(x_test[sample])
        loss = loss_fun(out,y_test[sample])
        print(f"Prediction: {out}, Target: {y_test[sample]}, Loss {loss} ")

    print(f"Loss: {total_loss/test_size}")