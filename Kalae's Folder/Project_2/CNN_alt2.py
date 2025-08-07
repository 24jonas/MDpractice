import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import tensorflow as tf 
from torchvision import datasets, transforms  

# Constants
img_size = 128
num_classes = 360
num_samples = 1000
batch_size = 32
epochs = 20
# Arrays



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("/Users/kalaeabrams/Desktop/Images", transform=transform)


dataset.Images.load_data()

#(X_train, Y_train), (X_test, Y_test)  = dataset.load_data()