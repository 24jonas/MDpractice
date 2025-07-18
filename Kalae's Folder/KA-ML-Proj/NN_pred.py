import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler



scalar = StandardScaler()


url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
Data = pd.read_csv(url)
#data = pd.read_csv('/input/digit-recognizer/train.csv')

#Hyper-parameter
#mpg vs displacement 8/10
#weight vs acceleration. more weight means less acceleration.

"""
Attempting to make to neural networks so that i understand the conepts pretty well.
mpg vs displacement looks like a clear negative linear relationship so i will train a model to 
predict the relation Y value.

On top of that I will use reLU to add a component of nonlinearity.

"""
#Hyperparameter:
alpha = 2
#learning rate
lr = 0.0001
#times we took our whole data set and pass it back in.
epoch = 10
batch_size = 8
#Callable functions.
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
PREDICTORS = ["mpg"]
TARGET = "displacement"




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

"""

def mse(actual, predicted):
    return np.mean((actual-predicted)**2)

#find the direction of greatest decrease.
#Gives a quantifiable number that we need to adjust
#predicted to reduce error.
def mse_gradient(actual, predicted):
    return 2*(predicted-actual)

#append the weights and biases arrays
#with a loop.
def initialized_layers(inputs):
    layers = []
    for i in range(1,len(inputs)):
        layers.append([np.random.rand(inputs[i-1],inputs[i])/ 5-0.1,
                      np.ones((1,inputs[i])) 
                       ])
    return layers

#layer parameters
layer_config = [1,10,10,1]

#layers = init_layers(layer_config)
        
#performs matrix multiplication as a loop.
def forward_pass(batch,layers):
    hiddens = [batch.copy()]
    for i  in range(len(layers)):
        batch = np.matmul(batch, layers[i][0]+layers[i][1])
        if i < len(layers) -1:
            batch = np.maximum(batch,0)
        hiddens.append(batch.copy())

    return batch, hiddens

def backward_pass(layers,hiddens,grad, lr):
    for i in range(len(layers)-1,-1,-1):
        if i != len(layers)-1:
            grad= np.multiply(grad,np.heaviside(hiddens[i+1], 0))

        w_grad = hiddens[i].T * grad
        b_grad = np.mean(grad,axis= 0)
        layers[i][0] -= w_grad *lr
        layers[i][1] -= b_grad *lr

        grad = grad *layers[i][0].T


#correlation coefficient -0.804
plt.scatter(Data['mpg'],Data['displacement'])
plt.xlabel('mpg')
plt.ylabel('displacement')
plt.title('mpg vs displacement')


corr = Data['mpg'].corr(Data['displacement'])
print("Correlation coefficient (mpg vs displacement):", corr)



prediction = lambda x, w1 = -12.2, b = 510: w1*x +b
plt.plot([10,40],[prediction(10),prediction(40)], 'green')
plt.show()

print(mse(Data['displacement'],Data['mpg']))

#we need to compensate for the non-linear component.


#mpg = tmax
#tmax_tmrw  = displacement

mpg_bins =pd.cut(Data["mpg"],25)
print("The Displacement Bins is:" )
print(mpg_bins)

print("ratios is ")
ratios = Data["displacement"]- 510 / Data['mpg']
print(ratios)
binned_ratio = ratios.groupby(mpg_bins).mean()
print("binned_ratio")
print(binned_ratio)

binned_mpg = Data["mpg"].groupby(mpg_bins).mean()

plt.scatter(binned_mpg, binned_ratio)
plt.xlabel('mpg_bins')
plt.ylabel('binned_ratio')
plt.title('mpg vs ratio ')
plt.show()
#
#
#
#
#
"""
Now that I graphed the mean of each of the data sets I can see that I wanna use AI 
to add a component that has nonlinearity.

We can add a nonlinear transoformation on top of the linear one

Multiple layers will capture interactions between features.

Layer one output = reLU(W_{1}X+B)

Layer Two:  W_{2}reLU(W_{1}X+B)+b

Layer Two output: reLU(W_{2}reLU(W_{1}X+B)+b)

More layers will make for better predictions because most data 
sets are nonlinear relationships.
"""
#times we took our whole data set and pass it back in.
epochs = 10
batch_size = 8

layer_config = [1,10,10,1]
layers = initialized_layers(layer_config)

#The Training Loop
for epoch in range(epochs):
    #average loss accross the whole traing cycle
    epoch_loss = 0

    for i in range(0, train_x.shape[0],batch_size):
        #how large imput is
        x_batch = train_x[i:(i+batch_size)]
        #how large output is
        y_batch = train_y[i:i+batch_size]

        pred, hidden = forward_pass(x_batch,layers)

        loss = mse_gradient(y_batch,prediction)
        epoch_loss += np.mean(loss**2)

        layers = backward_pass(layers,hidden,loss,lr)

    print(f"Epoch {epoch} Train MSE: {epoch_loss / train_x.shape /batch_size}")








"""

#correlation coefficient -0.543
plt.scatter(Data['cylinders'],Data['acceleration'])
plt.xlabel('cylinders')
plt.ylabel('acceleration')
plt.title('cylinders vs acceleration')
plt.show()

corr1 = Data['cylinders'].corr(Data['acceleration'])
print("Correlation coefficient (weight vs acceleration):", corr1)

"""
