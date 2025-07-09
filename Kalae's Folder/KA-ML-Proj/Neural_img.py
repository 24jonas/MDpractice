import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



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

"""
#Hyperparameter:
alpha = 2

#Callable functions.
def mse(actual, predicted):
    return np.mean((actual-predicted)**2)





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




#t_max = displacement
#
displacement_bins =pd.cut(Data["displacement"],25)
print(displacement_bins)

ratios = Data["displacement"]- 510 / Data['mpg']
print(ratios)
binned_ratio = ratios.groupby(displacement_bins).mean
print(binned_ratio)
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
