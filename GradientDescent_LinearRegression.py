import scipy.io
from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np
from sympy import symbols, diff
from sympy.core.numbers import E

path = './data.mat'
data_input = scipy.io.loadmat(path)

# for key in data_input:
#     print(key)

# print(len(data_input['x']))
# print(len(data_input['y']))

###test part
# plt.title = 'data.mat'
# plt.plot(data_input['x'], data_input['y'])
# plt.show()
# x_data = data_input['x']
# x_square = x_data * x_data
# for i,element in enumerate(x_square):
#     print(x_data[i],'**2 = ',element)
###test part


def liner_regression_GD(input_data_x, input_data_y,epochs):
    theta_0, theta_1 = np.random.randn(2,1)
    print('initial theta value = {},{}'.format(theta_0,theta_1))
    lr = 0.2  # The learning Rate
    # epochs = 1000  # The number of iterations to perform gradient descent

    n = float(len(input_data_x)) # Number of elements in X
    # print('X =', len(X))
    # print('Y =', len(Y))
    #Performing Gradient Descent 
    for i in range(epochs): 
        Y_pred = theta_1*input_data_x + theta_0  # The current predicted value of Y        
        Gradinat_1 = (2/n) * sum(input_data_x*(Y_pred - input_data_y))  # Derivative wrt theta_1
        Gradinat_0 = (2/n) * sum(Y_pred - input_data_y)  # Derivative wrt theta_0
        theta_1 = theta_1 - lr * Gradinat_1  # Update m
        theta_0 = theta_0 - lr * Gradinat_0  # Update c
    return theta_0, theta_1




theta = [0,0]
theta[0], theta[1] = liner_regression_GD(data_input['x'], data_input['y'], 1000)
print(theta[0],theta[1])
Y_pred_func = data_input['x']*theta[1] + theta[0]
figure_title = 'least square line : y = {} + {} * x'.format(theta[1],theta[0])

plt.title(figure_title)
plt.scatter(data_input['x'], data_input['y'])
plt.plot(data_input['x'],Y_pred_func,'r-')
plt.show()




