# -*- coding: utf-8 -*-

import numpy as np

# Assigning input values.
input_values = np.array([[0,0], [0,1], [1,1], [1,0]])
input_values.shape
input_values

# Assigning output values.
output = np.array([0,1,1,0])
output = output.reshape(4,1)
output.shape    

# Assigning weights.
weights = np.array([[0.1], [0.2]])
weights

# Adding the bias.
bias = 0.3

# The activartion function.
def sigmoid_func(x):
    return 1/(1 + np.exp(-x))

# The derivative of the sigmoid function.
def der(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))

# Updating weights.
for epochs in range(10000):
    input_array  = input_values
    
    weighted_sum = np.dot(input_array, weights) + bias
    first_output = sigmoid_func(weighted_sum)
    
    error = first_output - output
    total_error = np.square(np.subtract(first_output, output)).mean()
    #print(total_error)
        
    first_der = error
    second_der = der(first_output)
    derivative = first_der * second_der
    
    t_input = input_values.T
    final_derivative = np.dot(t_input, derivative)
    
    # Update weights.
    weights = weights - 0.05 * final_derivative
    
    # Update bias.
    for i in derivative:
        bias = bias - 0.05 * i
        
print(weights)
print(bias)

pred = np.array([0,0])

results = np.dot(pred, weights) + bias

res = sigmoid_func(results)

print(res)
















