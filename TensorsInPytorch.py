"""
This document contains notes from the Tensors in PyTorch Jupyter notebook
to aid the author in learning about deep learning with PyTorch.
"""

# Import PyTorch
import torch

# PyTorch is a framework for building and training neural networks
# Numpy arrays are tensors
# PyTorch moves these arrays to the GPU
# It provides a module for automatic gradient calculation for back-propagation
# and building neural networks.

# Neural networks are built from individual parts approximating neurons
# Each unit has some number of weighted inputs
# These inputs are summed (linear combination) then passed through an activation
# function to get the unit's output

# Mathematically: y = f(w1x1 + w2x2 + ... + b)
# With vectors, this is the dot product of two vectors:
# [x1 x2 ...xn] * [w1 w2 ... wn]

# Neural network computations are just linear algebra operations on tensors
# Vectors have 1 dimension
# Matrices have 2 dimensions
# An 3 dimensional tensor (array) can be used for RGB colour images

# Fundamental data structure for neural networks are tensors

'''
Code practice
'''


def activation(x):
    """ Sigmoid activation function
    advantages:
    - Real-valued
    - Can solve for differential
    - Acceptable mathematical representation of biological neuron behaviour:
    - Output shows if neuron is firing or not

    :param x: torch.Tensor
    :return: output of Sigmoid function
    """

    return 1/(1+torch.exp(-x))


# Generate random data
# The seed will make things predictable
torch.manual_seed(7)

# Create a tensor with shape (1,5) with values randomly distributed
# according to normal distribution with mean=0 sd=1
features = torch.randn((1, 5))

# Creates another tensor with the same shape as features and contains values
# from normal distribution
weights = torch.randn((1, 5))

# Creates a single value from normal distribution
bias = torch.randn((1, 1))

# PyTorch tensors can be added, multiplied, subtracted like Numpy arrays
# but they include things such as GPU acceleration

# Exercise: Calculate the output of the network with input features, weights,
# and bias.

weights = weights.view(5, 1)
y = activation(torch.sum(features * weights) + bias)

# This is not matrix multiplication - pure python uses elementwise
# multiplication

print("Output with elementwise multiplication: ")
print(y)

# Note tensors don't have the correct shapes to perform matrix multiplication
# If both features and weights have the same shape, we need to change the
# shape of weights to get matrix multiplication to work

# In order to do this, you can use:
# weights.reshape(a, b) - returns new tensor or a clone
# weights.resize_(a, b) - returns same tensor
# weights.view(a, b) - returns new tensor

# Exercise: Calculate the output of this network using matrix multiplication

y = activation(torch.mm(features, weights)+bias)
print("\nOutput using single layer network: ")
print(y)

'''
MultiLayer Networks
'''

# You can use the same inputs to power different hidden layers
# Stacking units into layers will make a network of neurons
# With multiple input units and output units, we now need to express weights
# as a matrix

# We can express a multilayer network mathematically:
# h-> = [h_1 h_2] = [x_1 x_2 ... x_n]*[w_11 w_12
#                                      w_21 w_22
#                                      ...  ...
#                                      w_n1 w_n2]
# where h = hidden layer
# w_11 = weights from node x_1 to h_1

# The network output is found by treating the hidden layer as inputs for
# the output unit. This is expressed by:
# y = f_2(f_1(x->*W_1 + B1)*W_2 + B2)

# 3 random normal variables
features = torch.randn((1, 3))

# Size of each layer
n_input = features.shape[1]  # number of input units, matches input features
n_hidden = 2                 # number of hidden units
n_output = 1                 # number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

# Calculate the output for this multi-layer network using the weights
# W1 and W2, and biases B1 and B2

h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)

print("\nOutput using multilayer network: ")
print(output)
