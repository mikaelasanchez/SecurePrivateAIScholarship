"""
This document contains notes from the Tensors in PyTorch Jupyter notebook
to aid the author in learning about deep learning with PyTorch.
"""

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

# Import PyTorch
import torch


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

print(y)

# Note tensors don't have the correct shapes to perform matrix multiplication
# If both features and weights have the same shape, we need to change the
# shape of weights to get matrix multiplication to work

# In order to do this, you can use:
# weights.reshape(a, b)
# weights.resize_(a, b)
# weights.view(a, b)

# Exercise: Calculate the output of this network using matrix multiplication

y = activation(torch.mm(features, weights)+bias)
print(y)

