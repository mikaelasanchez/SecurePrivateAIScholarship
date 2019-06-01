'''
This document contains notes from the Tensors in PyTorch Jupyter notebook
to aid the author in learning about deep learning with PyTorch.
'''

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

