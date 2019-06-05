"""
Deep learning networks tend to be massive with hundreds of layers
You can build these with only weight matrices but that would be difficult
PyTorch has a module 'nn' that provides an efficient way to build large
neural networks.
"""

# Import packages
import torch.utils.data
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F


# Building a neural network that will identify text in an image
# We will use the MNIST dataset, consisting of greyscale handwritten digits
# Each image is 28x28 px

# Import torchvision datasets
from torchvision import datasets, transforms

# Define a transform to normalise the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5],
                                                     [0.5]),
                                ])  # use [0.5 0.5 0.5] for RGB

# Download and load the training data with 64 images per batch
trainSet = datasets.MNIST('MNIST_data/', download=True, train=True,
                          transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64,
                                          shuffle=True)

# Now we have the training data loaded into trainLoader and we make that into
# an iterator so we can use this to loop through the dataset
# 'images' is a tensor with size (64, 1, 28, 28)
# 64 images per batch, 1 colour channel and 28x28 images

dataIter = iter(trainLoader)
images, labels = next(dataIter)

print(type(images))
print(images.shape)
print(labels.shape)

# Display image
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show()

# Now we'll try to build a simple network using weight matrices and matrix
# multiplications.

# The networks so far have been fully-connected/dense networks
# Each unit in one layer is connected to each unit in the next
# In dense networks, each input must be a one dimensional vector
# Our images are 28x28 2D tensors however, so we need to convert them

# We need to convert this batch of images:
# (64, 1, 28, 28) --> (64, 784)
# 784 = 28 * 28
# This conversion is called flattening

# In our previous network, we had one output unit. Now we need an output
# unit for each digit (each possibility) - 10 output units.

# We want our network to predict the digit shown in the image, so we'll
# calculate probabilities that the image is of any one digit (or "class")
# This becomes a discrete probability distribution over the classes that
# tells us the most likely class for the image.


# Define activation function
def activation(x):
    return 1/(1+torch.exp(-x))


# Flatten the input images
# images.shape[0] = 64 is the number of batches
# -1 indicates the program chooses an appropriate number of columns
# We could type 784 but -1 is like a shortcut
inputs = images.view(images.shape[0], -1)

# Create parameters

# Input layer to hidden layer
w1 = torch.randn(784, 256)  # 784 input units with 256 hidden units
b1 = torch.randn(256)       # 256 bias terms

# Hidden layer to output layer
w2 = torch.randn(256, 10)   # 256 hidden to 10 outputs
b2 = torch.randn(10)        # 10 bias terms

h = activation(torch.mm(inputs, w1) + b1)
output = torch.mm(h, w2) + b2

# Now that we have 10 outputs for our network, we want to pass an image
# into the network and get a probability distribution that tells us
# the most likely classes the image belongs to.

# To calculate this, we often use the softmax function:
# sigma(x_i) = e^(x_i) / sum^K_k(e^(x_k))
# This function squishes each input x_i between 0 and 1, then normalises
# the values to give a probability distribution, where all probabilities
# sum to one.

# dim(ension) = 1 means it will take the sum across the columns
# This will give us a vector of 64 elements
# This will output 64x64 until we do .view which will give us 64 rows
# but one value for each row


def softmax(x_i):
    return torch.exp(x_i) / torch.sum(torch.exp(x_i), dim=1).view(-1, 1)


probabilities = softmax(output)

print(probabilities.shape)         # Check shape, should be (64, 10)
print(probabilities.sum(dim=1))    # Should sum to 1

# PyTorch provides a module called nn that makes building networks much
# simpler. This is how to build the same one with:
# 784 inputs, 256 hidden unites, 10 output units and softmax output


class NumberNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer
        # Creates parameters for weights and bias
        # and automatically calculates linear transformation
        self.hidden = nn.Linear(784, 256)

        # Output layer, 10 unites - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each operation
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


# Create the network and look at its text representation
model = NumberNetwork()
print(model)

# We can define the network more concisely using torch.nn.functional
# module. We usually import this as F.


class NumberNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 unites - one for each digit
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        # Output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)

        return x

# Activation Functions:
# Sigmoid
# TanH
# ReLu

# In practice, ReLu is used almost exclusively as the activation function
# for hidden layers
