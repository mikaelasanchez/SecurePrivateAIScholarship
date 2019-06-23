# The network we had built previously needs to be trained as
# it doesn't know anything about our handwritten digits.
# Networks with non-linear activations work like universal
# function approximators.

# This means that if you have a function in the form of a list
# of inputs and outputs, there is a Neural Network that given
# any inputs will approximate the outputs very well.

# We can train a neural network to approximate any function,
# given enough data and computation time.

# At first, the network doesn't know the function mapping the
# inputs to the outputs, so we train it by showing examples of
# real data and then adjusting the network parameters such that
# it approximates the function.

# To find these, we need to know how poorly the network is
# predicting, so we calculate a loss function. This is a
# measure of our prediction error.

# Mean squared loss is often used in regression and binary
# classification problems:
# l = (1/2n)(sum{(y_i - ^y_i)^2})
# n - number of training samples
# y_i - true labels
# ^y_i - predicted labels

# By minimising this loss, we can find configurations where the
# loss is at a minimum and the network is able to predict the
# correct labels with high accuracy.
# We find this using a process called GRADIENT DESCENT
# The gradient is the slope of the loss function and points
# in the direction of fastest change
# We hence need to find the steepest slope

# For multilayer networks, we use BACKPROPAGATION to train
# This is really just an application of the chain rule in
# calculus.

# A small change in one of the weights, will create a small
# change in our loss.
# If we pass through our network backwards, we will see the
# changes going in the opposite direction.
# Going from the loss, to the layer, to the activation function,
# there will always be some derivative between the outputs and
# the inputs.

# l -> L2 -> S -> L1 -> W1
# l -> L2: dl/dL2
# L2 -> S: dl/dL2 x dL2/dS = dl/dS
# S -> L1: dl/dS x dS/dL1 = dl/dL1
# L1 -> W1: dl/dL1 x dL1/dW1 = dl/dW1

# This will allow us to calculate the gradient of the loss wrt
# to the weights

# If we want to minimise our loss, we can subtract the
# gradients from our weights.
# This will give us a new set of weights which will result in a
# smaller loss, generally

# How backpropagation works:
# 1) Do a forward pass
# 2) Go backwards through the network to calculate gradient
# 3) Update the weights
# 4) Repeat to sufficiently minimise the loss

# We update our weights by using this gradient with some
# learning rate alpha:
# W'_1 = W_1 x -a(dl/dW1)
# The learning rate a is set such that the weight update steps
# are small enough that the iterative method settles in a
# minimum

"""
Losses in PyTorch
-----------------
PyTorch provides losses such as cross-entropy loss:
nn.CrossEntropyLoss
You'll usually see the loss assigned to criterion

With a softmax output, you want to use cross-entropy as the loss
To actually calculate the loss, you first define the criterion
then pass in the output of your network and the correct labels

nn.CrossEntropyLoss
This criterion combines nn.LogSoftmax and nn.NLLLoss in a single
class. The input is expected to contain scores for each class.
"""

# This means we need to pass in the raw output of our network
# into the loss, not the output of softmax.
# These are called the LOGITS or SCORES
# We use the logits because softmax gives probabilities often
# very close to zero or one, but floats can't accurately
# represent values near zero or one.
# Typically we use log-probabilities

from torch import nn
import torch.utils.data
from torchvision import datasets, transforms
from torch import autograd
from torch import optim

# Define a transform to normalise the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5],
                                    [0.5]),
                                ])
# Download and load training data
trainSet = datasets.MNIST('~/.pytorch/MNIST_data/',
                          download=True, train=True,
                          transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)

# Build a feed forward network
# This is only returning the logits
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))

# Define the loss
criterion = nn.CrossEntropyLoss()

# Get our data
images, labels = next(iter(trainLoader))
# Flatten the images
images = images.view(images.shape[0], -1)

# Forward pass through the model to get our logits
logits = model(images)
# Calculate the loss with the logits ans the labels
loss = criterion(logits, labels)

print(loss)

# Now we know how to calculate a loss, we can use it in backpropagation!
# We import autograd to automatically calculate the gradients of our parameters
# wrt the loss
# We need to set requires_grad = True on a tensor to keep track of operations
# For example:

x = torch.randn(2, 2, requires_grad=True)  # Creating a tensor with requires_grad

# You can also toggle it globally using:
torch.set_grad_enabled(True)

# The grad_fn operation shows the function that generated a particular variable
# for example:
y = x**2
print(y)
print(y.grad_fn)

# The autograd module keeps track of these operations and knows how to calculate
# the gradient of each one.
# Let's reduce the tensor y to a scalar value, the mean
z = y.mean()
print(z)

# We can get the gradients after we calculate it, using the .backward method
# This differentiates the variable
z.backward()
print(x.grad)
print(x/2)

# For training, we need the gradients of the weights with respect to cost
# We do this by running data forward to calculate loss, then go backwards to
# calculate the gradients wrt to loss
# Once we have these, we can make a gradient descent step

# So now, to train the network, we need an optimiser to update the weights with
# the gradients. We get this from the optim package.
# Let's try the stochastic gradient descent (optim.SGD)

# Here we use parameters to optimise and learning rate
optimiser = optim.SGD(model.parameters(), lr=0.01)

# Remember you need to clear the gradients because gradients accumulate!
optimiser.zero_grad()

"""
In short, the general PyTorch process goes:
1 - forward pass
2 - network output to calculate loss
3 - backward pass with loss.backward() to calculate gradients
4 - take a step with the optimiser to update weights

each pass through the training set is called an epoch
"""
