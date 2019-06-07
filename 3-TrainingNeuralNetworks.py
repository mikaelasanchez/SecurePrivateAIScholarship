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
