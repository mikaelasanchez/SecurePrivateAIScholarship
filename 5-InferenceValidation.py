# Import modules
from torch import nn
import torch.utils.data
from torchvision import datasets, transforms
from torch import autograd
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
Making predictions is called inference (term from statistics)
Neural networks tend to perform too well on training data and aren't able
to generalise to data that hasn't been seen before
This is called "overfitting"

To test for overfitting, we measure performance on data called the validation
set.
We avoid overfitting through regularisation such as dropout while monitoring
the validation performance during training.
"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5],
                                    [0.5]),
                                ])

# Download and load training data
trainSet = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                 download=True, train=True,
                                 transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet,
                                          batch_size=64,
                                          shuffle=True)
# Download and load test data
testSet = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                download=True, train=False,
                                transform=transform)
testLoader = torch.utils.data.DataLoader(testSet,
                                         batch_size=64,
                                         shuffle=True)

# Define network model
model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Performance on the test data is usually just based on accuracy.
# Other options can be precision and recall, and top-5 error rate.

images, labels = next(iter(testLoader))
img = images.view(-1, 784)

# Get probabilities
ps = torch.exp(model(img))
print(ps.shape)

# We can get the most likely class using the ps.topk method
# This returns the k highest values. Since we just want the most likely class,
# we'll use ps.topk(1)
top_p, top_class = ps.topk(1, dim=1)
print(top_class[:10, :])   # Most likely classes for first 10 examples

# Now we can check if the predicted classes match the labels.
# We have to be careful of the shapes as...
# top_class is a 2D tensor (64, 1)
# labels is a 1D tensor with (64)
# This means top_class and labels must have the same shape
# If we do top_class == labels, equals will have the shape (64, 64)
equals = top_class == labels.view(*top_class.shape)

# If we sum all the values and divide by the number of values, we'll get the
# percentage of correct predictions. (same as finding the mean)
# We need to convert equals into a float tensor
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f"Accuracy: {accuracy.item()*100}%")

"""
Overfitting

The lower the validation loss, the better the network can generalise
The higher the validation loss, the more overfitting there is
"""

# A common method to reduce overfitting (outside of early stopping) is dropout
# We randomly drop input units, forcing the network to share information
# between weights, increasing its ability to generalise new data.
# We can do this by using the nn.Dropout module

model2 = nn.Sequential(nn.Linear(784, 256),
                       nn.ReLU(),
                       nn.Dropout(p=0.2),
                       nn.Linear(256, 128),
                       nn.ReLU(),
                       nn.Dropout(p=0.2),
                       nn.Linear(128, 64),
                       nn.ReLU(),
                       nn.Dropout(p=0.2),
                       nn.Linear(64, 10),
                       nn.LogSoftmax(dim=1))

# In training we use dropout to prevent overfitting, but during inference we
# want to use the entire network so we turn this off during:
# Validation, testing, making predictions
# To turn off dropout we do: model.eval()
# To turn on dropout we do: model.train()
# Validation loop looks like this:

with torch.no_grad():                   # turn off gradients
    model.eval()                        # evaluation mode
    for images, labels in testLoader:   # validation pass
        ...

model.train()                           # train mode

