"""
This is how you save and load models
This is so you can load pretrained models or continue training on new data
"""

from torch import nn
import torch.utils.data
from torchvision import datasets, transforms
from torch import autograd
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import fc_model

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainSet = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                 train=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64,
                                          shuffle=True)

# Download and load the test data
testSet = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                train=False, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=64,
                                         shuffle=True)

image, label = next(iter(trainLoader))

# Create the network, define the criterion and optimiser

model = fc_model.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)

fc_model.train(model, trainLoader, testLoader,
               criterion, optimiser, epochs=2)

# The parameters for PyTorch networks are stored in a model's state_dict.
# We can see the state dict contains the weight and bias matrices for each
# of our layers.
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())

# We need to rebuild the model as it was trained before we save it
# or else a tensor size error will be thrown

checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}

# Save the state of the model
torch.save(checkpoint, 'checkpoint.pth')


# We can create a function that loads states
def load_checkpoint(filepath):
    state = torch.load(filepath)
    loaded_model = fc_model.Network(state['input_size'],
                                    state['output_size'],
                                    state['hidden_layers'])
    loaded_model.load_state_dict(checkpoint['state_dict'])

    return loaded_model


model = load_checkpoint('checkpoint.pth')
print(model)
