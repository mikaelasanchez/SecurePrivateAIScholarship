"""
Exercise
"""
from torch import nn
import torch.utils.data
from torchvision import datasets, transforms

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
trainLoader = torch.utils.data.DataLoader(trainSet,
                                          batch_size=64,
                                          shuffle=True)

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Define the loss
criterion = nn.NLLLoss()

# Get our data
images, labels = next(iter(trainLoader))
# Flatten the images
images = images.view(images.shape[0], -1)

# Forward pass through the model to get our log probability
logps = model(images)
# Calculate the loss with the logits ans the labels
loss = criterion(logps, labels)

print(loss)

print("Before backward pass: \n", model[0].weight.grad)

loss.backward()

print("After backward pass: \n", model[0].weight.grad)
