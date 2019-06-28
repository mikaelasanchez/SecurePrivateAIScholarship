"""
The easiest way to load image data is with datasets.ImageFolder from
torchvision (documentation). In general you'll use ImageFolder.

Transforms are the list of processing steps built within the transforms module.
You need these as all images have different dimensions. You can use:
transforms.CenterCrop(), transforms.RandomResizedCrop(), transforms.Resize()

We'll also have to convert the images into tensors: transforms.ToTensor()
You'll combine these with transforms.Compose()

ImageFolder expects files and dictionaries to be constructed like so:
root/class/img.png

e.g.
root/dog/xxx.png
root/dog/xxy.png
root/cat/123.png
root/cat/789.png

So these images will be loaded into their respective class labels
"""
# Imports
from torch import nn
import torch.utils.data
from torchvision import datasets, transforms
from torch import autograd
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import fc_model
import helper

transform = transforms.Compose([transforms.Resize(255),      # Re-sizes to square
                                transforms.CenterCrop(224),  # Crops 255 px sides
                                transforms.ToTensor()])      # Convert to tensor

# Create dataset object and data generator to load images
dataSet = datasets.ImageFolder('dataset/training_set', transform=transform)
dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=32, shuffle=True)

# Testing data loader
images, labels = next(iter(dataLoader))
image = images[0].numpy().transpose((1, 2, 0))
plt.imshow(image)
plt.yticks([])
plt.xticks([])
plt.show()

"""
Data Augmentation 
Rotating, resizing, flipping simulates a larger dataset
This also helps the network to generalise images that aren't in the training set
"""

trainTransform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()])

testTransform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

trainData = datasets.ImageFolder('dataset/training_set', transform=trainTransform)
testData = datasets.ImageFolder('dataset/training_set', transform=testTransform)

trainLoader = torch.utils.data.DataLoader(trainData, batch_size=32)
testLoader = torch.utils.data.DataLoader(testData, batch_size=32)

# Training set
images, labels = next(iter(trainLoader))
fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
for i in range(4):
    ax = axes[i]
    helper.imshow(images[i], ax=ax, normalize=False)
plt.show()

# Testing set
images, labels = next(iter(testLoader))
fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
for i in range(4):
    ax = axes[i]
    helper.imshow(images[i], ax=ax, normalize=False)
plt.show()
