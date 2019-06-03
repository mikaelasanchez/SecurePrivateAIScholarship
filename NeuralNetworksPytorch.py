"""
Deep learning networks tend to be massive with hundreds of layers
You can build these with only weight matrices but that would be difficult
PyTorch has a module 'nn' that provides an efficient way to build large
neural networks.
"""

# Import packages
import torch.utils.data
import matplotlib.pyplot as plt


# Building a neural network that will identify text in an image
# We will use the MNIST dataset, consisting of greyscale handwritten digits
# Each image is 28x28 px

# Import torchvision datasets
from torchvision import datasets, transforms

# Define a transform to normalise the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5)),
                                ])

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
images, labels = dataIter.__next__()

print(type(images))
print(images.shape)
print(labels.shape)

# Display image
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
