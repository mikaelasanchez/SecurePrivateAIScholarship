"""
Transfer learning

Using pre-trained networks you can solve challenging problems in computer vision
Here we'll use the ImageNet dataset(1 million labeled images in 1000 categories)
This is used to train deep neural networks using an architecture called
convolutional layers.

Using a pre-trained network on images not in the training set is called transfer
learning.
Let's use transfer learning to train a network that can classify our cat and dog
photos with near perfect accuracy.
"""

# Imports
from torch import nn
import torch.utils.data
from torchvision import datasets, transforms, models
from collections import OrderedDict

# We will need to match the pre-trained model's input and normalisation
# Each colour channel was normalised separately
# mean: [0.485, 0.456, 0.0406] sd: [0.229, 0.224, 0.225]

trainTransform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(20),
                                     transforms.ToTensor()])

testTransform = transforms.Compose([transforms.Resize(240),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor])


trainData = datasets.ImageFolder('dataset/training_set', transform=trainTransform)
testData = datasets.ImageFolder('dataset/test_set', transform=testTransform)

trainLoader = torch.utils.data.DataLoader(trainData, batch_size=63, shuffle=True)
testLoader = torch.utils.data.DataLoader(testData, batch_size=32)

# Let's load in the DenseNet model
model = models.densenet121(pretrained=True)
# The classifier is Linear combination with 1024 input and 1000 output
# This was trained on ImageNet
# We need to retrain the classifier - keep features static but update the classifier

# Freeze our feature parameters
# This means when we run our tensors through, gradients won't be calculated
# This will also speed up training
for param in model.parameters():
    param.require_grad = False

# Here we will give Sequential a list of operations and a tensor will be passed sequentially
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))

# Attach classifier to model
model.classifier = classifier

# We should train this on the GPU
# Moves parameters of model to the GPU
# Make sure your tensors are also on the GPU if your model is also on the GPU
model.cuda()

# To move these back, you can do
model.cpu()

# Checks if you have a GPU that can use cuda
cuda = torch.cuda.is_available()

