# Imports
from torch import nn
import torch.utils.data
from torchvision import datasets, transforms
from torch import autograd
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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

# Define the model
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Define loss
criterion = nn.NLLLoss()

# Define optimiser
optimiser = optim.SGD(model.parameters(), lr=0.003)

# Number of epochs (passes)
epochs = 6

# Do training passes
for e in range(epochs):
    running_loss = 0
    for images, labels in trainLoader:
        # Flatten MNIST images into a  784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass!
        optimiser.zero_grad()               # zero gradients
        output = model.forward(images)      # forward pass
        loss = criterion(output, labels)    # calculate the loss
        loss.backward()                     # backward pass
        optimiser.step()                    # optimiser step

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainLoader)}")

# Loss should drop over time
# Show prediction:

# Grab some data
images, labels = next(iter(trainLoader))

# Resize images into a 1D vector
# Shape: (batch size, colour channels, pixels)
img = images[0].view(1, 784)
# or images.resize_(images.shape[0], 1, 784)

# Forward pass through the network
with torch.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim=1)
ps = ps.data.numpy().squeeze()

fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
ax1.axis('off')
ax2.barh(np.arange(10), ps)
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))

ax2.set_yticklabels(np.arange(10))

ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)

plt.tight_layout()
plt.show()

