# Import modules
from torch import nn
import torch.utils.data
from torchvision import datasets, transforms
from torch import autograd
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# TODO: Load data
# Define a transform to normalise the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5],
                                    [0.5]),
                                ])
# Download and load training data
trainSet2 = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',
                                  download=True, train=True,
                                  transform=transform)
trainLoader2 = torch.utils.data.DataLoader(trainSet2,
                                           batch_size=64,
                                           shuffle=True)


# TODO: Define network architecture
model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# TODO: Create network, define criterion and optimiser
# Adam speeds up fitting process and adjusts learning rate
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.003)

# TODO: Train the network
epochs = 6
for e in range(epochs):
    running_loss = 0
    for images, labels in trainLoader2:
        images = images.view(images.shape[0], -1)  # Flatten images

        # Training pass
        optimiser.zero_grad()               # zero the gradients
        output = model.forward(images)      # forward pass
        loss = criterion(output, labels)    # calculate loss
        loss.backward()                     # backward pass
        optimiser.step()                    # optimiser step

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainLoader2)}")

# TODO: Test out network!
# Grab some data
images, labels = next(iter(trainLoader2))
img = images[0].view(1, 784)

with torch.no_grad():
    logits = model.forward(img)

# Calculate class probabilities (softmax)
ps = F.softmax(logits, dim=1)
ps = ps.data.numpy().squeeze()

# Plot image and probabilities
fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
ax1.axis('off')
ax2.barh(np.arange(10), ps)
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(['T-shirt/top',
                     'Trouser',
                     'Pullover',
                     'Dress',
                     'Coat',
                     'Sandal',
                     'Shirt',
                     'Sneaker',
                     'Bag',
                     'Ankle Boot'], size='small')
ax2.set_title('Class Probability')
ax2.set_xlim(0, 1.1)
plt.tight_layout()

plt.show()
