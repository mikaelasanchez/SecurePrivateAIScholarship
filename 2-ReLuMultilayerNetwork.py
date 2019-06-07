# Imports
from torch import nn
import torch.nn.functional as f


class ReLuNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = f.relu(self.hidden(x))
        x = f.relu(self.hidden2(x))
        x = f.softmax(self.output(x), dim=1)

        # Loss layer
        return x
