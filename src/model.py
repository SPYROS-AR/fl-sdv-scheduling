import torch.nn as nn
import torch.nn.functional as F

from constants import LAYER_1, LAYER_2


class flCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=LAYER_1[0], out_channels=LAYER_1[1], kernel_size=LAYER_1[2], padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=LAYER_2[0], out_channels=LAYER_2[1], kernel_size=LAYER_2[2], padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        
        
        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=32)
        
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        
        
    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = self.flatten(x)
        

        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        
        return x