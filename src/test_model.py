import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import DEVICE
"""
Test model for federated learning. This model is designed to be more balanced in terms of parameter distribution across layers,
which can help in scenarios where clients have varying amounts of data and computational resources.
The architecture consists of four convolutional blocks followed by two fully connected layers,
making it suitable for image classification tasks.

"""

class fl_cnn_balanced(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        

        self.fc1 = nn.Linear(1024, 128)

        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def load_model():
    return fl_cnn_balanced().to(DEVICE)