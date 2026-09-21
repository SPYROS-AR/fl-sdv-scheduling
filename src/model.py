import torch
import torch.nn as nn
from constants import DEVICE

class ConvGNBlock(nn.Module):
    """
    Standard FL Convolution block: Conv2d -> GroupNorm -> ReLU -> optional MaxPool2d.
    GroupNorm prevents the running-statistic bias caused by BatchNorm on non-IID client partitions.
    """

    def __init__(self, in_channels, out_channels, num_groups=8, pool=False):

        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResNet9FL(nn.Module):
    """
    Standard ResNet-9 adapted for Federated Learning under extreme Non-IID conditions.
    Features:
      - Group Normalization across all feature stages.
      - Residual skip connections to stabilize client weight aggregation.
      - Global Average Pooling to reduce parameter overhead and local overfitting.
    """

    def __init__(self, in_channels=3, num_classes=10, dropout_rate=0.2):
        super().__init__()
        
        # Stage 0: Initial projection (32x32 -> 16x16)
        self.prep = ConvGNBlock(in_channels, 64, pool=True)

        # Stage 1: Downsample + Residual Block (16x16 -> 8x8)
        self.layer1 = ConvGNBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvGNBlock(128, 128, pool=False),
            ConvGNBlock(128, 128, pool=False)
        )
        # Stage 2: Intermediate Downsample (8x8 -> 4x4)
        self.layer2 = ConvGNBlock(128, 256, pool=True)
        # Stage 3: Downsample + Residual Block (4x4 -> 2x2)
        self.layer3 = ConvGNBlock(256, 256, pool=True)
        self.res2 = nn.Sequential(
            ConvGNBlock(256, 256, pool=False),
            ConvGNBlock(256, 256, pool=False)
        )
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal initialization for stable initial federated rounds."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1(out)
        out = self.res1(out) + out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.res2(out) + out
        return self.classifier(out)

def load_model():
    return ResNet9FL().to(DEVICE)