from torch import nn

from constants import DEVICE


class ConvGNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        pool: bool = False,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResNet9FL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.prep = ConvGNBlock(in_channels, 64, pool=True)
        self.layer1 = ConvGNBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvGNBlock(128, 128, pool=False),
            ConvGNBlock(128, 128, pool=False),
        )
        self.layer2 = ConvGNBlock(128, 256, pool=True)
        self.layer3 = ConvGNBlock(256, 256, pool=True)
        self.res2 = nn.Sequential(
            ConvGNBlock(256, 256, pool=False),
            ConvGNBlock(256, 256, pool=False),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.res1(x) + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res2(x) + x
        return self.classifier(x)


def load_model(num_classes: int = 10):
    """Create a ResNet-9 FL model with dataset-specific output classes."""
    return ResNet9FL(num_classes=num_classes).to(DEVICE)
