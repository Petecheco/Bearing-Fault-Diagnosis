import torch
import torch.nn as nn
import torch.nn.functional as F
from config.pooling_config import POOLING_DICT


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pooling_stride=2, pooling="max"):
        super(ConvBlock, self).__init__()
        assert pooling in POOLING_DICT.keys(), "Pooling must be one of {}".format(POOLING_DICT.keys())
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = POOLING_DICT[pooling](pooling_stride)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        out = self.pool(out)
        return out


class TICNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_of_classes, num_of_layers=3):
        super(TICNN, self).__init__()
        self.hidden_dims = [hidden_dim * (2 ** i) for i in range(num_of_layers)]
        self.feature_extractor = nn.ModuleList()
        for index, dim in enumerate(self.hidden_dims):
            if index == 0:
                self.feature_extractor.append(ConvBlock(in_channels, dim, kernel_size=64, stride=8, padding=30))
            else:
                self.feature_extractor.append(
                    ConvBlock(self.hidden_dims[index - 1], self.hidden_dims[index], kernel_size=3, padding=1))
        self.repeated_layers = nn.Sequential(
            ConvBlock(hidden_dim * (2 ** (len(self.hidden_dims) - 1)), hidden_dim * (2 ** (len(self.hidden_dims) - 1)),
                      kernel_size=3, padding=1),
            ConvBlock(hidden_dim * (2 ** (len(self.hidden_dims) - 1)), hidden_dim * (2 ** (len(self.hidden_dims) - 1)),
                      kernel_size=3, padding=1),
            ConvBlock(hidden_dim * (2 ** (len(self.hidden_dims) - 1)), hidden_dim * (2 ** (len(self.hidden_dims) - 1)),
                      kernel_size=3, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 100),
            nn.Linear(100, num_of_classes)
        )

    def forward(self, x, epoch):
        if epoch + 1 % 5 == 0:
            dropout_rate = 0
        else:
            dropout_rate = torch.rand(1) * 0.8 + 0.1
        x = F.dropout(x, p=dropout_rate)
        for layer in self.feature_extractor:
            x = layer(x)
        x = self.repeated_layers(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = TICNN(1, 16, 10, 3)
    data = torch.randn(10, 1, 2048)
    output = model(data)
    print(output)
