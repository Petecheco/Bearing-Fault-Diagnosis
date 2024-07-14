import torch
import torch.nn as nn
from config.pooling_config import POOLING_DICT


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_type, pooling_stride=1, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        assert pooling_type in POOLING_DICT.keys(), "Pooling must be one of {}".format(POOLING_DICT.keys())
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = POOLING_DICT[pooling_type](pooling_stride)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.gap(out)
        return out


class MSCNN(nn.Module):
    def __init__(self, in_channels, num_classes, num_scales=4):
        super(MSCNN, self).__init__()
        kernel_list = [9, 17, 33, 65]
        self.multi_scale_layer = nn.ModuleList()
        for i in range(num_scales):
            self.multi_scale_layer.append(
                ConvBlock(in_channels, 64, kernel_size=kernel_list[i], pooling_stride=120, pooling_type='max'))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        output_features = []
        for layer in self.multi_scale_layer:
            output = layer(x)
            output_features.append(output)
        fused_features = torch.cat((output_features[0], output_features[1], output_features[2], output_features[3]),
                                   dim=1)
        result = self.classifier(fused_features)
        return result


if __name__ == '__main__':
    model = MSCNN(in_channels=1, num_classes=10, num_scales=4)
    data = torch.randn((10, 1, 1024))
    output = model(data)
    print(output.shape)
