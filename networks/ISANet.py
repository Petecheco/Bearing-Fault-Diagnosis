import torch
import torch.nn as nn


class feature_extractor(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(feature_extractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=0)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1, padding=1)
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=hidden_size * 3, out_features=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        identity = out
        out1 = self.gap(out)
        out1 = self.conv2(out1)
        out1 = self.flatten(out1)
        out1 = self.fc1(out1)
        out = out1 * out + out
        out = self.mp(out)
        return out


class ISANetimpl(nn.Module):
    def __init__(self):
        super(ISANetimpl, self).__init__()
        self.fe1 = feature_extractor(1, hidden_size=32)
        self.fe2 = feature_extractor(32, hidden_size=32)
        self.fe3 = feature_extractor(32, 32)

        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32, out_features=6)

        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        out = self.fe1(x)
        out = self.fe2(out)
        features = self.fe3(out)

        classicier_out = self.pooling(features)
        classicier_out = self.flatten(classicier_out)
        classicier_out = self.fc1(classicier_out)

        domain_out = self.pooling(features)
        domain_out = self.flatten(domain_out)
        domain_out = self.fc2(domain_out)

        return classicier_out, domain_out
