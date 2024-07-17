import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, kernel_size=100, out_channels=32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 100),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 100),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, drop_out_rate):
        out = F.dropout(x, p=drop_out_rate)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.gap(out)
        return out


class LinearAttention(nn.Module):
    def __init__(self):
        super(LinearAttention, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        feature_1 = x1.sum(dim=1, keepdim=True)
        feature_2 = x2.sum(dim=1, keepdim=True)
        feature_3 = x3.sum(dim=1, keepdim=True)
        out1 = self.flatten(feature_1)
        out2 = self.flatten(feature_2)
        out3 = self.flatten(feature_3)
        out = torch.cat((out1, out2, out3), dim=1)
        weight = self.softmax(self.fc(out))
        return weight


class MSAFCN(nn.Module):
    def __init__(self):
        super(MSAFCN, self).__init__()
        self.granularity1 = nn.AvgPool1d(1)
        self.granularity2 = nn.AvgPool1d(2)
        self.granularity3 = nn.AvgPool1d(4)
        self.conv1 = BottleNeck()
        self.conv2 = BottleNeck()
        self.conv3 = BottleNeck()
        self.attention = LinearAttention()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x, drop_out_rate):
        g1 = self.granularity1(x)
        g2 = self.granularity2(x)
        g3 = self.granularity3(x)
        feature1 = self.conv1(x, drop_out_rate)
        feature2 = self.conv2(x, drop_out_rate)
        feature3 = self.conv3(x, drop_out_rate)
        weight = self.attention(feature1, feature2, feature3)
        weighted_feature1 = feature1 * weight[:, 0].unsqueeze(1).unsqueeze(-1)
        weighted_feature2 = feature2 * weight[:, 1].unsqueeze(1).unsqueeze(-1)
        weighted_feature3 = feature3 * weight[:, 2].unsqueeze(1).unsqueeze(-1)
        out = weighted_feature1 + weighted_feature2 + weighted_feature3
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    data = torch.randn((10,1,1024))
    model = MSAFCN()
    out = model(data,0)