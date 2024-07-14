import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_dim, input_length, hidden_dim, output_dim, dropout=None):
        super(CNN, self).__init__()
        self.flatten_dim = self.calculate_dim(input_length) * hidden_dim
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout is not None else nn.Identity(),
            nn.Linear(in_features=self.flatten_dim, out_features=output_dim),
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.classifier(out)
        return out

    def calculate_dim(self, input_length):
        for i in range(3):
            input_length = ((input_length - 5 + 2 * 2) // 1) + 1
            input_length = input_length // 2
        return input_length


if __name__ == '__main__':
    cnn = CNN(input_dim=1, input_length=2048, hidden_dim=64, output_dim=10)
    data = torch.randn((10, 1, 2048))
    out = cnn(data)
    print(out.shape)
