import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class FDGRU(nn.Module):
    """
    The implmentation of FDGRU, which takes input as [B, 4096, 1]
    """
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(FDGRU, self).__init__()
        self.input_linear = nn.Linear(64, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.bn_linear = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(65536, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
    def forward(self, x):
        x = x.squeeze(-1).detach().numpy()
        x_img = np.zeros((x.shape[0],64,64))
        for i in range(64):
            x_img[:,i] = x[:,i*64:(i+1)*64]
        input_x = torch.tensor(x_img,dtype=torch.float32,requires_grad=True)
        output = self.input_linear(input_x)
        residual = output
        output = self.dropout(output)
        output, _ = self.gru(output)
        output = output + residual
        output = self.dropout(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.bn1(output)
        output = self.activation(output)
        output = self.fc2(output)
        output = self.bn2(output)
        return output


if __name__ == '__main__':
    lables = torch.tensor([0,0,0,0,0,0,0,0,0,0],dtype=torch.long)
    loss = nn.CrossEntropyLoss()
    model = FDGRU(1024, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data = torch.randn((10,4096,1))
    optimizer.zero_grad()
    output = model(data)
    loss_cal = loss(output, lables)
    loss_cal.backward()
    optimizer.step()