import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, kernel_size=100, out_channels=32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 100)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(64, 64, 100)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.average_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, dropoutrate):
        dropout = nn.Dropout(dropoutrate)
        out = self.conv1(x)
        out = dropout(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.average_pool(out)
        return out


class CNN1(BaseCNN):
    def __init__(self):
        super(CNN1, self).__init__()


class CNN2(BaseCNN):
    def __init__(self):
        super(CNN2, self).__init__()


class CNN3(BaseCNN):
    def __init__(self):
        super(CNN3, self).__init__()


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, hidden_dim, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Linear projection layer for Q,K,V
        self.query_projection = nn.Linear(input_dim, hidden_dim * num_heads)
        self.key_projection = nn.Linear(input_dim, hidden_dim * num_heads)
        self.value_projection = nn.Linear(input_dim, hidden_dim * num_heads)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(hidden_dim * num_heads, input_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.query_projection(query).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        key = self.key_projection(key).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)
        value = self.value_projection(value).view(batch_size, -1, self.num_heads, self.hidden_dim).transpose(1, 2)

        # Score = QK^T
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))


        # Dropout for attention score for better robustness
        attention_probs = self.attention_dropout(attention_scores)

        # Attention = (QK^T)V
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim * self.num_heads)

        # Output projection
        output = self.output_projection(attention_output)

        return output


class AttentionModule(nn.Module):
    def __init__(self, num_heads, input_dim):
        super(AttentionModule, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=192,num_heads=num_heads)
        self.fc = nn.Linear(in_features=192,out_features=3)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x1, x2, x3):
        out = torch.cat((x1,x2,x3),dim=1)
        out = torch.transpose(out,1,2)
        attention_weight = self.multihead_attention(out, out, out)
        attention_weight = attention_weight[0]
        attention_weight = self.fc(attention_weight)
        attention_weight1 = self.softmax(attention_weight)
        return attention_weight1


class MSAFCN(nn.Module):
    def __init__(self):
        super(MSAFCN, self).__init__()
        self.multiscale1 = nn.AvgPool1d(2)
        self.multiscale2 = nn.AvgPool1d(3)
        self.multiscale3 = nn.AvgPool1d(4)
        self.con1 = CNN1()
        self.con2 = CNN2()
        self.con3 = CNN3()
        self.attention = AttentionModule(num_heads=8,input_dim=3)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(64, 100)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(100, 10)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x, drop1, drop2, drop3):
        out1 = self.multiscale1(x)
        out2 = self.multiscale2(x)
        out3 = self.multiscale3(x)
        out1 = self.con1(out1, drop1)
        out2 = self.con2(out2, drop2)
        out3 = self.con3(out3, drop3)
        attention_weight = self.attention(out1, out2, out3)
        weighted_output1 = out1 * attention_weight[:, :, 0].unsqueeze(1)
        weighted_output2 = out2 * attention_weight[:, :, 1].unsqueeze(1)
        weighted_output3 = out3 * attention_weight[:, :, 2].unsqueeze(1)
        out = weighted_output1 + weighted_output2 + weighted_output3
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

