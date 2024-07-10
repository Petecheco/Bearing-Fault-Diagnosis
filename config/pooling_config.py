import torch.nn as nn
POOLING_DICT = {
    "max": nn.MaxPool1d,
    "average": nn.AvgPool1d
}