import torch
import torch.nn as nn
from scipy.special.cython_special import sph_harm
from torch.nn import TransformerEncoderLayer


class SiT(nn.Module):
    """
    The implementation for SiT(Signal Transformer) for bearing fault diagnosis.
    """
    def __init__(self, num_classes, split_length=32, data_length=1024, hidden_dim=512):
        super().__init__()
        self.num_patches = data_length // split_length
        self.split_length = split_length
        self.positional_encoding = nn.Parameter(torch.randn(1,self.num_patches, hidden_dim))
        self.linear_encoder = nn.Sequential(
            nn.Linear(split_length, hidden_dim),
            nn.GELU(),
        )
        self.transformer_encoder_layer = TransformerEncoderLayer(hidden_dim, nhead=8, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
        self.mlp = nn.Sequential(
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        """
         input: x (batch_size, length, 1)
        """
        B, L, D = x.shape
        # Split the input first
        x_reshape = x.reshape(B, self.num_patches, self.split_length)
        # x = (B,num_patch, split_length)
        x_reshape = self.linear_encoder(x_reshape)
        # x = (B,num_patch,hidden_dim)
        x_reshape += self.positional_encoding
        features = self.encoder(x_reshape)
        # features = (B,num_patch,hidden_dim)
        output = self.mlp(features[:,:,-1].squeeze(-1))
        return output

