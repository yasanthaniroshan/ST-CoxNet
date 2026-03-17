import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, L]
        """
        # Squeeze: global avg pooling along time
        w = x.mean(dim=2)  # [B, C]

        # Excitation: small MLP
        w = F.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))  # [B, C]
        
        # Scale: multiply original feature map
        x = x * w.unsqueeze(-1)  # [B, C, L]
        return x