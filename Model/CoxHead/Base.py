from torch import nn

class CoxHead(nn.Module):
    def __init__(self, context_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, c_last):
        return self.net(c_last)
