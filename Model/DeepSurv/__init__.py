
from torch import nn
from Model.CoxHead.Base import CoxHead
import torch

class DeepSurvCox(nn.Module):
    def __init__(self, encoder, context, context_dim:int):
        super().__init__()
        self.encoder = encoder
        self.context = context
        self.cox_head = CoxHead(context_dim)

    def forward(self, rr_windows):
        B, T, W = rr_windows.shape
        z_list = []

        # REMOVE torch.no_grad() from here!
        for t in range(T):
            z_t = self.encoder(rr_windows[:, t, :])
            z_list.append(z_t)
        
        z_seq = torch.stack(z_list, dim=1)
        c_seq = self.context(z_seq)
        c_last = c_seq[:, -1, :]
        
        risk = self.cox_head(c_last).squeeze(-1)
        return risk, c_seq, z_seq