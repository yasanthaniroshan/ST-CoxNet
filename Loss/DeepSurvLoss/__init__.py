import torch
import torch.nn as nn

class DeepSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_h_pred, durations, events):
        """
        log_h_pred: [B, 1] Predicted log-hazard ratio from your model
        durations:  [B] Time remaining until AFib (T)
        events:     [B] 1 if AFib occurs (Even/Cont records), 0 if SR (Odd)
        """
        # Sort by duration in descending order (Longest duration first)
        # This is a requirement for the partial likelihood calculation
        durations, idx = torch.sort(durations, descending=True)
        log_h_pred = log_h_pred[idx]
        events = events[idx]

        # Log-sum-exp trick for numerical stability
        # We calculate log(sum(exp(h_j))) for all j in the risk set
        log_risk_set_sum = torch.logcumsumexp(log_h_pred, dim=0)
        
        # We only care about terms where an event actually happened (E=1)
        # Loss = - sum_{i:E_i=1} (h_i - log(sum_{j:T_j >= T_i} exp(h_j)))
        partial_log_likelihood = (log_h_pred.flatten() - log_risk_set_sum.flatten()) * events
        
        # We return the negative of the sum to minimize it
        return -torch.sum(partial_log_likelihood) / (torch.sum(events) + 1e-7)