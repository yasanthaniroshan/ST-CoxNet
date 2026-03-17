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


class ImprovedDeepSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_h_pred, durations, events):
        """
        log_h_pred: [B, 1] Predicted log-hazard ratio from your model
        durations:  [B] Time remaining until AFib
        events:     [B] 1 if AFib occurs, 0 if censored/SR
        """
        # Flatten tensors for easier broadcasting
        log_h_pred = log_h_pred.view(-1)
        durations = durations.view(-1)
        events = events.view(-1).bool()

        # We only compute the partial likelihood for patients with an observed event
        event_times = durations[events]
        event_scores = log_h_pred[events]

        # Edge case: No events in this batch
        if event_times.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=log_h_pred.device)

        # Build a Risk Matrix [Num_events, Batch_size]
        # risk_mask[i, j] is True if patient j is still at risk at event_times[i]
        # This elegantly handles ties: if T_i == T_j, both are at risk for each other
        risk_mask = durations.unsqueeze(0) >= event_times.unsqueeze(1)

        # Expand the predicted risk scores to match the mask dimensions
        expanded_scores = log_h_pred.unsqueeze(0).expand(event_times.size(0), -1)

        # Mask out patients NOT in the risk set by setting their score to -infinity
        # exp(-infinity) = 0, so they won't contribute to the logsumexp
        masked_scores = expanded_scores.masked_fill(~risk_mask, -1e9)

        # Calculate log(sum(exp(h_j))) for the valid risk sets
        log_risk_set_sum = torch.logsumexp(masked_scores, dim=1)

        # Compute the log partial likelihood
        log_likelihood = event_scores - log_risk_set_sum

        # Return the negative mean log partial likelihood
        return -log_likelihood.mean()