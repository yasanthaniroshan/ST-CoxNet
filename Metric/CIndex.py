
import numpy as np

class CIndex:
    @staticmethod
    def calculate(risk, duration, event):
        risk = risk.detach().cpu().numpy().reshape(-1)
        duration = duration.detach().cpu().numpy().reshape(-1)
        event = event.detach().cpu().numpy().reshape(-1)

        n = len(risk)
        if n <= 1:
            return 0.0

        dur_i = duration[:, None]
        dur_j = duration[None, :]
        event_i = event[:, None]

        comparable = (event_i == 1) & (dur_i < dur_j)

        risk_i = risk[:, None]
        risk_j = risk[None, :]

        concordant = (risk_i > risk_j) & comparable
        tied = (risk_i == risk_j) & comparable

        num = concordant.sum() + 0.5 * tied.sum()
        den = comparable.sum()

        return float(num) / float(den) if den > 0 else 0.0
    