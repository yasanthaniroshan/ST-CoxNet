import numpy as np
import matplotlib.pyplot as plt
import torch

epoch = 1

all_risk = [torch.rand(100) for _ in range(10)]
all_risk = torch.cat(all_risk)

actual_times = np.random.rand(100) * 10
pred_times = actual_times + np.random.normal(0, 2, size=100)

fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(2, 2)

# ---- Plot 1 : Scatter ----
ax1 = fig.add_subplot(gs[0,0])
ax1.scatter(actual_times, pred_times, alpha=0.3, s=8)

max_t = max(np.max(actual_times), np.max(pred_times))
ax1.plot([0, max_t], [0, max_t], 'r--', label="Perfect prediction")

ax1.set_xlabel("Actual Time-to-Event")
ax1.set_ylabel("Predicted Time-to-Event")
ax1.set_title(f"Prediction vs Actual (Epoch {epoch})")
ax1.legend()
ax1.grid(alpha=0.3)

# ---- Plot 2 : Hexbin Density ----
ax2 = fig.add_subplot(gs[0,1])
hb = ax2.hexbin(actual_times, pred_times, gridsize=50, cmap='Blues')
fig.colorbar(hb, ax=ax2, label="Density")

ax2.set_xlabel("Actual Time")
ax2.set_ylabel("Predicted Time")
ax2.set_title("Prediction Density")
ax2.grid(alpha=0.2)

# ---- Plot 3 : Risk Distribution (Full Width) ----
ax3 = fig.add_subplot(gs[1,:])

risk_np = all_risk.numpy()

ax3.hist(risk_np, bins=40, color="steelblue", alpha=0.75, edgecolor="black")

# Add statistics
mean_risk = np.mean(risk_np)
median_risk = np.median(risk_np)

ax3.axvline(mean_risk, color='red', linestyle='--', label=f"Mean = {mean_risk:.2f}")
ax3.axvline(median_risk, color='green', linestyle='--', label=f"Median = {median_risk:.2f}")

ax3.set_xlabel("Predicted Risk Score")
ax3.set_ylabel("Frequency")
ax3.set_title("Risk Score Distribution")
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot.png", dpi=300)
plt.show()