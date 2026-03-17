import threading
import queue
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
import wandb

class AsyncVisualizer(threading.Thread):
    def __init__(self, wandb_run, daemon=True):
        """
        Async visualizer for PCA + plotting + W&B logging.
        
        Args:
            wandb_run: W&B run object.
            daemon: set thread as daemon so it closes with main program.
        """
        super().__init__(daemon=daemon)
        self.wandb_run = wandb_run
        self.queue = queue.Queue()
        self._stop_signal = object()  # unique object to signal shutdown

    def submit(self, **kwargs):
        """
        Submit data for visualization.
        kwargs must include:
            epoch, encoder_embeddings_np, context_embeddings_np,
            actual_times, pred_times, all_risk, event_np, train_loss, val_c_index
        """
        self.queue.put(kwargs)

    def run(self):
        while True:
            item = self.queue.get()
            if item is self._stop_signal:
                break
            self._process(item)
            self.queue.task_done()

    def shutdown(self):
        """Signal the thread to stop and wait for queue to finish."""
        self.queue.put(self._stop_signal)
        self.join()

    def _process(self, item):
        """Internal method to compute PCA, plot, and log to W&B."""
        epoch = item['epoch']
        encoder_embeddings_np = item['encoder_embeddings_np']
        context_embeddings_np = item['context_embeddings_np']
        actual_times = item['actual_times']
        pred_times = item['pred_times']
        all_risk = item['all_risk']
        event_np = item['event_np']
        train_loss = item['train_loss']
        val_c_index = item.get('val_c_index', None)


        # --- PCA ---
        embedding_2d = PCA(n_components=2, random_state=42).fit_transform(encoder_embeddings_np)
        context_embedding_2d = PCA(n_components=2, random_state=42).fit_transform(context_embeddings_np)

        # --- Plotting ---
        fig = plt.figure(figsize=(12,12))
        fig.suptitle(f"Cox Model Predictions at Epoch {epoch}", fontsize=16)
        gs = fig.add_gridspec(3, 2)

        # Scatter plot
        ax1 = fig.add_subplot(gs[0,0])
        ax1.scatter(actual_times, pred_times, alpha=0.3, s=8)
        max_t = max(np.max(actual_times), np.max(pred_times))
        ax1.plot([0, max_t], [0, max_t], 'r--', label="Perfect prediction")
        ax1.set_xlabel("Actual Time-to-Event")
        ax1.set_ylabel("Predicted Time-to-Event")
        ax1.set_title(f"Prediction vs Actual")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Hexbin density
        ax2 = fig.add_subplot(gs[0,1])
        hb = ax2.hexbin(actual_times, pred_times, gridsize=50, cmap='Blues')
        fig.colorbar(hb, ax=ax2, label="Density")
        ax2.set_xlabel("Actual Time")
        ax2.set_ylabel("Predicted Time")
        ax2.set_title("Prediction Density")
        ax2.grid(alpha=0.2)

        # Risk distribution
        ax3 = fig.add_subplot(gs[1,:])
        ax3.hist(all_risk, bins=40, color="steelblue", alpha=0.75, edgecolor="black")
        mean_risk = np.mean(all_risk)
        median_risk = np.median(all_risk)
        ax3.axvline(mean_risk, color='red', linestyle='--', label=f"Mean = {mean_risk:.2f}")
        ax3.axvline(median_risk, color='green', linestyle='--', label=f"Median = {median_risk:.2f}")
        ax3.set_xlabel("Predicted Risk Score")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Risk Score Distribution")
        ax3.legend()
        ax3.grid(alpha=0.3)

        colors = np.where(event_np == 1, "red", "blue")

        # Encoder embedding
        ax4 = fig.add_subplot(gs[2,0])
        ax4.scatter(embedding_2d[:,0], embedding_2d[:,1], c=colors, alpha=0.7, s=8)
        ax4.set_title("Encoder Embedding PCA")
        ax4.set_xlabel("PCA 1")
        ax4.set_ylabel("PCA 2")

        # Context embedding
        ax5 = fig.add_subplot(gs[2,1])
        ax5.scatter(context_embedding_2d[:,0], context_embedding_2d[:,1], c=colors, alpha=0.7, s=8)
        ax5.set_title("Context Embedding PCA")
        ax5.set_xlabel("PCA 1")
        ax5.set_ylabel("PCA 2")

        plt.tight_layout(rect=[0,0,1,0.96])

        # W&B logging (C-index computation)
        # val_c_index = CIndex.calculate(torch.tensor(all_risk), torch.tensor(actual_times), torch.tensor(event_np))
        self.wandb_run.log({
            "Cox_epoch": epoch,
            "Cox_train_loss": train_loss,
            "Cox_val_c_index": val_c_index,
            "prediction_plot": wandb.Image(fig)
        })

        plt.close(fig)