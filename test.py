import random
import wandb
import hydra
import torch
from omegaconf import DictConfig

from Utils.Logger import get_logger
from Utils.Pipeline import build_pipeline
from omegaconf import OmegaConf
from dotenv import load_dotenv
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings('default')
logging.captureWarnings(True)
load_dotenv()  # Load environment variables from .env file

@hydra.main(config_path="conf", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    """
    Lightweight debug / quick-run entry that trains without WandB.
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    try:
        wandb.login()
        run = wandb.init(
            entity="eml-labs",
            project="PAF Prediction - CPC",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"CPCPreModel_Run_{wandb.util.generate_id()}",
            tags=["CPCPreModel", "Experiment"]
        )

        logger = get_logger(run=run)    
        pipeline = build_pipeline(cfg, logger=logger)
        trainer = pipeline.trainer
        pbar = tqdm(range(cfg.trainer.epochs), desc="Training Epochs")
        for epoch in pbar:
            avg_loss, avg_losses = trainer.train_epoch(pipeline.train_loader)
            pbar_dict = {
                "Epoch": epoch + 1,
                "Total Loss": avg_loss,
                **{f"Pred {i}": l for i, l in enumerate(avg_losses)}
            }
            pbar.set_postfix(pbar_dict)
            logger.log(pbar_dict)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        torch.save(pipeline.model.state_dict(), "first_stage_model.pth")
        artifact = wandb.Artifact("first_stage_model", type="model")
        artifact.add_file("first_stage_model.pth")
        run.log_artifact(artifact)
        if logger is not None:
            logger.info("Model checkpoint saved and logged to WandB.")
        if 'run' in locals():
            run.finish()
        


if __name__ == "__main__":
    main()
