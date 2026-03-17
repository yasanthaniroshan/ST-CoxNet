from omegaconf import DictConfig, OmegaConf
import hydra
import time

@hydra.main(version_base=None,config_path="conf", config_name="config_irida_hrv.yaml")
def my_app(cfg: DictConfig) -> None:
    print(cfg.cpc.batch_size)
    print("Done!")

if __name__ == "__main__":
    my_app()