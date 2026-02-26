import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from src.utils.seed import set_seed

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    set_seed(cfg.seed)

    device = torch.device(cfg.trainer.device)

    model = instantiate(cfg.model)
    model.to(device)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    print("Model and optimizer initialized successfully.")

if __name__ == "__main__":
    main()
