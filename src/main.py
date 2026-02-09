import os 
import wandb 
import torch
import numpy as np 
import random

from src.model import Task_Embedding, Show_Task, Memory_Components, Projections
from src.config import Config 
from src.utils.data_utils import build_dataloaders
from src.utils.logger import get_logger
from src.train import Trainer
from src.model import WM_Model


def run(config: Config):
    
    # First we create seed 
    seed_everything(config.optimization_config.seed)

    # Build / Load Dataloaders
    data_loaders = build_dataloaders(config)

    # Create Logger
    os.makedirs(config.path_config.log_folder, exist_ok=True)
    logger = get_logger("log", log_file_path=os.path.join(config.path_config.log_folder, "train.log"))

    # Create Model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WM_Model(config.model_config, device=device)

    # Create Trainer
    trainer = Trainer(
        model=model,
        dataloaders=data_loaders,
        config=config,
        device=device,
        logger=logger,
        wandb=None
    )

    trainer.fit()



def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True