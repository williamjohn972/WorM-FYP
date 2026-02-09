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
from src.args import build_parser
from src.tasks import Tasks

def main():
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run(config)

if __name__ == "__main__":
    main()


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

from src.config import Config, TaskConfig


def config_from_args(args):
    # Convert enums
    mem_arch = Memory_Components[args.mem_architecture]
    task_embedding = Task_Embedding[args.task_embedding_type]
    show_task_time = Show_Task[args.show_task_time]
    projection_type = Projections[args.projection_type]

    task_list = [Tasks[t] for t in args.tasks]

    # Base config
    config = Config(
        root_folder=args.root_folder,
        run_name=args.run_name,
        mem_architecture=mem_arch,
        gpu=args.gpu,
        seed=args.seed,
    )

    # Override Sub Configs
    config.task_config = TaskConfig(task_list)
    config.execution_config.gen_test = args.gen_test

    config.resumption_config.resume = args.resume
    config.resumption_config.resume_epoch = args.resume_epoch

    config.model_config.img_size = args.img_size
    config.model_config.resize_img_size = args.resize_img_size
    config.model_config.num_input_channels = args.num_input_channels
    config.model_config.max_seq_len = args.max_seq_len
    config.model_config.use_cnn = args.use_cnn

    config.model_config.mem_input_size = args.mem_input_size
    config.model_config.mem_hidden_size = args.mem_hidden_size
    config.model_config.mem_num_layers = args.mem_num_layers
    config.model_config.trf_dim_ff = args.trf_dim_ff
    config.model_config.task_embedding_type = task_embedding
    config.model_config.show_task_time = show_task_time
    config.model_config.projection_type = projection_type

    config.train_config.lr = args.lr
    config.train_config.batch_size = args.batch_size
    config.train_config.num_epochs = args.num_epochs
    config.train_config.samples_per_task = args.samples_per_task
    config.train_config.test_interval = args.test_interval

    config.optimization_config.num_workers = args.num_workers
    config.optimization_config.use_extracted_feats = args.use_extracted_feats

    return config
