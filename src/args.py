import argparse
from src.model import Memory_Components, Task_Embedding, Show_Task, Projections
from src.tasks import Tasks


def build_parser():
    parser = argparse.ArgumentParser("WM Benchmark Training")

    # Path Config
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)

    # Execution Config
    parser.add_argument("--gen_test", action="store_true")

    # Resumption Config
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_epoch", type=int, default=None)

    # Model Config
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--resize_img_size", type=int, default=32)
    parser.add_argument("--num_input_channels", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=20)

    parser.add_argument("--no_cnn", action="store_false", dest="use_cnn")
    parser.set_defaults(use_cnn = True)

    parser.add_argument("--mem_architecture", type=str, default="GRU")
    parser.add_argument("--mem_input_size", type=int, default=512)
    parser.add_argument("--mem_hidden_size", type=int, default=96)
    parser.add_argument("--mem_num_layers", type=int, default=1)
    parser.add_argument("--trf_dim_ff", type=int, default=2048)
    parser.add_argument("--projection_type", type=str, default="LINEAR")


    parser.add_argument("--task_embedding_type", type=str, default="LEARNABLE")
    parser.add_argument("--show_task_time", type=str, default="ALL")

    # Train Config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--samples_per_task", type=int, default=4800)
    parser.add_argument("--test_interval", type=int, default=5)


    # Optimization Config
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_extracted_feats", action="store_true")


    # Tasks
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[t.name for t in Tasks],
        help="List of task names (e.g. VISUAL_SERIAL_RECALL CHANGE_DETECTION_COLOR)",
    )

    return parser
