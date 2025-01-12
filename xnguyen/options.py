import argparse

parser = argparse.ArgumentParser(description="MindBridge Configuration")
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size per GPU",
)
parser.add_argument(
    "--eval_interval",
    type=int,
    default=1,
    help="Evaluate the model every x epochs",
)
parser.add_argument(
    "--use_fp16",
    # action=argparse.BooleanOptionalAction,
    action="store_true",
    default=False,
    help="""Whether or not
    to use half precision for training""",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="""Initial value of the
    weight decay.""",
)
parser.add_argument(
    "--epochs", default=20, type=int, help="Number of epochs of training."
)
parser.add_argument("--lr", default=2e-3, type=float, help=""" Learning rate""")
parser.add_argument("--min_lr", default=0.0, type=float)
parser.add_argument("--scheduler", default="cosine", type=str)
parser.add_argument("--resume", type=str, default="")
parser.add_argument(
    "--output_dir",
    default=".",
    type=str,
    help="Path to save logs and checkpoints.",
)
parser.add_argument(
    "--saveckp_freq",
    default=5,
    type=int,
    help="Save checkpoint every x epochs.",
)
parser.add_argument("--seed", default=216, type=int, help="Random seed.")
parser.add_argument(
    "--num_workers",
    default=4,
    type=int,
    help="Number of data loading workers per GPU.",
)
parser.add_argument(
    "--distributed",
    # default=True,
    # action=argparse.BooleanOptionalAction,
    action="store_true",
)
parser.add_argument(
    "--deepspeed",
    # default=True,
    # action=argparse.BooleanOptionalAction,
    action="store_true",
)
parser.add_argument(
    "--dist_url",
    default="env://",
    type=str,
    help="""url used to set up
    distributed training; see https://pytorch.org/docs/stable/distributed.html""",
)
parser.add_argument(
    "--local_rank",
    default=0,
    type=int,
    help="Please ignore and do not set this argument.",
)

# args = parser.parse_args()
