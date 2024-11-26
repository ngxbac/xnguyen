import argparse

parser = argparse.ArgumentParser(description="MindBridge Configuration")
parser.add_argument(
    "--model_name",
    type=str,
    default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="data/nsd_openmind/",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--prompt_type",
    type=str,
    default="individual",
)
parser.add_argument(
    "--subject",
    type=int,
    default=None,
    choices=[1, 2, 3, 4, 5, 6, 7, 8],
    nargs="+",
    help="subj want to be load in the model",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size per GPU",
)
parser.add_argument(
    "--clip_variant",
    type=str,
    default="ViT-L/14",
    choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
    help="OpenAI clip variant",
)
parser.add_argument(
    "--norm_embs",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Do l2-norming of CLIP embeddings",
)
parser.add_argument(
    "--eval_interval",
    type=int,
    default=10,
    help="Evaluate the model every x epochs",
)
parser.add_argument(
    "--cl_checkpoint",
    type=str,
    default="none",
)
parser.add_argument("--aux_loss_factor", type=float, default=1.0)
parser.add_argument("--clip_loss_factor", type=float, default=1.0)
parser.add_argument("--l1_loss_factor", type=float, default=1.0)
parser.add_argument("--mse_loss_factor", type=float, default=1.0)
parser.add_argument("--mixup_pct", type=float, default=0.33)

parser.add_argument(
    "--clip_decoder",
    type=str,
    default="transformer",
)


# Training/Optimization parameters
parser.add_argument(
    "--use_fp16",
    action=argparse.BooleanOptionalAction,
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
    "--use_prior",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="""Whether or not
    to use half precision for training""",
)
parser.add_argument(
    "--epochs", default=20, type=int, help="Number of epochs of training."
)
parser.add_argument("--lr", default=2e-3, type=float, help=""" Learning rate""")
parser.add_argument("--min_lr", default=0.0, type=float)
parser.add_argument("--scheduler", default="cosine", type=str)

# Misc
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
    default=True,
    action=argparse.BooleanOptionalAction,
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

args = parser.parse_args()
