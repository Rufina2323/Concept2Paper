import argparse
import logging
import os
import random

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from src.link_prediction.pipeline import TrainingPipeline
from src.utils import load_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Link prediction training pipeline")
    parser.add_argument("--config", default="configs/model_training.yaml")
    parser.add_argument(
        "--mode",
        choices=["train", "rank", "all"],
        default="all",
        help="train: fit + save models; rank: load + rank; all: both (default)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    cfg = load_config(args.config)

    pipeline = TrainingPipeline(cfg)

    if args.mode == "train":
        pipeline.train()
    elif args.mode == "rank":
        pipeline.rank()
    else:
        pipeline.run()

    logger.info("Done.")


if __name__ == "__main__":
    main()
