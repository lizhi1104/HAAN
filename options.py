import argparse
import uuid

parser = argparse.ArgumentParser(description="HAAN")
parser.add_argument(
    "--dataset",
    choices=["FineAction", "FineGym"],
    required=True,
    help="which dataset to run",
)
parser.add_argument(
    "--exp-name",
    type=str,
    default=str(uuid.uuid4()),
    help="experiment name (default: random uuid)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="output",
    help="the directory to store output models/logs",
)
parser.add_argument(
    "--evaluation-only", action="store_true", help="whether to run evaluation only"
)
parser.add_argument(
    "--input-models-dir", type=str, default="", help="the directory to read models"
)
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument(
    "--device", type=str, default="cuda", help="device type (default: cuda)"
)
