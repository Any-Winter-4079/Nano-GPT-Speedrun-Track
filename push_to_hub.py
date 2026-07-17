import os
import re
import argparse
from pathlib import Path
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import delete_repo, create_repo, upload_folder

# pushing outside of training ensures we have control over whether
# we want to overwrite the existing max_checkpoints_to_keep, store
# the current checkpoints (on the Hub) elsewhere before overwriting
# the HF repo, or even add the local checkpoints to the Hub without 
# deleting old ones if we skip calling the delete_repo function
HF_USER = os.environ.get("hf_user")
HF_TOKEN = os.environ.get("hf_token")
CONFIGS_AND_LOGS_DIR = Path("./configs_and_logs")
CHECKPOINTS_DIR = Path("./checkpoints")
CONFIG_FILE_SUFFIX = "config.txt"
LOG_FILE_SUFFIX = ".txt"

def parse_args():
    parser = argparse.ArgumentParser(description="Push a training run directory to Hugging Face Hub")
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="run timestamp to upload, e.g. 20260211_180749; if omitted, latest available is used",
    )
    parser.add_argument(
        "--repo-prefix",
        type=str,
        default="nanogpt",
        help="repo name prefix before timestamp",
    )
    parser.add_argument(
        "--with-metrics-in-name",
        action="store_true",
        # argparse store_true flag:
        # default is False, passing --with-metrics-in-name sets it to True
        help="append ddp world size, val loss, and train minutes to repo name when available",
    )
    return parser.parse_args()

def discover_latest_timestamp() -> str:
    candidates = set()
    for root in (CONFIGS_AND_LOGS_DIR, CHECKPOINTS_DIR):
        if root.is_dir():
            for path in root.iterdir():
                if path.is_dir():
                    candidates.add(path.name)

    if not candidates:
        raise FileNotFoundError(f"no timestamp directories found in {CONFIGS_AND_LOGS_DIR} or {CHECKPOINTS_DIR}")

    # timestamp format is sortable lexicographically
    return sorted(candidates)[-1]

def find_first_file(folder: Path, suffix: str) -> Path | None:
    if not folder.is_dir():
        return None
    matches = sorted(folder.glob(f"*{suffix}"))
    return matches[0] if matches else None

def parse_run_metadata(timestamp: str) -> dict:
    config_dir = CONFIGS_AND_LOGS_DIR / timestamp
    metadata = {
        "ddp_world_size": None,
        "val_loss": None,
        "total_train_time_min": None,
    }

    config_path = find_first_file(config_dir, CONFIG_FILE_SUFFIX)
    if config_path is not None:
        config_text = config_path.read_text(encoding="utf-8", errors="ignore")
        world_size_match = re.search(r"ddp world size:\s*(\d+)", config_text)
        if world_size_match:
            metadata["ddp_world_size"] = int(world_size_match.group(1))

    log_path = find_first_file(config_dir, LOG_FILE_SUFFIX)
    if log_path is not None:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")

        val_matches = re.findall(r"val loss:\s*([0-9]+(?:\.[0-9]+)?)", log_text)
        if val_matches:
            metadata["val_loss"] = float(val_matches[-1])

        train_time_matches = re.findall(
            r"total train time:\s*([0-9]+(?:\.[0-9]+)?)\s*min",
            log_text,
        )
        if train_time_matches:
            metadata["total_train_time_min"] = float(train_time_matches[-1])

    return metadata

def format_metric_value(value: float, precision: int) -> str:
    return f"{value:.{precision}f}".replace(".", "p")

def build_repo_id(
    hf_user: str,
    repo_prefix: str,
    timestamp: str,
    with_metrics_in_name: bool,
) -> str:
    name_parts = [repo_prefix, timestamp]

    if with_metrics_in_name:
        metadata = parse_run_metadata(timestamp)
        if metadata["ddp_world_size"] is not None:
            name_parts.append(f"g{metadata['ddp_world_size']}")
        if metadata["val_loss"] is not None:
            name_parts.append(f"vl{format_metric_value(metadata['val_loss'], 4)}")
        if metadata["total_train_time_min"] is not None:
            name_parts.append(f"t{format_metric_value(metadata['total_train_time_min'], 2)}m")

    repo_name = "_".join(name_parts)
    return f"{hf_user}/{repo_name}"

def upload_dir(repo_id: str, folder_path: Path, commit_message: str) -> None:
    if not folder_path.is_dir():
        print(f"skipping missing directory: {folder_path}")
        return

    upload_folder(
        repo_id=repo_id,
        folder_path=str(folder_path),
        commit_message=commit_message,
        token=HF_TOKEN,
        repo_type="model",
    )
    print(f"successfully pushed {folder_path}")

def push_to_hub(repo_id: str, timestamp: str):
    try:
        upload_dir(repo_id, CHECKPOINTS_DIR / timestamp, "checkpoints")
        upload_dir(repo_id, CONFIGS_AND_LOGS_DIR / timestamp, "config and logs")
        return True
    except Exception as e:
        print(f"failed to push files to HF: {e}")
        return False

def main():
    args = parse_args()

    timestamp = args.timestamp or discover_latest_timestamp()
    hub_repo_id = build_repo_id(
        hf_user=HF_USER,
        repo_prefix=args.repo_prefix,
        timestamp=timestamp,
        with_metrics_in_name=args.with_metrics_in_name,
    )

    print(f"timestamp: {timestamp}")
    print(f"hub repo id: {hub_repo_id}")

    try:
        delete_repo(repo_id=hub_repo_id, token=HF_TOKEN, repo_type="model")
        print(f"repository deleted: {hub_repo_id}")
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            print(f"repository {hub_repo_id} did not exist, skipping deletion")
        else:
            raise

    create_repo(repo_id=hub_repo_id, exist_ok=True, token=HF_TOKEN, repo_type="model")
    print(f"repository created: {hub_repo_id}")
    push_to_hub(repo_id=hub_repo_id, timestamp=timestamp)
    print(f"repository pushed to HF repo id: {hub_repo_id}")

if __name__ == "__main__":
    main()
