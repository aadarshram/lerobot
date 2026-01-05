"""
Script to load MT50 dataset from HuggingFace, split it into separate datasets
for each of the 50 tasks, and push each to HuggingFace or save locally.

Usage:
    python scripts/split_mt50_by_task.py \
        --mt50-repo-id <original_mt50_repo> \
        --target-username <your_hf_username> \
        --push-to-hub

## Prerequisites

1. Login to HuggingFace (if pushing to hub):
   ```bash
   huggingface-cli login
   ```

## Usage

To split the dataset and save locally without pushing to HuggingFace:

```bash
python scripts/split_mt50_by_task.py \
    --mt50-repo-id lerobot/metaworld_mt50 \
    --target-username your_hf_username
```

To split and push all task datasets to your HuggingFace account:

```bash
python scripts/split_mt50_by_task.py \
    --mt50-repo-id lerobot/metaworld_mt50 \
    --target-username your_hf_username \
    --push-to-hub
```

### Advanced Options

```bash
python scripts/split_mt50_by_task.py \
    --mt50-repo-id lerobot/metaworld_mt50 \
    --target-username your_hf_username \
    --push-to-hub \
    --private \
    --license apache-2.0 \
    --tags robotics manipulation metaworld \
    --root /path/to/cache \
    --output-dir /path/to/output
```

## Arguments

- `--mt50-repo-id`: Repository ID of the MT50 dataset on HuggingFace (required)
- `--target-username`: Your HuggingFace username (required)
- `--push-to-hub`: Upload datasets to HuggingFace Hub (optional flag)
- `--private`: Make uploaded datasets private (optional flag)
- `--license`: License for the datasets (default: apache-2.0)
- `--tags`: Additional tags for the datasets (space-separated)
- `--root`: Local cache directory for source dataset
- `--output-dir`: Output directory for split datasets

## Output

### Local Storage

When not using `--push-to-hub`, datasets are saved to:
```
~/.cache/huggingface/lerobot/mt50_split/
├── assembly-v2/
├── basketball-v2/
├── bin-picking-v2/
├── ...
└── window-open-v2/
```
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Split MT50 dataset by task and push to HuggingFace"
    )
    parser.add_argument(
        "--mt50-repo-id",
        type=str,
        required=True,
        help="Repository ID of the MT50 dataset (e.g., 'lerobot/metaworld_mt50')",
    )
    parser.add_argument(
        "--target-username",
        type=str,
        required=True,
        help="Your HuggingFace username where task datasets will be pushed",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory to use for downloading/writing datasets (default: ~/.cache/huggingface/lerobot/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for split datasets (default: ~/.cache/huggingface/lerobot/)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the split datasets to HuggingFace Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded datasets private",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="License for the datasets (default: apache-2.0)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Additional tags for the datasets",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download videos (can be very large, disabled by default)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached dataset before downloading (useful if download was corrupted)",
    )

    args = parser.parse_args()

    # Determine root path
    if args.root:
        root_path = Path(args.root)
    else:
        root_path = HF_LEROBOT_HOME / args.mt50_repo_id
    
    # Clear cache if requested
    if args.clear_cache and root_path.exists():
        logger.info(f"Clearing cached dataset at {root_path}...")
        import shutil
        shutil.rmtree(root_path)
        logger.info("Cache cleared!")

    # Load the MT50 dataset
    logger.info(f"Loading MT50 dataset from {args.mt50_repo_id}...")
    logger.info(f"Cache location: {root_path}")
    logger.info(f"Download videos: {args.download_videos}")
    
    try:
        mt50_dataset = LeRobotDataset(
            repo_id=args.mt50_repo_id, 
            root=args.root,
            download_videos=args.download_videos
        )
    except Exception as e:
        logger.error(f"\n❌ Failed to load dataset!")
        logger.error(f"Error: {type(e).__name__}: {str(e)[:200]}")
        logger.info("\n" + "="*70)
        logger.info("🔧 TROUBLESHOOTING - Corrupted Download")
        logger.info("="*70)
        logger.info("\nThe dataset appears to be corrupted. This happens when downloads")
        logger.info("are interrupted. To fix this, run:")
        logger.info(f"\n  python {Path(__file__).relative_to(Path.cwd())} \\")
        logger.info(f"    --mt50-repo-id {args.mt50_repo_id} \\")
        logger.info(f"    --target-username {args.target_username} \\")
        logger.info("    --clear-cache")
        logger.info("\nThis will delete the corrupted cache and re-download from scratch.")
        logger.info("="*70 + "\n")
        raise

    logger.info(f"Loaded dataset with {mt50_dataset.meta.total_episodes} episodes")
    logger.info(f"Total tasks in dataset: {mt50_dataset.meta.total_tasks}")
    logger.info(f"Tasks: {list(mt50_dataset.meta.tasks.index)}")

    # Group episodes by task
    logger.info("Grouping episodes by task...")
    task_to_episodes = defaultdict(list)
    
    # Load episodes metadata to get task information
    episodes_df = mt50_dataset.meta.episodes
    
    for episode_idx in range(mt50_dataset.meta.total_episodes):
        episode_tasks = episodes_df["tasks"][episode_idx]
        # Each episode should have one task in MT50
        if len(episode_tasks) != 1:
            logger.warning(
                f"Episode {episode_idx} has {len(episode_tasks)} tasks (expected 1). "
                f"Using first task: {episode_tasks[0]}"
            )
        task = episode_tasks[0]
        task_to_episodes[task].append(episode_idx)

    logger.info(f"Found {len(task_to_episodes)} unique tasks")
    
    # Print task distribution
    for task, episodes in sorted(task_to_episodes.items()):
        logger.info(f"  {task}: {len(episodes)} episodes")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = HF_LEROBOT_HOME / "mt50_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create default tags
    tags = ["LeRobot", "MT50", "MetaWorld"]
    if args.tags:
        tags.extend(args.tags)

    # Split dataset by task and push to hub
    logger.info("Splitting dataset by task...")
    
    for task, episode_indices in sorted(task_to_episodes.items()):
        # Clean task name for repo_id (replace spaces, special chars with hyphens)
        task_clean = task.lower().replace(" ", "-").replace("_", "-")
        task_repo_id = f"{args.target_username}/mt50-{task_clean}"
        
        logger.info(f"\nProcessing task: {task}")
        logger.info(f"  Episodes: {len(episode_indices)}")
        logger.info(f"  Target repo: {task_repo_id}")
        
        # Create split with just this task's episodes
        splits = {task: episode_indices}
        
        task_output_dir = output_dir / task_clean
        
        # Use split_dataset to create the task-specific dataset
        split_datasets = split_dataset(
            dataset=mt50_dataset,
            splits=splits,
            output_dir=task_output_dir.parent,
        )
        
        # Get the split dataset (there's only one in this case)
        task_dataset = split_datasets[task]
        
        logger.info(f"  Created dataset with {task_dataset.meta.total_episodes} episodes, "
                   f"{task_dataset.meta.total_frames} frames")
        
        if args.push_to_hub:
            logger.info(f"  Pushing to hub as {task_repo_id}...")
            
            # Load the dataset with the correct repo_id for pushing
            dataset_to_push = LeRobotDataset(
                repo_id=task_repo_id,
                root=task_dataset.root,
            )
            
            # Push to hub
            dataset_to_push.push_to_hub(
                tags=tags,
                license=args.license,
                private=args.private,
            )
            
            logger.info(f"  ✓ Successfully pushed to {task_repo_id}")
        else:
            logger.info(f"  Saved locally to {task_dataset.root}")

    logger.info("\n" + "="*60)
    logger.info("✓ All tasks processed successfully!")
    logger.info(f"Total tasks: {len(task_to_episodes)}")
    
    if args.push_to_hub:
        logger.info(f"\nAll datasets pushed to https://huggingface.co/{args.target_username}")
        logger.info("Repository IDs:")
        for task in sorted(task_to_episodes.keys()):
            task_clean = task.lower().replace(" ", "-").replace("_", "-")
            logger.info(f"  - {args.target_username}/mt50-{task_clean}")
    else:
        logger.info(f"\nAll datasets saved to {output_dir}")
        logger.info("Use --push-to-hub to upload them to HuggingFace")


if __name__ == "__main__":
    main()
