#!/usr/bin/env python3
"""
Merge individual MT10 task datasets into a unified metaworld_mt10 dataset.

This script downloads all MT10 task datasets from HuggingFace Hub,
merges them into a single unified dataset, and pushes the result back to the hub.

Usage:
    python src/lerobot/scripts/merge_mt10_datasets.py \
        --username aadarshram \
        --output-repo-id aadarshram/metaworld_mt10 \
        --push-to-hub
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MT10 tasks (v3 naming)
MT10_TASKS = [
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-open-v3",
    "drawer-close-v3", "button-press-topdown-v3", "peg-insert-side-v3",
    "window-open-v3", "window-close-v3"
]


def merge_mt10_datasets(
    username: str,
    output_repo_id: str,
    push_to_hub: bool = False,
    root: str | None = None,
):
    """
    Download and merge all MT10 task datasets into a unified dataset.
    
    Args:
        username: HuggingFace username where individual task datasets are stored
        output_repo_id: Repository ID for the merged dataset (e.g., "username/metaworld_mt10")
        push_to_hub: Whether to push the merged dataset to HuggingFace Hub
        root: Root directory for dataset storage (default: ~/.cache/huggingface/lerobot/)
    """
    # Construct repo IDs for all MT10 tasks
    repo_ids = [f"{username}/metaworld-{task}" for task in MT10_TASKS]
    
    logger.info(f"=" * 70)
    logger.info(f"Merging MT10 Datasets")
    logger.info(f"=" * 70)
    logger.info(f"Source username: {username}")
    logger.info(f"Output repo ID: {output_repo_id}")
    logger.info(f"Number of tasks: {len(MT10_TASKS)}")
    logger.info(f"Tasks to merge:")
    for i, task in enumerate(MT10_TASKS, 1):
        logger.info(f"  {i:2d}. {task:30s} -> {repo_ids[i-1]}")
    logger.info("")
    
    # Load all datasets
    logger.info("Loading datasets from HuggingFace Hub...")
    datasets = []
    total_episodes = 0
    total_frames = 0
    
    for i, repo_id in enumerate(repo_ids, 1):
        try:
            logger.info(f"[{i}/{len(repo_ids)}] Loading {repo_id}...")
            dataset = LeRobotDataset(repo_id, root=root)
            datasets.append(dataset)
            
            logger.info(f"  ✓ Loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
            total_episodes += dataset.num_episodes
            total_frames += dataset.num_frames
            
        except Exception as e:
            logger.error(f"  ✗ Failed to load {repo_id}: {e}")
            logger.error(f"  Make sure the dataset exists at https://huggingface.co/datasets/{repo_id}")
            raise
    
    logger.info("")
    logger.info(f"Summary before merge:")
    logger.info(f"  Total datasets: {len(datasets)}")
    logger.info(f"  Total episodes: {total_episodes}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info("")
    
    # Determine output directory
    if root:
        output_dir = Path(root) / output_repo_id
    else:
        from lerobot.utils.constants import HF_LEROBOT_HOME
        output_dir = HF_LEROBOT_HOME / output_repo_id
    
    # Merge datasets
    logger.info(f"Merging datasets into {output_repo_id}...")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    merged_dataset = merge_datasets(
        datasets=datasets,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
    )
    
    logger.info("")
    logger.info(f"=" * 70)
    logger.info(f"Merge Complete!")
    logger.info(f"=" * 70)
    logger.info(f"Merged dataset: {output_repo_id}")
    logger.info(f"Location: {output_dir}")
    logger.info(f"Episodes: {merged_dataset.meta.total_episodes}")
    logger.info(f"Frames: {merged_dataset.meta.total_frames}")
    logger.info(f"Tasks: {merged_dataset.meta.total_tasks}")
    logger.info("")
    
    # Push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {output_repo_id}")
        logger.info("This may take a while depending on dataset size...")
        
        merged_dataset.push_to_hub(
            tags=["metaworld", "robotics", "mt10", "multi-task"],
        )
        
        logger.info("")
        logger.info(f"=" * 70)
        logger.info(f"✓ Successfully pushed to hub!")
        logger.info(f"=" * 70)
        logger.info(f"View at: https://huggingface.co/datasets/{output_repo_id}")
    else:
        logger.info("Dataset saved locally (use --push-to-hub to upload)")
    
    return merged_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Merge individual MT10 task datasets into a unified metaworld_mt10 dataset"
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="HuggingFace username where task datasets are stored (e.g., 'aadarshram')",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        required=True,
        help="Repository ID for merged dataset (e.g., 'username/metaworld_mt10')",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for dataset storage (default: ~/.cache/huggingface/lerobot/)",
    )
    
    args = parser.parse_args()
    
    merge_mt10_datasets(
        username=args.username,
        output_repo_id=args.output_repo_id,
        push_to_hub=args.push_to_hub,
        root=args.root,
    )


if __name__ == "__main__":
    main()
