#!/usr/bin/env python3
"""
Push a locally cached LeRobot dataset to HuggingFace Hub.

Usage:
    python src/lerobot/scripts/push_dataset.py --repo-id aadarshram/metaworld-reach-v3
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Push a locally cached dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing the dataset (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=["metaworld", "robotics"],
        help="Tags to add to the dataset",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Loading dataset: {args.repo_id}")
    
    # Load the dataset from cache
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
    )
    
    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  - Episodes: {dataset.num_episodes}")
    logger.info(f"  - Frames: {dataset.num_frames}")
    logger.info(f"  - FPS: {dataset.fps}")
    
    # Push to hub
    logger.info(f"Pushing to HuggingFace Hub: {args.repo_id}")
    dataset.push_to_hub(
        tags=args.tags,
        private=args.private,
    )
    
    logger.info("✓ Successfully pushed to hub!")
    logger.info(f"  View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
