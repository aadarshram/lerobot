"""
Script to generate MetaWorld datasets using expert policies and save them in LeRobot format.

Usage:
    # Generate a single task dataset
    python src/lerobot/scripts/generate_MetaWorld_datasets.py \
        --task reach-v2 \
        --repo-id username/metaworld-reach-v2 \
        --num-episodes 50
    
    # Generate MT50 benchmark (all 50 tasks)
    python src/lerobot/scripts/generate_MetaWorld_datasets.py \
        --benchmark MT50 \
        --repo-id username/metaworld-mt50 \
        --num-episodes 50
    
    # Generate MT10 benchmark
    python src/lerobot/scripts/generate_MetaWorld_datasets.py \
        --benchmark MT10 \
        --repo-id username/metaworld-mt10 \
        --num-episodes 50 \
        --push-to-hub
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.metaworld import (
    DIFFICULTY_TO_TASKS,
    TASK_POLICY_MAPPING,
    TASK_DESCRIPTIONS,
    TASK_NAME_TO_ID,
    MetaworldEnv,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# MetaWorld task groups - MT = "Multi-Task"
MT50_TASKS = [
    "assembly-v3", "basketball-v3", "bin-picking-v3", "box-close-v3", "button-press-topdown-v3",
    "button-press-topdown-wall-v3", "button-press-v3", "button-press-wall-v3", "coffee-button-v3",
    "coffee-pull-v3", "coffee-push-v3", "dial-turn-v3", "disassemble-v3", "door-close-v3",
    "door-lock-v3", "door-open-v3", "door-unlock-v3", "hand-insert-v3", "drawer-close-v3",
    "drawer-open-v3", "faucet-open-v3", "faucet-close-v3", "hammer-v3", "handle-press-side-v3",
    "handle-press-v3", "handle-pull-side-v3", "handle-pull-v3", "lever-pull-v3", "peg-insert-side-v3",
    "pick-place-wall-v3", "pick-out-of-hole-v3", "reach-v3", "push-back-v3", "push-v3", "pick-place-v3",
    "plate-slide-v3", "plate-slide-side-v3", "plate-slide-back-v3", "plate-slide-back-side-v3",
    "peg-unplug-side-v3", "soccer-v3", "stick-push-v3", "stick-pull-v3", "push-wall-v3",
    "reach-wall-v3", "shelf-place-v3", "sweep-into-v3", "sweep-v3", "window-open-v3", "window-close-v3"
]

MT10_TASKS = [
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-open-v3",
    "drawer-close-v3", "button-press-topdown-v3", "peg-insert-side-v3",
    "window-open-v3", "window-close-v3"
]

def get_metaworld_features(camera_name="corner2", image_size=(480, 480)):
    """Define the features for MetaWorld datasets (matching lerobot/metaworld_mt50 format)."""
    h, w = image_size
    
    # Return features in LeRobot format (plain dict, not HuggingFace Features)
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"axes": ["x", "y", "z", "gripper"]},
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (39,),
            "names": ["keypoints"],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"axes": ["x", "y", "z", "gripper"]},
        },
        "task_id": {
            "dtype": "int16",
            "shape": (1,),
            "names": None,
        },
    }
    return features


def generate_episode(env, expert_policy, task_name, episode_idx, max_steps=500):
    """Generate a single episode using the expert policy."""
    frames = []
    success = False
    
    # Get task description and ID
    task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
    task_id = TASK_NAME_TO_ID.get(task_name, 0)
    
    obs, info = env.reset()
    raw_obs = env._env._get_obs()
    
    for step in range(max_steps):
        # Get expert action
        action = expert_policy.get_action(raw_obs)
        action = action[:4]  # Only use first 4 dimensions (gripper position + gripper open/close)
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_raw_obs = env._env._get_obs()
        
        # Create frame data (matching lerobot/metaworld_mt50 schema)
        frame = {
            "observation.image": obs["pixels"],
            "observation.state": raw_obs[:4].astype(np.float32),  # Robot state (x, y, z, gripper)
            "observation.environment_state": raw_obs.astype(np.float32),  # Full environment state
            "action": action.astype(np.float32),
            "task": task_desc,  # Task description
            "task_id": np.array([task_id], dtype=np.int16),
        }
        frames.append(frame)
        
        obs = next_obs
        raw_obs = next_raw_obs
        
        if terminated or truncated:
            success = info.get("success", False)
            break
    
    return frames, task_desc, success


def generate_metaworld_dataset(
    task_name,
    repo_id,
    num_episodes=50,
    push_to_hub=False,
    root=None,
    fps=80,
    private=False,
    camera_name="corner2",
    image_size=(480, 480),
    only_successful=True,
):
    """Generate a MetaWorld dataset for a single task.
    
    Args:
        only_successful: If True, only save successful episodes. Recommended for behavioral cloning.
    """
    logger.info(f"Generating dataset for task: {task_name}")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Number of episodes: {num_episodes}")
    
    # Check if expert policy exists
    if task_name not in TASK_POLICY_MAPPING:
        raise ValueError(f"No expert policy found for task '{task_name}'. Available tasks: {list(TASK_POLICY_MAPPING.keys())}")
    
    # Create environment using LeRobot's MetaworldEnv wrapper
    env = MetaworldEnv(
        task=task_name,
        camera_name=camera_name,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_height=image_size[0],
        observation_width=image_size[1],
    )
    
    # Get expert policy
    expert_policy = TASK_POLICY_MAPPING[task_name]()
    
    # Define features
    features = get_metaworld_features(camera_name, image_size)
    
    # Create dataset
    logger.info("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type="metaworld",
        root=root,
        use_videos=True,
    )
    
    # Get task description for this task
    task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
    logger.info(f"Task description: {task_desc}")
    
    # Generate episodes
    logger.info(f"Generating {num_episodes} episodes (only_successful={only_successful})...")
    success_count = 0
    saved_count = 0
    attempted = 0
    success_list = []  # Track success for each episode
    
    while saved_count < num_episodes:
        attempted += 1
        logger.info(f"Attempt {attempted} (saved: {saved_count}/{num_episodes})")
        
        # Generate episode
        frames, task_description, success = generate_episode(env, expert_policy, task_name, saved_count)
        
        # Skip failed episodes if only_successful is True
        if only_successful and not success:
            logger.info(f"  Skipping failed episode (success: {success})")
            continue
        
        # Add frames to dataset with task description
        for frame in frames:
            dataset.add_frame(frame)
        
        # Save episode
        dataset.save_episode()
        
        # Track success for this episode
        success_list.append(success)
        
        if success:
            success_count += 1
        saved_count += 1
        
        logger.info(f"  Episode {saved_count}: {len(frames)} frames, success: {success}")
    
    logger.info(f"\nDataset generation complete:")
    logger.info(f"  Saved episodes: {saved_count}")
    logger.info(f"  Successful: {success_count}/{saved_count} ({100*success_count/saved_count:.1f}%)")
    logger.info(f"  Total attempts: {attempted}")
    if only_successful and attempted > saved_count:
        logger.info(f"  Discarded failed: {attempted - saved_count}")
    
    # Finalize dataset
    logger.info("Finalizing dataset...")
    dataset.finalize()
    
    # Push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing to HuggingFace Hub: {repo_id}")
        dataset.push_to_hub(
            tags=["metaworld", "robotics", task_name],
            private=private,
        )
        logger.info("✓ Successfully pushed to hub!")
    else:
        logger.info(f"Dataset saved locally at: {dataset.root}")
    
    env.close()
    return dataset


def generate_benchmark_dataset(
    benchmark,
    base_repo_id,
    num_episodes_per_task=50,
    push_to_hub=False,
    root=None,
    fps=80,
    private=False,
    camera_name="corner2",
    image_size=(480, 480),
    only_successful=True,
):
    """Generate datasets for all tasks in a benchmark (MT10 or MT50)."""
    if benchmark == "MT50":
        tasks = MT50_TASKS
    elif benchmark == "MT10":
        tasks = MT10_TASKS
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Use 'MT10' or 'MT50'")
    
    logger.info(f"Generating {benchmark} benchmark: {len(tasks)} tasks")
    logger.info(f"Episodes per task: {num_episodes_per_task}")
    logger.info(f"Total episodes: {len(tasks) * num_episodes_per_task}")
    
    datasets = {}
    for idx, task in enumerate(tasks, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Task {idx}/{len(tasks)}: {task}")
        logger.info(f"{'='*70}\n")
        
        # Create repo_id for this task
        task_repo_id = f"{base_repo_id}-{task}"
        
        try:
            dataset = generate_metaworld_dataset(
                task_name=task,
                repo_id=task_repo_id,
                num_episodes=num_episodes_per_task,
                push_to_hub=push_to_hub,
                root=root,
                fps=fps,
                private=private,
                camera_name=camera_name,
                image_size=image_size,
                only_successful=only_successful,
            )
            datasets[task] = dataset
            logger.info(f"✓ Task {task} completed successfully!")
        except Exception as e:
            logger.error(f"✗ Failed to generate dataset for task {task}: {e}")
            continue
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Benchmark generation complete!")
    logger.info(f"Successfully generated {len(datasets)}/{len(tasks)} tasks")
    logger.info(f"{'='*70}\n")
    
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate MetaWorld datasets using expert policies"
    )
    
    # Task selection
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task",
        type=str,
        help="Single task name (e.g., 'reach-v2', 'push-v2')",
    )
    task_group.add_argument(
        "--benchmark",
        type=str,
        choices=["MT10", "MT50"],
        help="Generate entire benchmark (MT10 or MT50)",
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID to save the dataset (e.g., 'username/metaworld-reach-v2')",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to generate per task (default: 50). "
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=80,
        help="Frames per second (default: 80)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for dataset storage (default: ~/.cache/huggingface/lerobot/)",
    )
    
    # Environment configuration
    parser.add_argument(
        "--camera-name",
        type=str,
        default="corner2",
        choices=["corner", "corner2", "corner3", "topview", "behindGripper"],
        help="Camera view to use (default: corner2)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=480,
        help="Image width (default: 480)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Image height (default: 480)",
    )
    
    # Upload configuration
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on HuggingFace Hub",
    )
    parser.add_argument(
        "--only-successful",
        action="store_true",
        default=True,
        help="Only save successful episodes (default: True, recommended for behavioral cloning)",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Include failed episodes (overrides --only-successful)",
    )
    
    args = parser.parse_args()
    
    # Handle success filtering
    only_successful = args.only_successful and not args.include_failures
    
    image_size = (args.image_height, args.image_width)
    
    if args.task:
        # Generate single task dataset
        generate_metaworld_dataset(
            task_name=args.task,
            repo_id=args.repo_id,
            num_episodes=args.num_episodes,
            push_to_hub=args.push_to_hub,
            root=args.root,
            fps=args.fps,
            private=args.private,
            camera_name=args.camera_name,
            image_size=image_size,
            only_successful=only_successful,
        )
    else:
        # Generate benchmark datasets
        generate_benchmark_dataset(
            benchmark=args.benchmark,
            base_repo_id=args.repo_id,
            num_episodes_per_task=args.num_episodes,
            push_to_hub=args.push_to_hub,
            root=args.root,
            fps=args.fps,
            private=args.private,
            camera_name=args.camera_name,
            image_size=image_size,
            only_successful=only_successful,
        )


if __name__ == "__main__":
    main()