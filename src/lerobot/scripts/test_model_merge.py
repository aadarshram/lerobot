#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Pending work.....

"""
Model Weight Merging and Evaluation Script

This script:
1. Loads two pretrained ACT models from HuggingFace Hub
2. Merges their weights using linear combination
3. Evaluates the merged model on both source tasks
4. Compares performance against individual models

Example usage:
python -m lerobot.scripts.test_model_merge \
    --model1_path=lerobot/act_aloha_sim_transfer_cube_human \
    --model2_path=lerobot/act_aloha_sim_insertion_human \
    --merge_weight=0.5 \
    --num_episodes=10 \
    --save_merged_model=outputs/merged_model
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.envs.factory import make_env
from lerobot.envs.configs import AlohaEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed


@dataclass
class ModelMergeConfig:
    """Configuration for model merging and evaluation."""
    
    # Model paths (HuggingFace Hub repo IDs)
    model1_path: str = "lerobot/act_aloha_sim_transfer_cube_human"
    model2_path: str = "lerobot/act_aloha_sim_insertion_human"
    
    # Merging parameters
    merge_weight: float = 0.5  # Weight for model1 (model2 gets 1-merge_weight)
    
    # Evaluation parameters
    num_episodes: int = 10
    max_steps_per_episode: int = 400
    seed: int = 42
    
    # Output
    save_merged_model: str | None = None  # Path to save merged model
    output_dir: str = "outputs/model_merge_results"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModelMerger:
    """Handles merging of two model weights."""
    
    def __init__(self, model1: ACTPolicy, model2: ACTPolicy, weight: float = 0.5):
        """
        Initialize model merger.
        
        Args:
            model1: First ACT policy
            model2: Second ACT policy
            weight: Weight for model1 (0 to 1), model2 gets (1 - weight)
        """
        self.model1 = model1
        self.model2 = model2
        self.weight = weight
        self.device = next(model1.parameters()).device
        
        # Verify models have compatible architectures
        self._verify_compatibility()
    
    def _verify_compatibility(self):
        """Verify that the two models have compatible architectures."""
        model1_params = dict(self.model1.named_parameters())
        model2_params = dict(self.model2.named_parameters())
        
        if set(model1_params.keys()) != set(model2_params.keys()):
            raise ValueError(
                f"Models have different parameter sets!\n"
                f"Model1 has {len(model1_params)} parameters\n"
                f"Model2 has {len(model2_params)} parameters"
            )
        
        # Check parameter shapes
        for name in model1_params.keys():
            if model1_params[name].shape != model2_params[name].shape:
                raise ValueError(
                    f"Parameter '{name}' has different shapes:\n"
                    f"Model1: {model1_params[name].shape}\n"
                    f"Model2: {model2_params[name].shape}"
                )
        
        logging.info("✓ Models are compatible for merging")
    
    def merge(self) -> ACTPolicy:
        """
        Merge the two models using linear combination.
        
        Returns:
            New ACTPolicy with merged weights
        """
        logging.info(f"Merging models with weight {self.weight:.2f} for model1, {1-self.weight:.2f} for model2")
        
        # Create a new model with the same config as model1
        merged_model = ACTPolicy(self.model1.config)
        merged_model.to(self.device)
        
        # Merge parameters
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_merged, param_merged) in zip(
                self.model1.named_parameters(),
                self.model2.named_parameters(),
                merged_model.named_parameters()
            ):
                assert name1 == name2 == name_merged, f"Parameter name mismatch: {name1}, {name2}, {name_merged}"
                
                # Linear combination
                param_merged.copy_(
                    self.weight * param1.data + (1 - self.weight) * param2.data
                )
        
        logging.info("✓ Model weights merged successfully")
        return merged_model


class ModelEvaluator:
    """Evaluates models on ALOHA environments."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def evaluate_model(
        self,
        model: ACTPolicy,
        task: str,
        num_episodes: int = 10,
        max_steps: int = 400,
        seed: int = 42
    ) -> dict:
        """
        Evaluate a model on a specific ALOHA task.
        
        Args:
            model: ACT policy to evaluate
            task: Task name (e.g., "AlohaInsertion-v0", "AlohaTransferCube-v0")
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            seed: Random seed
            
        Returns:
            Dictionary with evaluation metrics
        """
        logging.info(f"Evaluating on task: {task}")
        
        # Create environment using LeRobot's make_env with proper AlohaEnv config
        # This ensures we get both images and state (agent_pos)
        env_cfg = AlohaEnv(task=task, obs_type="pixels_agent_pos")
        env_dict = make_env(env_cfg, n_envs=1)
        
        # Extract the actual environment from the nested dict structure
        # make_env returns {suite_name: {task_id: vec_env}}
        suite_name = list(env_dict.keys())[0]
        vec_env = env_dict[suite_name][0]  # Get the vectorized environment
        
        # Extract the single underlying environment to avoid VectorEnv wrapper issues
        # gym_aloha envs don't play nicely with VectorEnv's action handling
        env = vec_env.envs[0]
        
        # Set seeds
        set_seed(seed)
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        model.eval()
        
        for episode in tqdm(range(num_episodes), desc=f"Evaluating {task}"):
            obs, info = env.reset(seed=seed + episode)
            model.reset()
            
            episode_reward = 0
            step = 0
            done = False
            
            while not done and step < max_steps:
                # Prepare observation for policy
                obs_dict = self._prepare_observation(obs)
                
                # Get action from policy
                with torch.no_grad():
                    action = model.select_action(obs_dict)
                
                # Execute action
                # Since we extracted the single env from VectorEnv, squeeze the batch dimension
                obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
                
                episode_reward += reward
                step += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            
            # Check if episode was successful (task-specific)
            if "success" in info and info["success"]:
                success_count += 1
        
        env.close()
        
        # Compute statistics
        results = {
            "task": task,
            "num_episodes": num_episodes,
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": success_count / num_episodes,
            "all_rewards": episode_rewards,
        }
        
        logging.info(f"Results for {task}:")
        logging.info(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        logging.info(f"  Success Rate: {results['success_rate']*100:.1f}%")
        logging.info(f"  Mean Length: {results['mean_length']:.1f}")
        
        return results
    
    def _prepare_observation(self, obs):
        """Convert environment observation to policy input format."""
        # Use LeRobot's preprocess_observation utility which handles:
        # - Image conversion (H,W,C) -> (C,H,W) and uint8 -> float32 normalization
        # - Proper key mapping (pixels -> observation.images, agent_pos -> observation.state)
        # - Batch dimension handling
        obs_dict = preprocess_observation(obs)
        
        # Move tensors to the correct device
        for key in obs_dict:
            if isinstance(obs_dict[key], torch.Tensor):
                obs_dict[key] = obs_dict[key].to(self.device)
        
        return obs_dict


def compare_results(results_dict: dict[str, dict]):
    """Print a comparison table of evaluation results."""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS COMPARISON")
    print("="*80)
    
    # Get all tasks
    all_tasks = set()
    for model_results in results_dict.values():
        all_tasks.update(model_results.keys())
    
    for task in sorted(all_tasks):
        print(f"\nTask: {task}")
        print("-" * 80)
        print(f"{'Model':<30} {'Mean Reward':<20} {'Success Rate':<20}")
        print("-" * 80)
        
        for model_name, results in results_dict.items():
            if task in results:
                r = results[task]
                reward_str = f"{r['mean_reward']:.2f} ± {r['std_reward']:.2f}"
                success_str = f"{r['success_rate']*100:.1f}%"
                print(f"{model_name:<30} {reward_str:<20} {success_str:<20}")
    
    print("="*80)


@parser.wrap()
def main(cfg: ModelMergeConfig):
    """Main execution function."""
    init_logging()
    
    logging.info("="*80)
    logging.info("MODEL WEIGHT MERGING AND EVALUATION")
    logging.info("="*80)
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load the two models
    logging.info(f"\n1. Loading models from HuggingFace Hub...")
    logging.info(f"   Model 1: {cfg.model1_path}")
    logging.info(f"   Model 2: {cfg.model2_path}")
    
    model1 = ACTPolicy.from_pretrained(cfg.model1_path)
    model1.to(cfg.device)
    model1.eval()
    logging.info(f"   ✓ Model 1 loaded")
    
    model2 = ACTPolicy.from_pretrained(cfg.model2_path)
    model2.to(cfg.device)
    model2.eval()
    logging.info(f"   ✓ Model 2 loaded")
    
    # Step 2: Merge the models
    logging.info(f"\n2. Merging models...")
    merger = ModelMerger(model1, model2, weight=cfg.merge_weight)
    merged_model = merger.merge()
    merged_model.eval()
    
    # Step 3: Determine tasks to evaluate
    # Extract task names from model paths (heuristic)
    tasks = []
    if "transfer_cube" in cfg.model1_path.lower():
        tasks.append("AlohaTransferCube-v0")
    if "insertion" in cfg.model2_path.lower():
        tasks.append("AlohaInsertion-v0")
    
    # If heuristic didn't work, use default tasks
    if not tasks:
        tasks = ["AlohaInsertion-v0", "AlohaTransferCube-v0"]
    
    logging.info(f"\n3. Evaluating on tasks: {tasks}")
    
    # Step 4: Evaluate all models
    evaluator = ModelEvaluator(device=cfg.device)
    all_results = {}
    
    # Evaluate model 1
    logging.info("\n4a. Evaluating Model 1 (source)...")
    all_results["Model 1"] = {}
    for task in tasks:
        results = evaluator.evaluate_model(
            model1, task, cfg.num_episodes, cfg.max_steps_per_episode, cfg.seed
        )
        all_results["Model 1"][task] = results
    
    # Evaluate model 2
    logging.info("\n4b. Evaluating Model 2 (source)...")
    all_results["Model 2"] = {}
    for task in tasks:
        results = evaluator.evaluate_model(
            model2, task, cfg.num_episodes, cfg.max_steps_per_episode, cfg.seed
        )
        all_results["Model 2"][task] = results
    
    # Evaluate merged model
    logging.info("\n4c. Evaluating Merged Model...")
    all_results[f"Merged (α={cfg.merge_weight})"] = {}
    for task in tasks:
        results = evaluator.evaluate_model(
            merged_model, task, cfg.num_episodes, cfg.max_steps_per_episode, cfg.seed
        )
        all_results[f"Merged (α={cfg.merge_weight})"][task] = results
    
    # Step 5: Print comparison
    compare_results(all_results)
    
    # Step 6: Save merged model if requested
    if cfg.save_merged_model:
        save_path = Path(cfg.save_merged_model)
        save_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"\n5. Saving merged model to {save_path}")
        merged_model.save_pretrained(save_path)
        logging.info(f"   ✓ Merged model saved")
    
    # Save results to file
    import json
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, tasks_results in all_results.items():
            json_results[model_name] = {}
            for task, metrics in tasks_results.items():
                json_results[model_name][task] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
        json.dump(json_results, f, indent=2)
    
    logging.info(f"\n✓ Results saved to {results_file}")
    logging.info("\nDone!")


if __name__ == "__main__":
    main()