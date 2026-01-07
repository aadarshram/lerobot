"""
In-Context Learning with external policy

Current Flow:
1. Collect N demonstrations via teleoperation or expert policy in simulation
2. Send demonstrations to external policy as context (default, LLM)
3. External policy returns an adapted trajectory
4. Execute the external policy generated trajectory

Supports both real-world robots and simulation (MetaWorld).

Run:
python src/lerobot/scripts/ICL/icl_main.py \
  --use_simulation=true \
  --sim_task="reach-v3" \
  --num_demonstrations=2 \
  --demo_duration_s=3 \
  --exec_duration_s=3 \
  --fps=20 \
  --openai_api_key="${OPENAI_API_KEY}" \
  --debug_mode=false
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from openai import OpenAI

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def busy_wait(seconds: float):
    """Busy wait for a specified number of seconds."""
    if seconds > 0:
        time.sleep(seconds)


@dataclass
class InContextConfig:
    """Configuration for simple context learning."""
    
    # Mode selection
    use_simulation: bool = False  # If True, use MetaWorld simulation instead of real robot
    
    # Real robot setup (only used if use_simulation=False)
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    
    # Simulation setup (only used if use_simulation=True)
    sim_task: str = "reach-v3"  # MetaWorld task name
    sim_render_mode: str = "rgb_array"  # Render mode for simulation
    
    # Common parameters
    display_data: bool = False  # Whether to display data in Rerun
    fps: int = 30
    num_demonstrations: int = 1  # N demonstrations to collect
    demo_duration_s: int = 10  # How long to demonstrate
    exec_duration_s: int = 10  # How long to execute (can match demo_duration_s)
    
    # Debug mode settings
    debug_mode: bool = False  # If True, save videos and skip LLM, just replay a demo
    video_output_dir: str = "outputs/icl_videos"  # Directory to save demo and execution videos
    
    # LLM settings
    openai_api_key: str | None = None  # OpenAI API key (or set OPENAI_API_KEY env var)
    llm_model: str = "gpt-4o-mini-2024-07-18"  # OpenAI model to use
    llm_prompt: str = "You are a robot control policy. Given demonstration trajectories, output an adapted trajectory."

class InContextLearning:
    """
    In-context learning using external policy (default, LLM) in real/sim
    """

    def __init__(self, cfg: InContextConfig):
        """Initialize the session."""
        self.cfg = cfg
        
        if cfg.use_simulation:
            # Initialize MetaWorld simulation environment using LeRobot's MetaworldEnv
            logging.info(f"Initializing MetaWorld simulation with task: {cfg.sim_task}")
            try:
                from lerobot.envs.metaworld import MetaworldEnv
            except ImportError:
                raise ImportError(
                    "MetaWorld is required for simulation mode. Install with: pip install metaworld"
                )
            
            # Create MetaWorld environment using LeRobot's wrapper
            # This already handles expert policies, observation formatting, etc.
            self.env = MetaworldEnv(
                task=cfg.sim_task,
                camera_name="corner2",
                obs_type="state",
                render_mode=cfg.sim_render_mode,
                observation_width=480,
                observation_height=480,
            )
            
            # Get action and observation spaces
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            
            self.robot = None
            self.teleop = None
            self.teleop_action_processor = None
            self.robot_action_processor = None
            self.robot_observation_processor = None
            
            logging.info(f"Simulation initialized: action_space={self.action_space}, obs_space={self.observation_space}")
        else:
            # Initialize real robot and teleoperator
            if cfg.robot is None or cfg.teleop is None:
                raise ValueError("robot and teleop configs must be provided when use_simulation=False")
            
            logging.info("Initializing real robot setup...")
            self.robot = make_robot_from_config(cfg.robot)
            self.teleop = make_teleoperator_from_config(cfg.teleop)
            
            # Initialize processors
            (
                self.teleop_action_processor,
                self.robot_action_processor,
                self.robot_observation_processor,
            ) = make_default_processors()
            
            # Connect devices
            logging.info("Connecting robot and teleoperator...")
            self.robot.connect()
            self.teleop.connect()
            
            if not self.robot.is_connected or not self.teleop.is_connected:
                raise ValueError("Failed to connect robot or teleoperator!")
            
            self.env = None
            self.action_space = None
            self.observation_space = None
        
        # Initialize OpenAI client
        if cfg.openai_api_key:
            self.llm_client = OpenAI(api_key=cfg.openai_api_key)
        else:
            logging.warning("No OpenAI API key provided. LLM functionality will be limited.")
            self.llm_client = None
        
        # Keyboard listener
        self.listener, self.events = init_keyboard_listener()
        
        # Initialize visualization
        if cfg.display_data:
            init_rerun(session_name="simple_context_learning")
        
        # Create video output directory 
        self.video_dir = Path(cfg.video_output_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Video output directory: {self.video_dir}")
        
        logging.info("Simple context learning initialized successfully")

    def run(self):
        """
        Main execution: collect demos → external policy processes context → execute external policy output.
        In debug mode: collect demos → save videos → wait for approval → replay one demo → save execution video
        """
        
        # Collect N demonstrations
        log_say(f"Collecting {self.cfg.num_demonstrations} demonstration(s)")
        demonstrations = self._collect_demonstrations()
        
        if not demonstrations:
            log_say("No demonstrations collected. Exiting.")
            return
        
        log_say(f"Collected {len(demonstrations)} demonstration(s)")

        # Get current environment state for execution
        log_say("Resetting environment for execution...")
        if self.cfg.use_simulation:
            current_obs, _ = self.env.reset()
        else:
            current_obs = self.robot.get_observation()
            current_obs = self.robot_observation_processor(current_obs)

        log_say("Sending demonstrations to external policy for processing...")
        if self.cfg.debug_mode:
            adapted_trajectory = random.choice(demonstrations)
            log_say("Debug mode: skipping LLM, using a random demonstration")
        else:
            adapted_trajectory = self._query_llm_for_policy(demonstrations, current_obs) # For now, supports only LLM
        
        if not adapted_trajectory:
            log_say("Policy did not return a valid trajectory. Falling back to random selection.")
            adapted_trajectory = random.choice(demonstrations)
        
        # Execute the policy-adapted trajectory
        log_say("Executing policy-adapted trajectory")
        self._execute_demonstration(adapted_trajectory, save_video=True, video_name="execution")
                
        log_say("Execution complete!")
    
    def _collect_demonstrations(self) -> list[list[dict]]:
        """
        Collect N demonstrations via teleoperation (real robot) or expert policy (simulation). 
        For now, it is simple scripted policy to collect data in sim.
        
        Returns:
            List of demonstrations, where each demonstration is a list of
            (observation, action) pairs.
        """
        demonstrations = []
        
        for demo_idx in range(self.cfg.num_demonstrations):
            log_say(f"Recording demonstration {demo_idx + 1}/{self.cfg.num_demonstrations}")
            
            if self.cfg.use_simulation:
                log_say("Using expert policy in simulation to demonstrate the task")
                demo_trajectory = self._collect_simulation_demo()
            else:
                log_say("Use teleoperation to demonstrate the task")
                demo_trajectory = self._collect_real_robot_demo()
            
            if demo_trajectory:
                demonstrations.append(demo_trajectory)
                logging.info(
                    f"Demonstration {demo_idx + 1} recorded: {len(demo_trajectory)} frames"
                )
                
                # Save video 
                self._save_demo_video(demo_trajectory, demo_idx)
            
            # Pause between demonstrations if collecting multiple
            if demo_idx < self.cfg.num_demonstrations - 1:
                log_say("Prepare for the next demonstration...")
                time.sleep(2.0)
                
        return demonstrations
    
    def _collect_real_robot_demo(self) -> list[dict]:
        """Collect a demonstration using real robot teleoperation."""
        demo_trajectory = []
        t0 = time.perf_counter()
        
        while time.perf_counter() - t0 < self.cfg.demo_duration_s:
            if self.events["exit_early"] or self.events["stop_recording"]:
                log_say("Demonstration interrupted")
                break
            
            loop_t0 = time.perf_counter()
            
            # Get observation
            obs = self.robot.get_observation()
            obs_processed = self.robot_observation_processor(obs)
            
            # Get teleop action
            action = self.teleop.get_action()
            action_processed = self.teleop_action_processor((action, obs))
            
            # Execute action
            robot_action = self.robot_action_processor((action_processed, obs_processed))
            self.robot.send_action(robot_action)
            
            # Store in trajectory
            demo_trajectory.append({
                "observation": obs_processed,
                "action": action_processed,
                "robot_action": robot_action,
            })
            
            # Visualize
            if self.cfg.display_data:
                log_rerun_data(
                    observation=obs_processed,
                    action=action_processed,
                )
            
            # Maintain FPS
            dt_s = time.perf_counter() - loop_t0
            busy_wait(max(1.0 / self.cfg.fps - dt_s, 0.0))
        
        return demo_trajectory
    
    def _collect_simulation_demo(self) -> list[dict]:
        """
        Collect a demonstration using MetaWorld expert policy.
        """
        demo_trajectory = []
        obs, info = self.env.reset()
        
        # Get raw observation for expert policy
        raw_obs = self.env._env._get_obs()
        expert_policy = self.env.expert_policy
        
        max_steps = int(self.cfg.demo_duration_s * self.cfg.fps)
        
        for step in range(max_steps):
            if self.events["exit_early"] or self.events["stop_recording"]:
                log_say("Demonstration interrupted")
                break
            
            loop_t0 = time.perf_counter()
            
            # Render frame for video (important for state observations)
            rendered_frame = self.env.render()
            
            # Get expert action
            action = expert_policy.get_action(raw_obs)
            action = action[:4]  # Only use first 4 dimensions (x, y, z, gripper)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Store in trajectory BEFORE updating obs, include rendered frame
            demo_trajectory.append({
                "observation": obs,
                "action": action,
                "robot_action": action,
                "reward": reward,
                "success": info.get("success", False),
                "rendered_frame": rendered_frame,  # For video saving
            })
            
            # Update observation for next iteration
            obs = next_obs
            raw_obs = self.env._env._get_obs()
            
            # Visualize
            if self.cfg.display_data:
                vis_obs = {}
                if "pixels" in obs:
                    vis_obs["pixels"] = obs["pixels"]
                if "agent_pos" in obs:
                    vis_obs["agent_pos"] = obs["agent_pos"]
                log_rerun_data(
                    observation=vis_obs,
                    action=action,
                )
            
            if terminated or truncated:
                logging.info(f"Episode ended at step {step}, success={info.get('success', False)}")
                break
            
            # Maintain FPS
            dt_s = time.perf_counter() - loop_t0
            busy_wait(max(1.0 / self.cfg.fps - dt_s, 0.0))
        
        return demo_trajectory
    
    def _save_demo_video(self, demo_trajectory: list[dict], demo_idx: int):
        """
        Save a demonstration trajectory as a video.
        
        Args:
            demo_trajectory: The demonstration trajectory containing observations
            demo_idx: Index of the demonstration
        """
        frames = []
        for step in demo_trajectory:
            # First try rendered frame (for state observations)
            if "rendered_frame" in step and step["rendered_frame"] is not None:
                frames.append(step["rendered_frame"].copy())
            else:
                # Fallback to pixels in observation
                obs = step.get("observation", {})
                if "pixels" in obs:
                    frames.append(obs["pixels"].copy())
        
        if frames:
            video_path = self.video_dir / f"demo_{demo_idx + 1}.mp4"
            imageio.mimsave(video_path, frames, fps=self.cfg.fps)
            logging.info(f"Saved demo video: {video_path}")
        else:
            logging.warning(f"No frames to save for demo {demo_idx + 1}")
    
    def _query_llm_for_policy(self, demonstrations: list[list[dict]], current_obs: dict) -> list[dict]:
        """
        Send demonstrations and current observation to LLM and get an adapted trajectory.
        
        Args:
            demonstrations: List of demonstration trajectories
            current_obs: Current observation from the environment (with new object positions)
            
        Returns:
            Adapted trajectory (list of action dicts)
        """
        # Format demonstrations with observations and actions
        demo_data = self._format_demonstrations_for_llm(demonstrations)
        
        # Format current observation
        current_state = self._format_observation_for_llm(current_obs)
        
        # Calculate expected trajectory length
        demo_lengths = [len(demo) for demo in demonstrations]
        avg_demo_length = int(np.mean(demo_lengths))
        
        # Create the prompt for adaptation
        system_prompt = """You are an expert robot control policy. You will receive:
1. Demonstration trajectories showing a robot performing a task
2. Current environment state with potentially different object positions

Your task is to adapt the demonstrated actions to work with the current state.
Output an adapted trajectory as a JSON list of actions."""
        
        user_prompt = f"""I have {len(demonstrations)} demonstration(s) of a robot manipulation task.

DEMONSTRATIONS:
{demo_data}

CURRENT ENVIRONMENT STATE (for execution):
{current_state}

The demonstrations show the robot performing the task with objects at certain positions.
The current state shows the NEW positions of objects.

Please analyze the demonstrations and adapt the actions to work with the current object positions.

IMPORTANT: The demonstrations contain {avg_demo_length} total steps. You MUST output {avg_demo_length} adapted actions to complete the full trajectory. Do NOT output only a few sample actions - output the COMPLETE trajectory.

Output format (JSON):
{{
    "reasoning": "Brief explanation of how you adapted the trajectory",
    "adapted_actions": [
        [x, y, z, gripper],  // Action 1
        [x, y, z, gripper],  // Action 2
        ... // Continue for ALL {avg_demo_length} actions
        [x, y, z, gripper]   // Action {avg_demo_length}
    ]
}}

The adapted_actions should:
1. Account for the difference between demonstration positions and current positions
2. Contain EXACTLY {avg_demo_length} actions (same as the demonstration length)
3. Form a complete trajectory from start to finish"""

        try:
            # Query the LLM
            logging.info("Querying LLM for trajectory adaptation...")
            response = self.llm_client.chat.completions.create(
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent adaptations
            )
            
            llm_output = response.choices[0].message.content
            logging.info(f"LLM response: {llm_output}")
            
            # Parse the LLM's response
            # Remove markdown code blocks if present
            llm_output = llm_output.strip()
            if llm_output.startswith("```"):
                # Extract content between ``` markers
                parts = llm_output.split("```")
                if len(parts) >= 3:
                    llm_output = parts[1]
                else:
                    llm_output = parts[-1] if len(parts) > 1 else llm_output
                # Remove language identifier if present
                if llm_output.startswith("json"):
                    llm_output = llm_output[4:]
                llm_output = llm_output.strip()
            
            # Remove JavaScript-style comments (//) that LLM might include
            import re
            llm_output = re.sub(r'//.*?(?=\n|$)', '', llm_output)
            
            # Try to parse JSON
            try:
                result = json.loads(llm_output)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {e}")
                logging.error(f"Attempted to parse:\n{llm_output}")
                raise ValueError(f"LLM returned invalid JSON: {e}") from e
            reasoning = result.get("reasoning", "No reasoning provided")
            adapted_actions = result.get("adapted_actions", [])
            
            log_say(f"LLM reasoning: {reasoning}")
            log_say(f"Generated {len(adapted_actions)} adapted actions")
            
            # Validate trajectory length
            demo_lengths = [len(demo) for demo in demonstrations]
            avg_demo_length = int(np.mean(demo_lengths))
            if len(adapted_actions) < avg_demo_length * 0.5:
                logging.warning(
                    f"LLM output only {len(adapted_actions)} actions but demo has {avg_demo_length} steps! "
                    f"Trajectory may be incomplete."
                )
            
            if not adapted_actions:
                log_say("No adapted actions returned, falling back to first demonstration")
                return demonstrations[0]
            
            # Convert adapted actions to trajectory format
            adapted_trajectory = []
            # Use observations from first demonstration as template
            template_demo = demonstrations[0]
            
            for i, action in enumerate(adapted_actions):
                # Use template observation or repeat last one if we exceed demo length
                obs_idx = min(i, len(template_demo) - 1)
                template_obs = template_demo[obs_idx]["observation"]
                
                adapted_trajectory.append({
                    "observation": template_obs,  # Use template (will be overridden during execution)
                    "action": np.array(action, dtype=np.float32),
                    "robot_action": np.array(action, dtype=np.float32),
                    "reward": 0.0,
                    "success": False,
                })
            
            return adapted_trajectory
                
        except Exception as e:
            logging.error(f"Error querying LLM: {e}", exc_info=True)
            return None
    
    def _format_observation_for_llm(self, obs: dict) -> str:
        """
        Format a single observation for LLM using state-based observations.
        
        Args:
            obs: Observation dictionary (with state-based data)
            
        Returns:
            Formatted string describing the observation
        """
        lines = []
        
        # Extract positions from state observations
        if "end_effector_pos" in obs:
            end_effector = obs["end_effector_pos"]
            if isinstance(end_effector, np.ndarray):
                lines.append(f"End-effector position: [{', '.join([f'{v:.3f}' for v in end_effector])}]")
        
        if "gripper_state" in obs:
            gripper = obs["gripper_state"]
            if isinstance(gripper, np.ndarray):
                lines.append(f"Gripper state: {gripper[0]:.3f}")
        
        if "object_pos" in obs:
            object_pos = obs["object_pos"]
            if isinstance(object_pos, np.ndarray):
                lines.append(f"Object position: [{', '.join([f'{v:.3f}' for v in object_pos])}]")
        
        # Fallback to agent_pos if state observations not available (for real robot)
        if not lines and "agent_pos" in obs:
            agent_pos = obs["agent_pos"]
            if isinstance(agent_pos, np.ndarray):
                lines.append(f"Robot state: [{', '.join([f'{v:.3f}' for v in agent_pos])}]")
        
        return "\n".join(lines) if lines else "State information not available"
    
    def _format_demonstrations_for_llm(self, demonstrations: list[list[dict]]) -> str:
        """
        Format demonstrations into a readable summary for the LLM.
        Include both observations and actions.
        
        Args:
            demonstrations: List of demonstration trajectories
            
        Returns:
            Formatted string describing the demonstrations
        """
        summary_lines = []
        
        for demo_idx, demo in enumerate(demonstrations):
            # Get basic stats about the demonstration
            num_steps = len(demo)
            
            summary_lines.append(f"\n=== Demonstration {demo_idx + 1} ===")
            summary_lines.append(f"Total steps: {num_steps}")
            
            # Get initial state from first observation
            if num_steps > 0:
                first_obs = demo[0].get("observation", {})
                summary_lines.append(f"\nInitial state:")
                summary_lines.append(self._format_observation_for_llm(first_obs))
            
            # Sample some key trajectory points with both state and action
            sample_indices = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
            sample_indices = [i for i in sample_indices if i < num_steps]
            
            summary_lines.append(f"\nKey trajectory points:")
            for idx in sample_indices:
                step = demo[idx]
                obs = step.get("observation", {})
                action = step["action"]
                
                # Format action
                if isinstance(action, np.ndarray):
                    action_str = f"[{', '.join([f'{v:.3f}' for v in action])}]"
                else:
                    action_str = str(action)
                
                # Show object position if available
                obj_pos_str = ""
                if "object_pos" in obs:
                    obj_pos = obs["object_pos"]
                    if isinstance(obj_pos, np.ndarray):
                        obj_pos_str = f", object_pos=[{', '.join([f'{v:.3f}' for v in obj_pos])}]"
                
                summary_lines.append(f"  Step {idx}: action={action_str}{obj_pos_str}")
            
            # Success info if available
            if "success" in demo[-1]:
                success = demo[-1]["success"]
                summary_lines.append(f"\nTask success: {success}")
        
        return "\n".join(summary_lines)
    
    def _average_demonstrations(self, demonstrations: list[list[dict]]) -> list[dict]:
        """
        Average multiple demonstrations into a single trajectory.
        
        Args:
            demonstrations: List of demonstration trajectories
            
        Returns:
            Averaged trajectory
        """
        if len(demonstrations) == 1:
            return demonstrations[0]
        
        # Find the minimum length
        min_length = min(len(demo) for demo in demonstrations)
        
        averaged_trajectory = []
        
        for step_idx in range(min_length):
            # Collect all actions at this timestep
            actions_at_step = [demo[step_idx] for demo in demonstrations]
            
            # Average the robot actions (numerical values)
            avg_robot_action = {}
            first_robot_action = actions_at_step[0]["robot_action"]
            
            for key in first_robot_action.keys():
                values = [action["robot_action"][key] for action in actions_at_step]
                avg_robot_action[key] = np.mean(values)
            
            # Use the observation and processed action from the first demo as reference
            averaged_trajectory.append({
                "observation": actions_at_step[0]["observation"],
                "action": actions_at_step[0]["action"],
                "robot_action": avg_robot_action,
            })
        
        logging.info(f"Averaged {len(demonstrations)} demonstrations into {len(averaged_trajectory)} steps")
        return averaged_trajectory
    
    def _execute_demonstration(self, demonstration: list[dict], save_video: bool = False, video_name: str = "execution"):
        """
        Execute trajectory.
        
        Args:
            demonstration: List of (observation, action) pairs to replay
            save_video: Whether to save execution as video
            video_name: Name for the video file (without extension)
        """
        log_say(f"Executing trajectory with {len(demonstration)} steps")
        
        if self.cfg.use_simulation:
            self._execute_simulation_demo(demonstration, save_video, video_name)
        else:
            self._execute_real_robot_demo(demonstration)
    
    def _execute_real_robot_demo(self, demonstration: list[dict]):
        """Execute demonstration on real robot."""
        t0 = time.perf_counter()
        executed_steps = 0
        
        for step_idx, step in enumerate(demonstration):
            if self.events["exit_early"] or self.events["stop_recording"]:
                log_say("Execution interrupted")
                break
            
            # Check if we've exceeded execution duration
            if time.perf_counter() - t0 >= self.cfg.exec_duration_s:
                break
            
            loop_t0 = time.perf_counter()
            
            # Get current observation (for context, though not used in simple version)
            current_obs = self.robot.get_observation()
            current_obs_processed = self.robot_observation_processor(current_obs)
            
            # Use the stored action from the demonstration
            robot_action = step["robot_action"]
            
            # Execute the action
            self.robot.send_action(robot_action)
            executed_steps += 1
            
            # Visualize
            if self.cfg.display_data:
                log_rerun_data(
                    observation=current_obs_processed,
                    action=step["action"],
                )
            
            # Maintain FPS
            dt_s = time.perf_counter() - loop_t0
            busy_wait(max(1.0 / self.cfg.fps - dt_s, 0.0))
        
        logging.info(
            f"Execution completed: {executed_steps}/{len(demonstration)} steps executed"
        )
    
    def _execute_simulation_demo(self, demonstration: list[dict], save_video: bool = False, video_name: str = "execution"):
        """Execute demonstration in simulation."""
        obs, info = self.env.reset()
        executed_steps = 0
        total_reward = 0
        success = False
        
        # Collect frames for video if needed
        frames = [] if save_video else None
        
        for step_idx, step in enumerate(demonstration):
            if self.events["exit_early"] or self.events["stop_recording"]:
                log_say("Execution interrupted")
                break
            
            loop_t0 = time.perf_counter()
            
            # Render frame for video (before taking action)
            if save_video:
                rendered_frame = self.env.render()
                if rendered_frame is not None:
                    frames.append(rendered_frame.copy())
                elif "pixels" in obs:
                    # Fallback to pixels if render returns None
                    frames.append(obs["pixels"].copy())
            
            # Use the stored action from the demonstration
            action = step["robot_action"]
            
            # Execute in simulation
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            executed_steps += 1
            total_reward += reward
            success = info.get("success", False)
            
            # Visualize
            if self.cfg.display_data:
                log_rerun_data(
                    observation={"pixels": obs["pixels"]},
                    action=action,
                )
            
            obs = next_obs
            
            if terminated or truncated:
                logging.info(f"Episode ended at step {step_idx}")
                break
            
            # Maintain FPS
            dt_s = time.perf_counter() - loop_t0
            busy_wait(max(1.0 / self.cfg.fps - dt_s, 0.0))
        
        # Save video if requested
        if save_video and frames:
            video_path = self.video_dir / f"{video_name}.mp4"
            imageio.mimsave(video_path, frames, fps=self.cfg.fps)
            logging.info(f"Saved execution video: {video_path}")
        
        logging.info(
            f"Execution completed: {executed_steps}/{len(demonstration)} steps executed, "
            f"total_reward={total_reward:.2f}, success={success}"
        )
    
    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up...")
        if self.robot is not None:
            self.robot.disconnect()
        if self.env is not None:
            self.env.close()
        self.listener.stop()


@parser.wrap()
def main(cfg: InContextConfig):
    """
    Main entry point.
    If no cfg is provided, uses default settings for testing.
    """
    init_logging()
    
    mode = "simulation" if cfg.use_simulation else "real robot"
    logging.info(f"Configuration: N={cfg.num_demonstrations} demonstrations, mode={mode}")
    if cfg.use_simulation:
        logging.info(f"Simulation task: {cfg.sim_task}")
    
    # Validate configuration
    if not cfg.use_simulation and (cfg.robot is None or cfg.teleop is None):
        raise ValueError("robot and teleop configs must be provided when use_simulation=False")
    
    # Initialize and run
    session = InContextLearning(cfg)
    
    try:
        session.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        session.cleanup()


if __name__ == "__main__":
    main()
