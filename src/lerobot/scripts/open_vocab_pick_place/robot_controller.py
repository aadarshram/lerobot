"""
Robot Controller Interface for LeRobot Pick-and-Place
Supports SO100, Koch, and other LeRobot robots
"""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import torch

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot


class PickPlaceController:
    """High-level controller for pick-and-place operations with LeRobot robots"""
    
    def __init__(
        self,
        robot_type: str = "so100",
        robot_overrides: Optional[List[str]] = None,
        mock: bool = False
    ):
        """
        Initialize robot controller.
        
        Args:
            robot_type: Type of robot (so100, koch, etc.)
            robot_overrides: Optional configuration overrides
            mock: If True, run in simulation mode without real robot
        """
        self.robot_type = robot_type
        self.mock = mock
        
        if not mock:
            print(f"Initializing {robot_type} robot...")
            self.robot = make_robot(robot_type, overrides=robot_overrides or [])
            self.robot.connect()
            print("Robot connected!")
        else:
            print(f"Running in MOCK mode (no real robot)")
            self.robot = None
        
        # Workspace bounds (customize for your setup)
        self.workspace = {
            "x": (0.15, 0.45),  # meters from robot base
            "y": (-0.30, 0.30),
            "z": (0.0, 0.30),
        }
        
        # Predefined locations (in robot frame)
        self.special_locations = {
            "trc": self._compute_corner("trc"),
            "tlc": self._compute_corner("tlc"),
            "brc": self._compute_corner("brc"),
            "blc": self._compute_corner("blc"),
            "mid": self._compute_corner("mid"),
        }
        
        # Motion parameters
        self.approach_height = 0.15  # Height to approach from above (m)
        self.grasp_height_offset = 0.02  # How much to descend for grasp (m)
        self.move_duration = 2.0  # Duration for each motion (seconds)
        self.gripper_close_pos = 1.0  # Gripper closed position
        self.gripper_open_pos = 0.0  # Gripper open position
    
    def _compute_corner(self, location: str) -> np.ndarray:
        """Compute workspace corner/center coordinates"""
        x_min, x_max = self.workspace["x"]
        y_min, y_max = self.workspace["y"]
        z = self.workspace["z"][0]
        
        locations = {
            "tlc": [x_min, y_max, z],
            "trc": [x_max, y_max, z],
            "blc": [x_min, y_min, z],
            "brc": [x_max, y_min, z],
            "mid": [(x_min + x_max) / 2, (y_min + y_max) / 2, z],
        }
        
        return np.array(locations[location])
    
    def is_within_workspace(self, position: np.ndarray) -> bool:
        """Check if position is within workspace bounds"""
        x, y, z = position
        return (
            self.workspace["x"][0] <= x <= self.workspace["x"][1] and
            self.workspace["y"][0] <= y <= self.workspace["y"][1] and
            self.workspace["z"][0] <= z <= self.workspace["z"][1]
        )
    
    def move_to_home(self):
        """Move robot to home/neutral position"""
        if self.mock:
            print("MOCK: Moving to home position")
            return
        
        print("Moving to home position...")
        # Use robot's home position if available
        if hasattr(self.robot, 'go_home'):
            self.robot.go_home()
        else:
            # Default home position (standing up)
            home_joints = np.zeros(len(self.robot.joints))
            self._move_joints(home_joints, duration=3.0)
    
    def _move_joints(self, target_joints: np.ndarray, duration: float = 2.0):
        """Move robot joints smoothly to target"""
        if self.mock:
            return
        
        start_time = time.time()
        start_joints = self.robot.get_state()
        
        while time.time() - start_time < duration:
            alpha = (time.time() - start_time) / duration
            current_joints = start_joints + alpha * (target_joints - start_joints)
            self.robot.send_action(current_joints)
            time.sleep(0.01)
        
        # Final position
        self.robot.send_action(target_joints)
    
    def _move_ee_to(
        self, 
        target_pos: np.ndarray, 
        target_orientation: Optional[np.ndarray] = None,
        duration: float = 2.0
    ):
        """
        Move end-effector to target position using inverse kinematics.
        
        Note: This is a simplified version. Real implementation would need:
        - Proper IK solver for your robot
        - Collision checking
        - Trajectory planning
        """
        if self.mock:
            print(f"MOCK: Moving EE to {target_pos}")
            return
        
        # For robots with IK support
        if hasattr(self.robot, 'inverse_kinematics'):
            target_joints = self.robot.inverse_kinematics(target_pos, target_orientation)
            self._move_joints(target_joints, duration)
        else:
            print("Warning: IK not implemented for this robot. Using approximate motion.")
            # Fallback: would need robot-specific implementation
    
    def set_gripper(self, position: float):
        """
        Control gripper.
        
        Args:
            position: 0.0 for open, 1.0 for closed
        """
        if self.mock:
            action = "closing" if position > 0.5 else "opening"
            print(f"MOCK: Gripper {action}")
            return
        
        # Send gripper command (robot-specific)
        gripper_action = np.array([position])
        if hasattr(self.robot, 'set_gripper'):
            self.robot.set_gripper(gripper_action)
        else:
            print("Warning: Gripper control not implemented")
    
    def pick_and_place(
        self,
        pick_pos: np.ndarray,
        place_pos: np.ndarray,
        pick_approach_height: Optional[float] = None,
        place_approach_height: Optional[float] = None
    ):
        """
        Execute pick-and-place motion.
        
        Args:
            pick_pos: 3D position to pick from [x, y, z]
            place_pos: 3D position to place at [x, y, z]
            pick_approach_height: Height to approach pick from (defaults to self.approach_height)
            place_approach_height: Height to approach place from (defaults to self.approach_height)
        """
        if pick_approach_height is None:
            pick_approach_height = self.approach_height
        if place_approach_height is None:
            place_approach_height = self.approach_height
        
        # Validate positions
        if not self.is_within_workspace(pick_pos):
            raise ValueError(f"Pick position {pick_pos} outside workspace")
        if not self.is_within_workspace(place_pos):
            raise ValueError(f"Place position {place_pos} outside workspace")
        
        print(f"Executing pick-and-place: {pick_pos} -> {place_pos}")
        
        # 1. Open gripper
        print("Opening gripper...")
        self.set_gripper(self.gripper_open_pos)
        time.sleep(0.5)
        
        # 2. Move above pick location
        print("Moving above pick location...")
        approach_pos = pick_pos.copy()
        approach_pos[2] += pick_approach_height
        self._move_ee_to(approach_pos, duration=self.move_duration)
        time.sleep(0.3)
        
        # 3. Descend to pick
        print("Descending to pick...")
        grasp_pos = pick_pos.copy()
        grasp_pos[2] += self.grasp_height_offset
        self._move_ee_to(grasp_pos, duration=1.0)
        time.sleep(0.3)
        
        # 4. Close gripper
        print("Closing gripper...")
        self.set_gripper(self.gripper_close_pos)
        time.sleep(1.0)  # Wait for grasp to stabilize
        
        # 5. Lift object
        print("Lifting object...")
        self._move_ee_to(approach_pos, duration=1.0)
        time.sleep(0.3)
        
        # 6. Move above place location
        print("Moving to place location...")
        place_approach = place_pos.copy()
        place_approach[2] += place_approach_height
        self._move_ee_to(place_approach, duration=self.move_duration)
        time.sleep(0.3)
        
        # 7. Descend to place
        print("Descending to place...")
        place_grasp = place_pos.copy()
        place_grasp[2] += self.grasp_height_offset
        self._move_ee_to(place_grasp, duration=1.0)
        time.sleep(0.3)
        
        # 8. Open gripper
        print("Opening gripper...")
        self.set_gripper(self.gripper_open_pos)
        time.sleep(0.5)
        
        # 9. Retract
        print("Retracting...")
        self._move_ee_to(place_approach, duration=1.0)
        time.sleep(0.3)
        
        print("Pick-and-place complete!")
    
    def get_special_location(self, name: str) -> Optional[np.ndarray]:
        """Get predefined location coordinates"""
        return self.special_locations.get(name)
    
    def disconnect(self):
        """Disconnect from robot"""
        if self.robot is not None:
            print("Disconnecting robot...")
            self.robot.disconnect()
            print("Robot disconnected")


def main():
    """Test robot controller in mock mode"""
    
    # Initialize in mock mode
    controller = PickPlaceController(robot_type="so100", mock=True)
    
    # Test workspace
    print("\nWorkspace bounds:")
    print(f"  X: {controller.workspace['x']}")
    print(f"  Y: {controller.workspace['y']}")
    print(f"  Z: {controller.workspace['z']}")
    
    print("\nSpecial locations:")
    for name, pos in controller.special_locations.items():
        print(f"  {name}: {pos}")
    
    # Test pick and place
    pick_pos = np.array([0.3, 0.1, 0.05])
    place_pos = controller.get_special_location("mid")
    
    print(f"\nTest pick-and-place:")
    print(f"  Pick: {pick_pos}")
    print(f"  Place: {place_pos}")
    
    controller.pick_and_place(pick_pos, place_pos)
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()
