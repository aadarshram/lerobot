"""
Main Execution Script for Open Vocabulary Pick-and-Place
Integrates perception, command parsing, and robot control
"""

import argparse
import time
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Optional

from perception import ObjectDetector
from command_parser import CommandParser, LLMProvider
from robot_controller import PickPlaceController
from camera_utils import CameraInterface, CameraCalibration


class OpenVocabPickPlace:
    """Main class orchestrating open vocabulary pick-and-place"""
    
    def __init__(
        self,
        robot_type: str = "so100",
        camera_index: int = 0,
        llm_provider: str = "groq",
        calibration_file: Optional[str] = None,
        mock_robot: bool = False,
        debug: bool = False
    ):
        """
        Initialize the system.
        
        Args:
            robot_type: Type of LeRobot robot (so100, koch, etc.)
            camera_index: Camera device index
            llm_provider: LLM provider for command parsing (groq/openai)
            calibration_file: Path to camera calibration file
            mock_robot: Run without real robot (for testing)
            debug: Enable debug visualizations
        """
        self.debug = debug
        self.mock_robot = mock_robot
        
        print("=" * 60)
        print("Initializing Open Vocabulary Pick-and-Place System")
        print("=" * 60)
        
        # Initialize components
        print("\n1. Initializing perception...")
        self.detector = ObjectDetector(score_threshold=0.1)
        
        print("\n2. Initializing command parser...")
        provider = LLMProvider.GROQ if llm_provider.lower() == "groq" else LLMProvider.OPENAI
        self.parser = CommandParser(provider=provider)
        
        print("\n3. Initializing robot controller...")
        self.robot = PickPlaceController(robot_type=robot_type, mock=mock_robot)
        
        print("\n4. Initializing camera...")
        self.camera = CameraInterface(camera_index=camera_index)
        self.calibration = CameraCalibration(
            camera_index=camera_index,
            calibration_file=calibration_file
        )
        
        print("\n✓ System initialized successfully!")
        print("=" * 60)
    
    def capture_scene(self) -> np.ndarray:
        """Capture current scene image"""
        print("\nCapturing scene...")
        image = self.camera.capture()
        
        if image is None:
            raise RuntimeError("Failed to capture image from camera")
        
        if self.debug:
            cv2.imwrite("debug_scene.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print("Saved scene to debug_scene.jpg")
        
        return image
    
    def detect_objects(
        self, 
        image: np.ndarray, 
        object_names: List[str]
    ) -> dict:
        """
        Detect objects in the scene.
        
        Args:
            image: RGB image
            object_names: List of object names to detect
            
        Returns:
            Dictionary mapping object names to their world coordinates
        """
        print(f"\nDetecting objects: {object_names}")
        
        # Run detection
        detections = self.detector.detect(image, object_names)
        
        # Get best instance for each object and convert to world coordinates
        object_positions = {}
        best_detections = []
        
        for obj_name in object_names:
            best = self.detector.get_best_instance(
                detections, 
                obj_name, 
                image_shape=image.shape[:2]
            )
            
            if best is not None:
                # Convert pixel to world coordinates
                px, py = self.detector.get_detection_center_pixel(
                    best, 
                    image.shape[:2]
                )
                
                world_pos = self.calibration.pixel_to_world(px, py)
                
                if world_pos is not None:
                    object_positions[obj_name] = world_pos
                    best_detections.append(best)
                    print(f"  ✓ Found {obj_name} at {world_pos}")
                else:
                    print(f"  ✗ Could not compute world position for {obj_name}")
            else:
                print(f"  ✗ {obj_name} not detected")
        
        # Debug visualization
        if self.debug and best_detections:
            self.detector.visualize(
                image, 
                detections, 
                best_detections,
                save_path="debug_detections.png"
            )
        
        return object_positions
    
    def execute_instruction(self, instruction: str):
        """
        Execute natural language pick-and-place instruction.
        
        Args:
            instruction: Natural language command (e.g., "pick the cup and place it in the corner")
        """
        print("\n" + "=" * 60)
        print(f"INSTRUCTION: {instruction}")
        print("=" * 60)
        
        # 1. Parse instruction
        print("\n[1/4] Parsing instruction...")
        commands = self.parser.extract_commands(instruction)
        
        if not commands:
            print("✗ Failed to parse instruction")
            return False
        
        print(f"Parsed {len(commands)} command(s):")
        for i, (pick_obj, place_loc) in enumerate(commands, 1):
            print(f"  {i}. Pick '{pick_obj}' → Place at '{place_loc}'")
        
        # 2. Capture scene
        print("\n[2/4] Capturing scene...")
        image = self.capture_scene()
        
        # 3. Detect all required objects
        print("\n[3/4] Detecting objects...")
        
        # Collect all objects we need to detect
        objects_to_detect = set()
        for pick_obj, place_loc in commands:
            objects_to_detect.add(pick_obj)
            # If place location is an object (not a predefined location), detect it too
            if place_loc not in self.robot.special_locations:
                objects_to_detect.add(place_loc)
        
        object_positions = self.detect_objects(image, list(objects_to_detect))
        
        # 4. Execute each command
        print("\n[4/4] Executing commands...")
        
        for i, (pick_obj, place_loc) in enumerate(commands, 1):
            print(f"\n--- Command {i}/{len(commands)} ---")
            
            # Get pick position
            if pick_obj not in object_positions:
                print(f"✗ Cannot pick '{pick_obj}': object not detected")
                continue
            
            pick_pos = object_positions[pick_obj]
            
            # Get place position
            if place_loc in self.robot.special_locations:
                # Predefined location (e.g., "trc", "mid")
                place_pos = self.robot.special_locations[place_loc]
                print(f"Place location: {place_loc} (predefined)")
            elif place_loc in object_positions:
                # Another object
                place_pos = object_positions[place_loc]
                print(f"Place location: near {place_loc}")
            else:
                print(f"✗ Cannot determine place location '{place_loc}'")
                continue
            
            # Execute pick and place
            try:
                self.robot.pick_and_place(pick_pos, place_pos)
                print(f"✓ Command {i} completed successfully")
                
                # Wait a bit between commands
                if i < len(commands):
                    time.sleep(1.0)
                    
            except Exception as e:
                print(f"✗ Command {i} failed: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("INSTRUCTION EXECUTION COMPLETE")
        print("=" * 60)
        
        return True
    
    def interactive_mode(self):
        """Run in interactive mode with user input"""
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Enter pick-and-place instructions in natural language.")
        print("Examples:")
        print("  - 'Pick the cup and place it in the top right corner'")
        print("  - 'Move the bottle to the middle'")
        print("  - 'Put the phone next to the laptop'")
        print("\nType 'quit' or 'exit' to stop.")
        print("=" * 60)
        
        # Move to home position
        self.robot.move_to_home()
        
        while True:
            try:
                instruction = input("\n> ").strip()
                
                if instruction.lower() in ['quit', 'exit', 'q']:
                    print("Exiting interactive mode...")
                    break
                
                if not instruction:
                    continue
                
                self.execute_instruction(instruction)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
    
    def shutdown(self):
        """Clean shutdown of all components"""
        print("\nShutting down...")
        
        try:
            self.robot.move_to_home()
        except:
            pass
        
        self.robot.disconnect()
        self.camera.disconnect()
        
        print("✓ Shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Open Vocabulary Pick-and-Place with LeRobot"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="so100",
        help="Robot type (so100, koch, etc.)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="groq",
        choices=["groq", "openai"],
        help="LLM provider for command parsing"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to camera calibration file"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Single instruction to execute (non-interactive mode)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without real robot (for testing)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with visualizations"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = OpenVocabPickPlace(
        robot_type=args.robot,
        camera_index=args.camera,
        llm_provider=args.llm,
        calibration_file=args.calibration,
        mock_robot=args.mock,
        debug=args.debug
    )
    
    try:
        if args.instruction:
            # Single instruction mode
            system.execute_instruction(args.instruction)
        else:
            # Interactive mode
            system.interactive_mode()
    
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()
