"""
Example configuration for open vocabulary pick-and-place
Copy this to config.py and customize for your setup
"""

# Robot Configuration
ROBOT_CONFIG = {
    "type": "so100",  # Options: so100, koch, aloha, etc.
    "mock": False,    # Set True for testing without robot
}

# Camera Configuration
CAMERA_CONFIG = {
    "index": 0,       # Camera device index
    "width": 640,     # Image width
    "height": 480,    # Image height
    "fps": 30,        # Frames per second
}

# Workspace Bounds (in meters, relative to robot base)
WORKSPACE = {
    "x": (0.15, 0.45),  # Forward/backward from robot
    "y": (-0.30, 0.30),  # Left/right from robot
    "z": (0.0, 0.30),    # Height above table
}

# Predefined Locations (will be computed from workspace)
# These are automatically generated, but you can override specific ones
SPECIAL_LOCATIONS = {
    # "trc": [0.45, 0.30, 0.0],  # Top right corner
    # "tlc": [0.15, 0.30, 0.0],  # Top left corner
    # "brc": [0.45, -0.30, 0.0], # Bottom right corner
    # "blc": [0.15, -0.30, 0.0], # Bottom left corner
    # "mid": [0.30, 0.0, 0.0],   # Middle
}

# Motion Parameters
MOTION_CONFIG = {
    "approach_height": 0.15,        # Height to approach from above (m)
    "grasp_height_offset": 0.02,    # Descend distance for grasp (m)
    "move_duration": 2.0,           # Duration for motions (seconds)
    "gripper_close_pos": 1.0,       # Gripper closed (0-1)
    "gripper_open_pos": 0.0,        # Gripper open (0-1)
}

# Detection Parameters
DETECTION_CONFIG = {
    "model": "google/owlvit-base-patch32",  # OWL-ViT model
    "score_threshold": 0.1,                  # Minimum confidence
    "device": "cuda",                        # cuda or cpu
}

# LLM Configuration
LLM_CONFIG = {
    "provider": "groq",                      # groq or openai
    "model": "llama-3.3-70b-versatile",     # Model name
    # API keys are read from environment variables:
    # GROQ_API_KEY or OPENAI_API_KEY
}

# Camera Calibration
# Set calibration_file to None to use defaults
# Or provide path to your calibration file
CALIBRATION_CONFIG = {
    "calibration_file": None,  # e.g., "calibration/my_camera.json"
    "table_height": 0.0,       # Z-coordinate of table in robot frame
}

# Debug Settings
DEBUG_CONFIG = {
    "save_images": True,         # Save debug images
    "save_detections": True,     # Save detection visualizations
    "output_dir": "debug_output", # Directory for debug files
    "verbose": True,             # Print detailed logs
}

# Example Usage Scenarios

# Scenario 1: Lab Setup with SO100
LAB_SETUP = {
    **ROBOT_CONFIG,
    "type": "so100",
    "workspace": {
        "x": (0.20, 0.50),
        "y": (-0.35, 0.35),
        "z": (0.0, 0.25),
    },
}

# Scenario 2: Desktop Setup with Koch
DESKTOP_SETUP = {
    **ROBOT_CONFIG,
    "type": "koch",
    "workspace": {
        "x": (0.10, 0.30),
        "y": (-0.20, 0.20),
        "z": (0.0, 0.20),
    },
}

# Scenario 3: Testing/Development
TEST_SETUP = {
    **ROBOT_CONFIG,
    "mock": True,  # No real robot needed
}


# Helper function to load config
def get_config(setup="default"):
    """
    Get configuration for a specific setup.
    
    Args:
        setup: "default", "lab", "desktop", or "test"
    
    Returns:
        Dictionary with all configuration parameters
    """
    if setup == "lab":
        robot_config = LAB_SETUP
    elif setup == "desktop":
        robot_config = DESKTOP_SETUP
    elif setup == "test":
        robot_config = TEST_SETUP
    else:
        robot_config = ROBOT_CONFIG
    
    return {
        "robot": robot_config,
        "camera": CAMERA_CONFIG,
        "workspace": robot_config.get("workspace", WORKSPACE),
        "motion": MOTION_CONFIG,
        "detection": DETECTION_CONFIG,
        "llm": LLM_CONFIG,
        "calibration": CALIBRATION_CONFIG,
        "debug": DEBUG_CONFIG,
    }
