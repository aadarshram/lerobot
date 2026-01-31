"""
Camera Utilities for Real Robot Pick-and-Place
Handles camera calibration and pixel-to-world coordinate transformation
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict
import json

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera


class CameraCalibration:
    """Camera calibration and coordinate transformation utilities"""
    
    def __init__(
        self,
        camera_index: int = 0,
        calibration_file: Optional[str] = None,
        image_size: Tuple[int, int] = (640, 480)
    ):
        """
        Initialize camera calibration.
        
        Args:
            camera_index: Camera device index or name
            calibration_file: Path to calibration file (JSON)
            image_size: Camera image size (width, height)
        """
        self.camera_index = camera_index
        self.image_size = image_size
        self.calibration_file = calibration_file
        
        # Camera intrinsics (will be loaded or estimated)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Extrinsics (camera pose relative to robot base)
        self.camera_to_robot_transform = None
        
        # Load calibration if available
        if calibration_file and Path(calibration_file).exists():
            self.load_calibration(calibration_file)
        else:
            print("Warning: No calibration file found. Using default parameters.")
            self._initialize_default_calibration()
    
    def _initialize_default_calibration(self):
        """Initialize with default/estimated camera parameters"""
        # Rough estimate for typical webcam
        width, height = self.image_size
        focal_length = width  # Rough approximation
        
        self.camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        
        # Default: Camera looking down from 0.5m height
        # This is a placeholder - MUST be calibrated for real use!
        self.camera_to_robot_transform = np.eye(4)
        self.camera_to_robot_transform[:3, 3] = [0.0, 0.0, 0.5]  # 0.5m above table
    
    def load_calibration(self, filepath: str):
        """Load calibration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['dist_coeffs'])
        self.camera_to_robot_transform = np.array(data['camera_to_robot_transform'])
        
        print(f"Loaded calibration from {filepath}")
    
    def save_calibration(self, filepath: str):
        """Save calibration to JSON file"""
        data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'camera_to_robot_transform': self.camera_to_robot_transform.tolist(),
            'image_size': self.image_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved calibration to {filepath}")
    
    def pixel_to_camera_ray(self, px: int, py: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel to 3D ray in camera frame.
        
        Args:
            px, py: Pixel coordinates
            
        Returns:
            (ray_origin, ray_direction) in camera frame
        """
        # Normalize pixel coordinates
        point_2d = np.array([px, py, 1.0])
        
        # Invert camera matrix to get ray direction
        ray_cam = np.linalg.inv(self.camera_matrix) @ point_2d
        ray_cam = ray_cam / np.linalg.norm(ray_cam)
        
        # Ray origin is camera center
        ray_origin = np.array([0, 0, 0])
        
        return ray_origin, ray_cam
    
    def pixel_to_world(
        self, 
        px: int, 
        py: int, 
        table_height: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Project pixel onto table plane in robot/world frame.
        
        Args:
            px, py: Pixel coordinates
            table_height: Z-coordinate of table in robot frame
            
        Returns:
            3D position [x, y, z] in robot frame, or None if invalid
        """
        # Get ray in camera frame
        ray_origin_cam, ray_dir_cam = self.pixel_to_camera_ray(px, py)
        
        # Transform ray to robot frame
        # For simplicity, assuming camera looking straight down
        # Real implementation needs proper transformation
        
        # Extract camera position in robot frame
        cam_pos_robot = self.camera_to_robot_transform[:3, 3]
        
        # Simple case: camera looking straight down (negative Z)
        # More complex transformations would use rotation matrix
        t = (table_height - cam_pos_robot[2]) / (-1.0)  # Assuming looking down
        
        if t < 0:
            return None
        
        # Approximate intersection (simplified)
        # This assumes camera is axis-aligned - real version needs proper transform
        world_x = cam_pos_robot[0] + ray_dir_cam[0] * t
        world_y = cam_pos_robot[1] + ray_dir_cam[1] * t
        world_z = table_height
        
        return np.array([world_x, world_y, world_z])
    
    def calibrate_with_checkerboard(
        self,
        images: list,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025
    ):
        """
        Calibrate camera using checkerboard pattern.
        
        Args:
            images: List of calibration images
            checkerboard_size: Number of inner corners (width, height)
            square_size: Size of checkerboard squares in meters
        """
        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints = []  # 3D points in real world
        imgpoints = []  # 2D points in image plane
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                imgpoints.append(corners_refined)
        
        # Calibrate
        if len(objpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            
            print(f"Calibration successful with {len(objpoints)} images")
            print(f"RMS error: {ret}")
        else:
            print("Calibration failed: no checkerboard detected in any image")


class CameraInterface:
    """Interface for capturing images from LeRobot cameras"""
    
    def __init__(
        self,
        camera_index: int = 0,
        fps: int = 30,
        width: int = 640,
        height: int = 480
    ):
        """
        Initialize camera interface.
        
        Args:
            camera_index: Camera device index
            fps: Frames per second
            width, height: Image resolution
        """
        self.camera_index = camera_index
        self.fps = fps
        self.width = width
        self.height = height
        
        # Try to use LeRobot's OpenCV camera
        try:
            self.camera = OpenCVCamera(camera_index, fps, width, height)
            self.camera.connect()
            print(f"Connected to camera {camera_index}")
        except Exception as e:
            print(f"Warning: Could not initialize LeRobot camera: {e}")
            print("Falling back to OpenCV VideoCapture")
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture single frame.
        
        Returns:
            RGB image as numpy array (H, W, 3) or None if failed
        """
        try:
            if isinstance(self.camera, OpenCVCamera):
                # LeRobot camera
                image = self.camera.read()
            else:
                # OpenCV VideoCapture
                ret, image = self.camera.read()
                if not ret:
                    return None
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    def disconnect(self):
        """Disconnect camera"""
        if isinstance(self.camera, OpenCVCamera):
            self.camera.disconnect()
        else:
            self.camera.release()
        print("Camera disconnected")


def main():
    """Test camera utilities"""
    
    # Initialize camera
    camera = CameraInterface(camera_index=0)
    
    # Capture test image
    print("Capturing image...")
    image = camera.capture()
    
    if image is not None:
        print(f"Captured image shape: {image.shape}")
        
        # Save test image
        cv2.imwrite("test_camera_capture.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Saved test image to test_camera_capture.jpg")
    else:
        print("Failed to capture image")
    
    # Test calibration (with default parameters)
    calibration = CameraCalibration(image_size=(image.shape[1], image.shape[0]))
    
    # Test pixel to world conversion
    center_px = image.shape[1] // 2
    center_py = image.shape[0] // 2
    world_pos = calibration.pixel_to_world(center_px, center_py)
    print(f"Center pixel ({center_px}, {center_py}) -> World: {world_pos}")
    
    # Cleanup
    camera.disconnect()


if __name__ == "__main__":
    main()
