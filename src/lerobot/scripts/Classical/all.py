# Import libraries

import os
import time
import math
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imageio_ffmpeg
from base64 import b64encode
from IPython.display import HTML

import pybullet as p
import pybullet_data

import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers.image_utils import ImageFeatureExtractionMixin

# Helpers

def play_video(path):
  mp4 = open(path, 'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML('<video width=480 controls><source src="%s" type="video/mp4"></video>' % data_url)


# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


class TableTopEnv:
  """
  Defines a simple Table Top Environment in Pybullet with KUKA arm and objects from Google Scanned Objects
  """
  def __init__(self, config, gui=False):
      self.config = config
      self.gui = gui
      self.client_id = None

      # IDs
      self.plane_id = None
      self.table_id = None
      self.kuka_id = None
      self.gripper_id = None
      self.object_ids = {}

      # Robot constants
      self.ee_idx = 6 # End-Effector id
      self.num_joints = None

      # Camera
      self.cam_params = {}

      # Workspace bounds
      self.workspace = {
          "x": (0.53, 1.1),
          "y": (-0.85, 0.45),
          "z": (0.65, 1.0),
       }

      # Define corners and middle
      xmin, xmax = self.workspace["x"]
      ymin, ymax = self.workspace["y"]
      z = self.workspace["z"][0]
      self.special_pts = {
          "tlc": [xmin, ymax, z],
          "trc": [xmax, ymax, z],
          "blc": [xmin, ymin, z],
          "brc": [xmax, ymin, z],
          "mid": [(xmin + xmax) / 2, (ymin + ymax) / 2, z],
      }

  def reset(self, object_dict, retracted_arm=False, side_view=False, use_gso=False):
      '''
      retracted_arm param if True, loads in position that does not obstruct camera view
      side_view if True, loads env with side 3rd person camera.
      '''
      self._connect()
      self._load_scene()
      self._load_robot(retracted_arm=retracted_arm)
      self._setup_camera(side_view=side_view)
      self.load_objects(object_dict,use_gso=use_gso)
      # Warmup sim run
      for _ in range(50):
          p.stepSimulation()

  def _connect(self):
      if p.isConnected():
        p.disconnect()

      mode = p.GUI if self.gui else p.DIRECT
      self.client_id = p.connect(mode)
      p.setAdditionalSearchPath(pybullet_data.getDataPath())
      p.setGravity(0, 0, -10) # 10 m/s^2 gravity downward
      p.setTimeStep(1./240.)

  def close(self):
    if self.client_id is not None:
        p.disconnect(self.client_id)
        self.client_id = None

  def _load_scene(self):
      self.plane_id = p.loadURDF("plane.urdf")

      self.table_id = p.loadURDF(
          "table/table.urdf",
          basePosition=[1.0, -0.2, 0.0],
          baseOrientation=[0, 0, 0.7071, 0.7071]
      )

  def _load_robot(self, retracted_arm=False):
      self.kuka_id = p.loadURDF(
          "kuka_iiwa/model_vr_limits.urdf",
            basePosition=[1.4, -0.2, 0.6],
            baseOrientation=[0, 0, 0, 1]
      )

      self.gripper_id = p.loadSDF(
          "gripper/wsg50_one_motor_gripper_new_free_base.sdf"
      )[0]

      self._attach_gripper()
      self._reset_robot(retracted_arm=retracted_arm)
      self.num_joints = p.getNumJoints(self.kuka_id)

  def _attach_gripper(self):
      p.createConstraint(
          self.kuka_id, 6,
          self.gripper_id, 0,
          p.JOINT_FIXED,
          [0, 0, 0],
          [0, 0, 0.05],
          [0, 0, 0]
      )

      cid = p.createConstraint(
          self.gripper_id, 4,
          self.gripper_id, 6,
          jointType=p.JOINT_GEAR,
          jointAxis=[1, 1, 1],
          parentFramePosition=[0, 0, 0],
          childFramePosition=[0, 0, 0]
      )

      p.changeConstraint(cid, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)

  def _reset_robot(self, retracted_arm=False):

      if retracted_arm:
          # Retracted arm configuration to not obstruct camera view
          kuka_joints = [
              0.0, 0.0, 0.0, -0.5, 0.0, 1.0, 0.0
          ]
          # Adjust gripper base position and orientation for retracted arm
          gripper_base_pos = [1.2, -0.2, 1.5]
          gripper_base_orn = p.getQuaternionFromEuler([0, math.pi/2, 0])
      else:
          # Original, normal operating configuration
          kuka_joints = [
               0.0, 0.75, 0.0, 1.570793, 0.0, -1.036725, 0.0
          ] # 0.0, 0.0, 0.0, 1.570793, 0.0, -1.036725, 0.0
          gripper_base_pos = [0.923103, -0.2, 1.250036]
          gripper_base_orn = [-0.0, 0.964531, -0.0, -0.263970]

      for j in range(p.getNumJoints(self.kuka_id)):
          p.resetJointState(
              self.kuka_id, j, kuka_joints[j])
          p.setJointMotorControl2(
              self.kuka_id, j,
              p.POSITION_CONTROL,
              kuka_joints[j], 0
          )

      p.resetBasePositionAndOrientation(
          self.gripper_id,
          gripper_base_pos,
          gripper_base_orn
      )


      gripper_joints = [
          0.000000, -0.011130, -0.206421, 0.205143, -0.009999, 0.000000, -0.010055, 0.000000
          ]

      for j in range(p.getNumJoints(self.gripper_id)):
          p.resetJointState(self.gripper_id, j, gripper_joints[j])
          p.setJointMotorControl2(self.gripper_id, j, p.POSITION_CONTROL, gripper_joints[j], 0)

  def load_objects(self, object_dict, use_gso=False):
      """
      object_dict:
        name -> {path ("pathto.urdf), position ([x,y,z]), scale [optional]}
      """

      if use_gso:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        p.setAdditionalSearchPath('/content/drive/MyDrive/GSO')
        for name, cfg in object_dict.items():
          obj_id = p.loadURDF(
              cfg["path"],
              basePosition=cfg["position"],
              globalScaling=cfg.get("scale", 1.0)
          )
          self.object_ids[name] = obj_id

      else:
        for name, cfg in object_dict.items():
          obj_id = p.loadURDF(
              cfg["path"],
              basePosition=cfg["position"],
              globalScaling=cfg.get("scale", 1.0)
          )
          self.object_ids[name] = obj_id


  def _setup_camera(self, side_view=False):
    if side_view: # 3rd person corner view
      self.cam_params = dict(
        target=[0.85, -0.2, 0.65],
        distance=1.5,
        yaw=45,
        pitch=-35,
        roll=0,
        up_axis=2,
        up_vector=[0, 0, 1],
        width=480,
        height=480,
        fov=60,
        near=0.01,
        far=100
      )

    else: # top-view
      self.cam_params = dict(
          target=[0.85, -0.2, 0.65],
          distance=1.5, # 1.5
          yaw=0,
          pitch=-90,
          roll=0,
          up_axis=2,
          up_vector=[0, 0, 1],
          width=480,
          height=480,
          fov=60,
          near=0.01,
          far=100
      )

  def render(self):
      c = self.cam_params

      view = p.computeViewMatrixFromYawPitchRoll(
          c["target"], c["distance"],
          c["yaw"], c["pitch"], c["roll"], 2
      )

      proj = p.computeProjectionMatrixFOV(
          c["fov"],
          c["width"] / c["height"],
          c["near"], c["far"]
      )

      img = p.getCameraImage(
          c["width"], c["height"], view, proj
      )[2][:, :, :3]

      return img

  def is_within_bounds(self, pos):
    if pos is None:
        return False
    x, y, _ = pos
    return (self.workspace["x"][0] <= x <= self.workspace["x"][1]) and (self.workspace["y"][0] <= y <= self.workspace["y"][1])

  def pixel_to_world(self, px, py):
    """
    Projects pixel (px, py) onto tabletop plane (z = workspace["z"][0]).
    """
    c = self.cam_params

    view = p.computeViewMatrixFromYawPitchRoll(
        c["target"], c["distance"],
        c["yaw"], c["pitch"], c["roll"], c["up_axis"]
    )
    proj = p.computeProjectionMatrixFOV(
        c["fov"], c["width"] / c["height"],
        c["near"], c["far"]
    )

    view = np.array(view).reshape(4, 4).T
    proj = np.array(proj).reshape(4, 4).T

    inv_vp = np.linalg.inv(proj @ view)

    # Normalize pixel to NDC
    x_ndc = (2.0 * px) / self.cam_params["width"] - 1.0
    y_ndc = 1.0 - (2.0 * py) / self.cam_params["height"]

    # Two points on ray in clip space
    near = np.array([x_ndc, y_ndc, -1.0, 1.0])
    far  = np.array([x_ndc, y_ndc,  1.0, 1.0])

    # Transform to world space
    near_w = inv_vp @ near
    far_w  = inv_vp @ far

    near_w /= near_w[3]
    far_w  /= far_w[3]

    ray_dir = far_w[:3] - near_w[:3]
    ray_dir /= np.linalg.norm(ray_dir)

    ray_origin = near_w[:3]

    # Intersect with table plane z = constant
    z_plane = self.workspace["z"][0] # Use the defined workspace Z for the table plane
    t = (z_plane - ray_origin[2]) / ray_dir[2]

    if t < 0:
        return None

    world = ray_origin + t * ray_dir
    world = world.tolist()

    return world if self.is_within_bounds(world) else None

  def move_ee(self, position, orientation=None):
      if orientation is None:
          orientation = p.getQuaternionFromEuler([0, 1.01 * math.pi, 0])

      joint_poses = p.calculateInverseKinematics(
          self.kuka_id,
          self.ee_idx,
          position,
          orientation
      )

      for j in range(p.getNumJoints(self.kuka_id)):
          p.setJointMotorControl2(
              self.kuka_id,
              j,
              p.POSITION_CONTROL,
              joint_poses[j]
          )

  def _set_gripper(self, width):
      p.setJointMotorControl2(self.gripper_id, 4, p.POSITION_CONTROL, width, force=100)
      p.setJointMotorControl2(self.gripper_id, 6, p.POSITION_CONTROL, width, force=100)

  def open_gripper(self):
      self._set_gripper(0.0)

  def close_gripper(self):
      self._set_gripper(0.15) # 0.05

  def start_video(self, path="vid.mp4", fps=30):
    c = self.cam_params
    self._video = imageio_ffmpeg.write_frames(
        path,
        (c["width"], c["height"]),
        fps=fps
    )
    self._video.send(None)  # seed

  def record_frame(self, image):
      if hasattr(self, "_video"):
          self._video.send(np.ascontiguousarray(image))

  def stop_video(self):
    if hasattr(self, "_video"):
        self._video.close()
        del self._video

  def validate_pick_place(self, src, tgt):
      if src is None or tgt is None:
          return False
      if not self.is_within_bounds(src):
          return False
      if not self.is_within_bounds(tgt):
          return False
      return True

  def _move_and_step(self, target_pos, gripper_open, steps=1, record_every = 8):
      self.move_ee(target_pos)
      if gripper_open:
          self.open_gripper()
      else:
          self.close_gripper()

      record_every = 1 if steps < record_every else record_every
      for i in range(steps):
          p.stepSimulation()
          if hasattr(self, "_video") and i % record_every == 0:
              self.record_frame(self.render())

  def _move_and_step_slow(self, target_pos, gripper_open, steps=100, record_every=8):
    # Get the starting position of the end-effector
    # getLinkState returns [pos, orn, ...], index 4 is the world link frame position
    current_pos = np.array(p.getLinkState(self.kuka_id, self.ee_idx)[4])
    target_pos = np.array(target_pos)

    # Control the gripper once at the start
    if gripper_open:
        self.open_gripper()
    else:
        self.close_gripper()

    # Step through the simulation
    record_every = 1 if steps < record_every else record_every

    for i in range(steps):
        # Calculate the intermediate position (Linear Interpolation)
        fraction = (i + 1) / steps
        intermediate_pos = current_pos + (target_pos - current_pos) * fraction

        # Command the robot to the intermediate position
        self.move_ee(intermediate_pos)

        # Advance physics
        p.stepSimulation()

        # Record if necessary
        if hasattr(self, "_video") and i % record_every == 0:
            self.record_frame(self.render())

  def execute_pick_and_place(
      self,
      src_xyz,
      tgt_xyz,
      z_offset_grab=0.32,# 0.15
      z_offset_release=0.1, # maybe this isnt needed
      record_video=False,
      video_path="vid.mp4"
  ):
      """
      Classical pick-and-place with fixed top-down grasp.
      """
      if record_video:
          self.start_video(video_path)

      # PICK
      self._move_and_step_slow(
          [src_xyz[0], src_xyz[1], src_xyz[2] + z_offset_grab + 0.2],
          gripper_open=True,
          steps=200
      )

      #self._move_and_step(
          #src_xyz,
          #gripper_open=True,
          #steps=200
      #)

      self._move_and_step(
          [src_xyz[0], src_xyz[1], src_xyz[2] + z_offset_grab],
          gripper_open=False,
          steps=200
      )

      self._move_and_step(
          [src_xyz[0], src_xyz[1], src_xyz[2] + z_offset_grab + 0.2],
          gripper_open=False,
          steps=200
      )


      # TRANSFER
      self._move_and_step_slow(
          [tgt_xyz[0], tgt_xyz[1], src_xyz[2] + z_offset_grab + 0.2],
          gripper_open=False,
          steps=200
      )

      # PLACE
      self._move_and_step(
          [tgt_xyz[0], tgt_xyz[1], src_xyz[2] + z_offset_grab],
          gripper_open=False,
          steps=200
      )

      self._move_and_step(
          [tgt_xyz[0], tgt_xyz[1], src_xyz[2] + z_offset_grab],
          gripper_open=True,
          steps=200
      )

      self._move_and_step(
          [tgt_xyz[0], tgt_xyz[1], src_xyz[2] + z_offset_grab + 0.2],
          gripper_open=True,
          steps=400
      )

      if record_video:
          self.stop_video()


  def generate_random_coordinates(self, n, min_distance=0.5, x_range=(0.6, 1.0), y_range=(-0.80, 0.40), z_fixed=0.65, max_retries_per_point=1000):
      """
      Generates n random 3D coordinates within specified x and y ranges,
      with z fixed, and ensures a minimum distance between them.

      Args:
          n (int): The number of coordinates to generate.
          min_distance (float): The minimum Euclidean distance required between any two points.
          x_range (tuple): A tuple (min_x, max_x) for the x-coordinate.
          y_range (tuple): A tuple (min_y, max_y) for the y-coordinate.
          z_fixed (float): The fixed z-coordinate.
          max_retries_per_point (int): Maximum attempts to place a single point.

      Returns:
          numpy.ndarray: An array of [x, y, z] coordinates.

      Raises:
          ValueError: If it's not possible to place 'n' points with the given constraints.
      """
      if n <= 0:
          return np.array([])

      coordinates = []
      x_min, x_max = x_range
      y_min, y_max = y_range

      for _ in range(n):
          placed = False
          retries = 0
          while not placed and retries < max_retries_per_point:
              x = random.uniform(x_min, x_max)
              y = random.uniform(y_min, y_max)
              new_coord = [x, y, z_fixed]

              is_valid = True
              for existing_coord in coordinates:
                  if math.dist(new_coord, existing_coord) < min_distance:
                      is_valid = False
                      break

              if is_valid:
                  coordinates.append(new_coord)
                  placed = True
              else:
                  retries += 1

          if not placed:
              raise ValueError(f"Could not place all {n} points with min_distance={min_distance}. Only {len(coordinates)} points were placed.")

      return np.array(coordinates)
  
# Set env

def setup_env(retracted_arm=False, side_view=False):
  env = TableTopEnv(config=None)
  # Define objects
  n_objs = 2
  # Obj locs
  generated_coords = env.generate_random_coordinates(
      n=2,
      min_distance=0.2
  )

  use_gso = True
  load_from_drive= True

  if use_gso:
      if load_from_drive:
        objects = {
          "obj1": {
              "path": "Bus/Bus.urdf",
              "position": generated_coords[0],
              "scale": 1.0,
          },
          "obj2": {
              "path": "Shoe/Shoe.urdf",
              "position": generated_coords[1],
              "scale": 1.0,
          },
       }
  else:
      # Default cube
      objects = {
          "cube": {
              "path": "cube.urdf",
              "position": env.special_pts["mid"], # [0.85,-0.6,0.65]
              "scale": 0.1,
          }
      }

  env.reset(objects, retracted_arm=retracted_arm, side_view=side_view, use_gso=True)

  return env

env=setup_env()

# Tests

# Workspace-camera alignment
env._setup_camera(side_view=False) # Reconfigure camera to top-down for this test
img = env.render()
h, w, _ = img.shape
center_world = env.pixel_to_world(w // 2, h // 2)
assert np.linalg.norm(np.array(center_world) - np.array(env.cam_params["target"])) < 1, "Camera-World not aligned!"

env._setup_camera(side_view=True) # comment this line out if sideview isnt needed


# Capture perception img
scene_image = env.render()
plt.imshow(Image.fromarray(scene_image))
plt.show()

class Detector:
    def __init__(
        self,
        model_name="google/owlvit-base-patch32",
        device=None,
        score_threshold=0.01,
        env=None # env parameter for workspace bounds checking
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.env = env

        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image, text_queries):
        inputs = self.processor(
            images=image,
            text=text_queries,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        logits = torch.max(outputs.logits[0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().numpy()
        labels = logits.indices.cpu().numpy()
        boxes = outputs.pred_boxes[0].cpu().numpy()

        return {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
            "text_queries": text_queries,
        }

    def get_best_instance(self, detections, query):
        best = None

        for score, box, label_idx in zip(
            detections["scores"],
            detections["boxes"],
            detections["labels"],
        ):
            if score < self.score_threshold:
                continue

            if detections["text_queries"][label_idx] != query:
                continue

            cx, cy, _, _ = box # Get center coordinates of the detected box
            px = int(cx * self.env.cam_params["width"]) # Convert pixel coordinates to world coordinates for bounds checking
            py = int(cy * self.env.cam_params["height"])
            world_coord = self.env.pixel_to_world(px, py)

            if world_coord is None or not self.env.is_within_bounds(world_coord): # Filter out detections not within workspace bounds
                continue

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "box_norm": box,
                    "label": query,
                    "world_coord": world_coord # Store world coordinates in best instance
                }

        return best

    def plot(self, image, detections, best=None, plot_only_best = True):
        H, W = image.shape[:2]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        ax.axis("off")

        if not plot_only_best:
          for score, box, label in zip(
              detections["scores"],
              detections["boxes"],
              detections["labels"],
          ):
              if score < self.score_threshold:
                  continue

              cx, cy, w, h = box
              x0 = (cx - w / 2) * W
              x1 = (cx + w / 2) * W
              y0 = (cy - h / 2) * H
              y1 = (cy + h / 2) * H

              ax.plot([x0, x1, x1, x0, x0],
                      [y0, y0, y1, y1, y0], "r")

        # Plot the best detection in green edge
        if best is not None:
            cx, cy, w, h = best["box_norm"]
            x0 = (cx - w/2) * W
            x1 = (cx + w/2) * W
            y0 = (cy - h/2) * H
            y1 = (cy + h/2) * H

            ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],"g")
            ax.text(x0, y1, f"{best['label']} ({best['score']:.2f})",
                    color="green",
                    bbox=dict(facecolor="white", edgecolor="green"))

        plt.show()

class Detector:
    def __init__(
        self,
        model_name="google/owlvit-base-patch32",
        device=None,
        score_threshold=0.01,
        env=None # env parameter for workspace bounds checking
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.env = env

        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image, text_queries):
        inputs = self.processor(
            images=image,
            text=text_queries,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        logits = torch.max(outputs.logits[0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().numpy()
        labels = logits.indices.cpu().numpy()
        boxes = outputs.pred_boxes[0].cpu().numpy()

        return {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
            "text_queries": text_queries,
        }

    def get_best_instance(self, detections, query):
        best = None

        for score, box, label_idx in zip(
            detections["scores"],
            detections["boxes"],
            detections["labels"],
        ):
            if score < self.score_threshold:
                continue

            if detections["text_queries"][label_idx] != query:
                continue

            cx, cy, _, _ = box # Get center coordinates of the detected box
            px = int(cx * self.env.cam_params["width"]) # Convert pixel coordinates to world coordinates for bounds checking
            py = int(cy * self.env.cam_params["height"])
            world_coord = self.env.pixel_to_world(px, py)

            if world_coord is None or not self.env.is_within_bounds(world_coord): # Filter out detections not within workspace bounds
                continue

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "box_norm": box,
                    "label": query,
                    "world_coord": world_coord # Store world coordinates in best instance
                }

        return best

    def plot(self, image, detections, best=None, plot_only_best = True):
        H, W = image.shape[:2]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        ax.axis("off")

        if not plot_only_best:
          for score, box, label in zip(
              detections["scores"],
              detections["boxes"],
              detections["labels"],
          ):
              if score < self.score_threshold:
                  continue

              cx, cy, w, h = box
              x0 = (cx - w / 2) * W
              x1 = (cx + w / 2) * W
              y0 = (cy - h / 2) * H
              y1 = (cy + h / 2) * H

              ax.plot([x0, x1, x1, x0, x0],
                      [y0, y0, y1, y1, y0], "r")

        # Plot the best detection in green edge
        if best is not None:
            cx, cy, w, h = best["box_norm"]
            x0 = (cx - w/2) * W
            x1 = (cx + w/2) * W
            y0 = (cy - h/2) * H
            y1 = (cy + h/2) * H

            ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],"g")
            ax.text(x0, y1, f"{best['label']} ({best['score']:.2f})",
                    color="green",
                    bbox=dict(facecolor="white", edgecolor="green"))

        plt.show()

    def detect_and_refine(self, image, query):
        # Initial Object Detection
        initial_detections = self.detect(image, [query])
        initial_best_detection = self.get_best_instance(initial_detections, query)

        if not initial_best_detection:
            print(f"Initial detection for '{query}' failed.")
            return None, None

        # Crop the image based on initial detection
        cx_initial, cy_initial, w_initial, h_initial = initial_best_detection['box_norm']
        H_orig, W_orig, _ = image.shape
        x0_orig = int((cx_initial - w_initial / 2) * W_orig)
        y0_orig = int((cy_initial - h_initial / 2) * H_orig)
        x1_orig = int((cx_initial + w_initial / 2) * W_orig)
        y1_orig = int((cy_initial + h_initial / 2) * H_orig)

        x0_orig = max(0, x0_orig)
        y0_orig = max(0, y0_orig)
        x1_orig = min(W_orig, x1_orig)
        y1_orig = min(H_orig, y1_orig)

        cropped_image = image[y0_orig:y1_orig, x0_orig:x1_orig]

        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            print("Cropped image is empty or invalid.")
            return None, None

        # Refined Object Detection on cropped image
        refined_detections = self.detect(cropped_image, [query])
        refined_best_detection = self.get_best_instance(refined_detections, query)

        if not refined_best_detection:
            print(f"Refined detection for '{query}' failed on cropped image.")
            return initial_best_detection, None

        # Transform refined bounding box coordinates back to original image's frame
        cx_refined_norm_cropped, cy_refined_norm_cropped, w_refined_norm_cropped, h_refined_norm_cropped = refined_best_detection['box_norm']

        H_cropped, W_cropped, _ = cropped_image.shape
        x0_refined_cropped = int((cx_refined_norm_cropped - w_refined_norm_cropped / 2) * W_cropped)
        y0_refined_cropped = int((cy_refined_norm_cropped - h_refined_norm_cropped / 2) * H_cropped)
        x1_refined_cropped = int((cx_refined_norm_cropped + w_refined_norm_cropped / 2) * W_cropped)
        y1_refined_cropped = int((cy_refined_norm_cropped + h_refined_norm_cropped / 2) * H_cropped)

        x0_refined_orig = x0_orig + x0_refined_cropped
        y0_refined_orig = y0_orig + y0_refined_cropped
        x1_refined_orig = x0_orig + x1_refined_cropped
        y1_refined_orig = y0_orig + y1_refined_cropped

        # Recalculate normalized box for the refined detection in original image coordinates
        new_cx_norm = (x0_refined_orig + x1_refined_orig) / (2 * W_orig)
        new_cy_norm = (y0_refined_orig + y1_refined_orig) / (2 * H_orig)
        new_w_norm = (x1_refined_orig - x0_refined_orig) / W_orig
        new_h_norm = (y1_refined_orig - y0_refined_orig) / H_orig

        # Create a new 'best' dictionary for the refined detection, similar to get_best_instance's output
        refined_best_detection_orig_coords = {
            'score': refined_best_detection['score'],
            'box_norm': np.array([new_cx_norm, new_cy_norm, new_w_norm, new_h_norm], dtype=np.float32),
            'label': query,
            'world_coord': refined_best_detection['world_coord'] # This is already in original world coords
        }

        return initial_best_detection, refined_best_detection_orig_coords
    
# Test detector

setup_env(retracted_arm=False)

env._setup_camera(side_view=False) # comment this line out if sideview isnt needed

detector = Detector(device=device, score_threshold=1e-3, env=env) # Score threshold of 0.0 evaluates all the detections to find the best one.

image = env.render()
text_queries = ["shoe", "robot arm"]
detections = detector.detect(image, text_queries)
best = detector.get_best_instance(detections, text_queries[0]) # Can find best for each label. Loop over to find for all instances

detector.plot(image, detections, best, plot_only_best=True)

if best is not None:
    cx, cy, _, _ = best["box_norm"]
    px = int(cx * image.shape[1])
    py = int(cy * image.shape[0])
    world = env.pixel_to_world(px, py)
    world[2]=0.65
    print(world)

import os
import json
from groq import Groq
from google.colab import userdata

class GroqCommandParser:
    def __init__(self, api_key):
        # We pass the key in explicitly for simplicity
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def _get_system_prompt(self):
        """
        Returns the system prompt definitions.
        """
        return """
        You are a robotic command parser. Extract "pick" and "place" actions.

        Rules:
        1. Output a JSON list of lists: [["object_name", "target_location"], ...]
        2. Normalize strict corners ONLY:
           - "top right corner" -> "trc"
           - "top left corner" -> "tlc"
           - "middle" -> "mid"
           - "bottom right corner" -> "brc"
           - "bottom left corner" -> "blc"
           - "center" -> "mid"
        3. CRITICAL: If the location is a named object (e.g., "next to the bear"), KEEP the object name (e.g., "bear"). Do NOT invent codes.
        4. Return ONLY the JSON. No explanation.
        """

    def extract_commands(self, user_input):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f'Input: "{user_input}"'}
                ],
                model=self.model,
                temperature=0,
            )

            # Clean and Parse
            response_text = chat_completion.choices[0].message.content
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            return [tuple(x) for x in json.loads(cleaned_text)]

        except Exception as e:
            print(f"Error: {e}")
            return []


# Get key securely from Colab Secrets
try:
    my_api_key = userdata.get('GROQ_API_KEY')
except ImportError:
    # Fallback if not running in Colab or Secrets not set
    print("Add API key in Secrets")

# Instantiate the class
parser = GroqCommandParser(api_key=my_api_key)

# Test it
text = "Pick the soda and place in top right corner. Then pick the apple and put it next to the shoes."
commands = parser.extract_commands(text)

print("Robot Plan:")
print(commands)

# Setup openai
import openai
from google.colab import userdata
openai_api_key = userdata.get("OPENAI_API_KEY")
openai.api_key = openai_api_key
ENGINE = "gpt-3.5-turbo-instruct"

#LLM Cache
overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}

def gpt3_call(engine, prompt, max_tokens=1, temperature=0, logprobs=1, echo=True):
    if isinstance(prompt, str):
        prompt = [prompt]

    full_query = "".join(prompt)
    key = (engine, full_query, max_tokens, temperature, logprobs, echo)

    if key in LLM_CACHE:
        return LLM_CACHE[key]

    response = openai.Completion.create(
        model=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
        echo=echo,
    )

    LLM_CACHE[key] = response
    return response

def gpt3_scoring_next_step(
    system_prompt: str,
    task_instruction: str,
    options: list[str],
    engine=ENGINE,
    verbose=False,
):
    scores = {}

    for option in options:
        prompt = f"""{system_prompt} \n

TASK: {task_instruction}. \n STEP 1:"""

        response = openai.Completion.create(
            model=engine,
            prompt=prompt,
            suffix=" " + option,   # force continuation
            max_tokens=len(option.split()) + 2,
            temperature=0.0,
            logprobs=1,
            echo=False,
        )

        choice = response.choices[0]

        # Sum logprobs of GENERATED tokens (the option)
        logprob = sum(
            lp for lp in choice.logprobs.token_logprobs if lp is not None
        )

        scores[option] = logprob

        if verbose:
            print(f"{logprob:.3f}\t{option}")

    return scores



def make_options(pick_targets=None, place_targets=None, options_in_api_form=False, termination_string="done()"):
  options = []
  for pick in pick_targets:
    for place in place_targets:
      if options_in_api_form:
        option = "robot.pick_and_place({}, {})".format(pick, place)
      else:
        option = "Pick the {} and place it on the {}.".format(pick, place)
      options.append(option)

  options.append(termination_string)
  print("Considering", len(options), "options")
  return options


PICK_TARGETS = ["Bus", "Shoe"]
PLACE_TARGETS = ["Bus", "Shoe", "top right corner", "middle", "bottom left corner", "bottom right corner"]

SYSTEM_PROMPT = (
    "You are a helpful household robot. "
    "You execute tasks by choosing the correct next action."
)

TASK = "Put the footwear over the vehicle."

options = make_options(PICK_TARGETS, PLACE_TARGETS)

scores = gpt3_scoring_next_step(
    system_prompt=SYSTEM_PROMPT,
    task_instruction=TASK,
    options=options,
    engine=ENGINE,
    verbose=True,
)

print("\nTop ranked actions:")
for action, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"{score:.3f} | {action}")


from moviepy.editor import VideoFileClip, concatenate_videoclips
# SETTING UP THE ENV

# Setup Env
env = TableTopEnv(config=None)
# Define objects
n_objs = 2
# Obj locs
generated_coords = env.generate_random_coordinates(
    n=2,
    min_distance=0.5
)
# Download and Load objects for scene
use_gso = True
load_from_drive= True

if use_gso:
    if load_from_drive:
      objects = {
        "obj1": {
            "path": "Can/Can.urdf",
            "position": generated_coords[0],
            "scale": 1.0,
        },
        "obj2": {
            "path": "Shoe/Shoe.urdf",
            "position": generated_coords[1],
            "scale": 1.0,
        },
    }
else:
    # Default cube
    objects = {
        "cube": {
            "path": "cube.urdf",
            "position": [0.85, -0.6, 0.65],
            "scale": 0.1,
        }
    }
# Reset env loading scene and objects
env.reset(objects, retracted_arm=False, side_view=True, use_gso=True)


# Get key securely from Colab Secrets
try:
    my_api_key = userdata.get('GROQ_API_KEY')
except ImportError:
    # Fallback if not running in Colab or Secrets not set
    print("Add API key in Secrets")

# Instantiate the class
parser = GroqCommandParser(api_key=my_api_key)

# Enter Instruction Here
instruction = "Pick up the shoe and place near the can."
commands = parser.extract_commands(instruction)


pick = [cmd[0] for cmd in commands]
place = [cmd[1] for cmd in commands]

print(f"Pick: {pick}")
print(f"Place: {place}")


# List to keep track of individual video clips
video_segments = []

# Setup detector
detector = Detector(device=device, score_threshold=1e-3, env=env) # Score threshold of 0.0 evaluates all the detections to find the best one.

#  Iterate through actions
for k in range(len(pick)):
    print(f"--- Executing Step {k+1}: Pick {pick[k]} -> Place {place[k]} ---")

    image = env.render()

    text_queries = []
    world_coords = {}

    #  Determine what needs to be detected
    if place[k] in env.special_pts:
        # If target is a fixed location (like 'trc'), we already have coords.
        # We only need to detect the object to pick.
        world_coords[place[k]] = env.special_pts[place[k]]
        text_queries.append(pick[k])
    else:
        # If target is relative (e.g., "next to shoes"), we must detect both.
        text_queries.append(pick[k])
        text_queries.append(place[k])

    #  Run Detector on the fresh image
    detections = detector.detect(image, text_queries)

    #  Process Detections (Pixel -> World)
    for query in text_queries:
        best = detector.get_best_instance(detections, query)
        if best is not None:
            cx, cy, _, _ = best["box_norm"]
            px = int(cx * image.shape[1])
            py = int(cy * image.shape[0])
            world_coords[query] = env.pixel_to_world(px, py)
        else:
            print(f"WARNING: Could not find '{query}' in the scene!")

    #  Execute Move if coordinates are found
    if pick[k] in world_coords and place[k] in world_coords:
        pick_src = world_coords[pick[k]]
        place_tgt = world_coords[place[k]]

        # Save each action as a separate file to avoid overwriting
        step_video_path = f"action_{k}.mp4"

        env.execute_pick_and_place(
            pick_src,
            place_tgt,
            record_video=True,
            video_path=step_video_path
        )

        # Add to list for stitching later
        if os.path.exists(step_video_path):
            video_segments.append(step_video_path)

    else:
        print(f"Skipping step {k+1} due to missing detection.")

# Stitch Videos into One "Single Output"
if video_segments:
    print("Stitching videos...")
    try:
        clips = [VideoFileClip(v) for v in video_segments]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile("full_mission.mp4")
    except Exception as e:
        print(f"Error stitching videos: {e}")
        # Fallback: just play the last action if stitching fails
else:
    print("No actions were executed successfully.")

# Cleanup
env.close()

# Play the combined video
play_video("full_mission.mp4")
