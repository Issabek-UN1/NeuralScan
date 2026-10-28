"""
NeuralScan: Edge-AI Hybrid 3D Metrology Scanner
Global Configuration & Hardware/Model Hyperparameters
"""

import os
from pathlib import Path
import numpy as np
import torch

# ==============================================================================
# Hardware Communication & UART Parameters
# ==============================================================================
SERIAL_PORT = os.getenv("NEURALSCAN_PORT", "COM3")
BAUD_RATE = 115200
SERIAL_TIMEOUT_S = 2.0

# Mechanical Settling Time (strictly 80-120 ms to dampen turntable microvibrations)
SETTLING_TIME_S = 0.100

# ==============================================================================
# Optical & Camera Sensor Parameters
# ==============================================================================
CAMERA_INDEX = int(os.getenv("NEURALSCAN_CAM_INDEX", "0"))
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CAMERA_FPS = 30

# Fixed exposure & white balance control
AUTO_EXPOSURE_VAL = 0.25 # cv2.CAP_PROP_AUTO_EXPOSURE flag for manual mode in OpenCV
MANUAL_EXPOSURE_VAL = -6 # Manual exposure step (sensor specific)
CAMERA_WARMUP_FRAMES = 5

# ==============================================================================
# Metric Geometry & Laser Triangulation Rig
# ==============================================================================
# b: Baseline distance between camera optical center and laser emitter (mm)
BASELINE_MM = 160.0

# Z0: Calibrated distance from camera sensor plane to turntable rotation axis (mm)
# Updated to 300 mm (30 cm) based on physical optical rig geometry
WORKING_DISTANCE_MM = float(os.getenv("NEURALSCAN_DISTANCE_MM", "300.0"))

# alpha: Convergence angle of the laser sheet relative to camera optical axis (deg)
LASER_ANGLE_DEG = 28.5
LASER_ANGLE_RAD = np.deg2rad(LASER_ANGLE_DEG)

# Camera Intrinsic Estimates (Full HD 1920x1080)
# f: Focal length in pixels. Calibrated or estimated for ~61 deg horizontal FOV:
FOCAL_LENGTH_PX = 1620.0
PRINCIPAL_POINT_X = 960.0 # x0
PRINCIPAL_POINT_Y = 540.0 # y0

# Turntable calibrated axis in metric camera space (mm)
TURNTABLE_AXIS_X = 0.0
TURNTABLE_AXIS_Z = WORKING_DISTANCE_MM

# Physical Work Volume & Anti-Hallucination Envelope (mm)
TURNTABLE_RADIUS_MAX_MM = 110.0 # Bounding cylinder radius for 200-220 mm turntable platter
TURNTABLE_HEIGHT_MAX_MM = 140.0 # Maximum scanned object height +/- 140 mm from turntable center

# Stepper kinematics (Mitsumi M35SP-7T)
TOTAL_STEPS = 48
STEP_ANGLE_DEG = 7.5 # 360.0 / 48 = 7.5 deg

# ==============================================================================
# Analytical Laser Line Extraction Hyperparameters
# ==============================================================================
# Peak exponent (gamma >= 2) for weighted center of mass sharpening
LASER_GAMMA = 3.0

# Horizontal peak integration window W: [x_max - WINDOW_RADIUS, x_max + WINDOW_RADIUS]
WINDOW_RADIUS = 15

# Intensity difference thresholds to suppress ambient illumination noise
MIN_DIFF_THRESHOLD = 20
MIN_PEAK_INTENSITY = 35

# Metric Z bounds for physical filtering (reject rays outside the turntable volume)
Z_MIN_MM = WORKING_DISTANCE_MM - 140.0 # 110 mm
Z_MAX_MM = WORKING_DISTANCE_MM + 140.0 # 390 mm

# ==============================================================================
# Edge-AI Deep Learning Architecture
# ==============================================================================
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available() # Strict FP16 requirement on RTX 4060
REMBG_MODEL = "u2net"

# ==============================================================================
# 3D Metric Alignment & Point Cloud Fusion
# ==============================================================================
# Minimum valid laser points required per frame to compute robust OLS scale/shift
MIN_LASER_POINTS_FOR_ALIGNMENT = 25

# RANSAC / IQR outlier filtering bounds for OLS regression
RANSAC_RESIDUAL_THRESHOLD_MM = 4.0

# Downsampling and Statistical Outlier Removal (SOR)
VOXEL_SIZE_MM = 0.8
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

# Screened Poisson Surface Reconstruction
POISSON_DEPTH = 8
DENSITY_TRIM_PERCENTILE = 5.0 # Trim spurious vertices from low-density boundaries

# ==============================================================================
# File System Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
FRAMES_DIR = OUTPUT_DIR / "frames"
MODELS_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
