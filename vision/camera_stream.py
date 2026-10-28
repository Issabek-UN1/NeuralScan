"""
Camera Stream & Dual-Frame Capture Engine with Fixed Exposure Control
Supports physical USB cameras and high-precision synthetic simulation.
"""

import time
import logging
from typing import Tuple, Optional
import cv2
import numpy as np
import config

logger = logging.getLogger("NeuralScan.Vision")


class CameraStream:
    """
    Manages USB camera capture with fixed exposure, manual white balance,
    and synchronous dual-frame (Laser ON / Laser OFF) acquisition.
    """

    def __init__(self, cam_index: int = config.CAMERA_INDEX, is_mock: bool = False):
        self.cam_index = cam_index
        self.is_mock = is_mock
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_open = False
        self.width = config.FRAME_WIDTH
        self.height = config.FRAME_HEIGHT

    def open(self) -> bool:
        """
        Initializes camera device and locks auto-exposure and white balance.
        """
        if self.is_mock:
            self.is_open = True
            logger.info("[MOCK] Virtual Full HD Camera stream initialized.")
            return True

        logger.info(f"Connecting to USB camera index {self.cam_index} (1080p)...")
        # Try DirectShow on Windows for fastest and most reliable capture
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        
        if not self.cap or not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.cam_index)

        if not self.cap or not self.cap.isOpened():
            logger.warning(f"Could not open physical camera at index {self.cam_index}.")
            self.is_open = False
            return False

        # Configure Full HD resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        # STRICT REGULATION: Fix exposure and disable auto-gain/auto-white-balance
        # On Windows OpenCV, cv2.CAP_PROP_AUTO_EXPOSURE = 0.25 (or 1 / manual mode)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.AUTO_EXPOSURE_VAL)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, config.MANUAL_EXPOSURE_VAL)
            self.cap.set(cv2.CAP_PROP_AUTOWBFINE, 0)
        except Exception as e:
            logger.debug(f"Exposure lock warning (sensor specific): {e}")

        # Sensor warm-up / flush stale buffer frames
        for _ in range(config.CAMERA_WARMUP_FRAMES):
            self.cap.read()

        self.is_open = True
        logger.info(f"Camera opened successfully: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return True

    def read_frame(self) -> np.ndarray:
        """
        Reads a single BGR frame from the camera.
        """
        if self.is_mock or not self.is_open or self.cap is None:
            return self._generate_synthetic_frame(step_index=0, laser_active=False)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame from camera. Returning blank frame.")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return frame

    def capture_pair(self, controller, step_index: int = 0, total_steps: int = config.TOTAL_STEPS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes dual-frame acquisition:
        1. Ambient frame (Laser OFF) -> I_ambient
        2. Laser active frame (Laser ON) -> I_laser_on
        3. Difference frame -> I_diff = max(I_laser_on - I_ambient, 0)
        """
        if self.is_mock:
            ambient = self._generate_synthetic_frame(step_index, laser_active=False, total_steps=total_steps)
            laser_on = self._generate_synthetic_frame(step_index, laser_active=True, total_steps=total_steps)
            diff = self.compute_difference_frame(ambient, laser_on)
            return ambient, laser_on, diff

        # 1. Turn Laser OFF -> capture ambient frame
        controller.laser_off()
        time.sleep(0.035) # settle optical state
        ambient = self.read_frame()

        # 2. Turn Laser ON -> capture laser active frame
        controller.laser_on()
        time.sleep(0.040)
        laser_on = self.read_frame()

        # 3. Compute differential frame
        diff = self.compute_difference_frame(ambient, laser_on)
        return ambient, laser_on, diff

    @staticmethod
    def compute_difference_frame(ambient: np.ndarray, laser_on: np.ndarray) -> np.ndarray:
        """
        Computes calibrated difference frame isolating the laser line.
        Weights red channel higher since KY-008 is 650nm red.
        """
        # Convert to 16-bit to prevent uint8 underflow
        diff_bgr = np.clip(laser_on.astype(np.int16) - ambient.astype(np.int16), 0, 255).astype(np.uint8)

        # Enhanced red-channel weighting for 650nm laser:
        # Red channel minus mean of blue/green eliminates neutral ambient reflections
        b, g, r = cv2.split(diff_bgr)
        red_diff = np.clip(r.astype(np.int16) - (b.astype(np.int16) + g.astype(np.int16)) // 2, 0, 255).astype(np.uint8)

        # Threshold low noise
        _, red_thresh = cv2.threshold(red_diff, config.MIN_DIFF_THRESHOLD, 255, cv2.THRESH_TOZERO)
        return red_thresh

    def _generate_synthetic_frame(self, step_index: int, laser_active: bool, total_steps: int = config.TOTAL_STEPS) -> np.ndarray:
        """
        Generates realistic synthetic test frame simulating a 3D target on the turntable.
        Used for verification, test suites, and mock mode.
        """
        frame = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        
        # Draw background turntable and environment
        cv2.ellipse(frame, (960, 850), (380, 70), 0, 0, 360, (50, 50, 55), -1)
        cv2.ellipse(frame, (960, 850), (380, 70), 0, 0, 360, (70, 70, 75), 2)

        theta = np.deg2rad(step_index * (360.0 / max(total_steps, 1)))
        
        # Synthetic object: Fluted architectural vase / gear profile
        # Parametric radius: R(y, theta)
        center_x = 960
        y_top, y_bottom = 320, 840
        
        # Render shaded 3D object body
        for y in range(y_top, y_bottom, 2):
            norm_y = (y - y_top) / (y_bottom - y_top)
            # Profile radius with fluting
            base_r = 110 + 45 * np.sin(norm_y * np.pi) + 20 * np.cos(norm_y * 3 * np.pi)
            # Asymmetry modulated by rotation theta
            r_val = base_r * (1.0 + 0.15 * np.cos(4 * theta + norm_y * 2))
            
            x1 = int(center_x - r_val)
            x2 = int(center_x + r_val)
            
            # Simple Lambertian shading across width
            shade_grad = np.linspace(90, 180, x2 - x1, endpoint=True)
            for xi, s in zip(range(x1, x2), shade_grad):
                frame[y:y+2, xi] = [int(s * 0.75), int(s * 0.85), int(s * 0.95)]

        # If laser is active, project red laser line using triangulation formula
        if laser_active:
            tan_alpha = np.tan(config.LASER_ANGLE_RAD)
            for y in range(y_top, y_bottom):
                norm_y = (y - y_top) / (y_bottom - y_top)
                base_r = 110 + 45 * np.sin(norm_y * np.pi) + 20 * np.cos(norm_y * 3 * np.pi)
                r_val = base_r * (1.0 + 0.15 * np.cos(4 * theta + norm_y * 2))
                
                # Metric depth Z of front surface:
                # Turntable center is at Z0 = 250 mm.
                # Surface front point is at Z = Z0 - delta_Z(theta)
                z_surface = config.WORKING_DISTANCE_MM - (r_val * 0.45 * np.cos(theta))
                
                # Triangulation: Z = (b * f) / ((x_c - x0) + f * tan_alpha)
                # => x_c - x0 = (b * f) / Z - f * tan_alpha
                # => x_c = x0 + (b * f) / Z - f * tan_alpha
                xc = config.PRINCIPAL_POINT_X + (config.BASELINE_MM * config.FOCAL_LENGTH_PX) / z_surface - (config.FOCAL_LENGTH_PX * tan_alpha)
                xc_int = int(round(xc))

                # Draw Gaussian laser profile with saturation at peak
                for dx in range(-4, 5):
                    intensity = int(255 * np.exp(-0.5 * (dx / 1.2) ** 2))
                    cur_x = xc_int + dx
                    if 0 <= cur_x < self.width:
                        cur_val = frame[y, cur_x]
                        frame[y, cur_x] = [
                            min(255, cur_val[0] + intensity // 6),
                            min(255, cur_val[1] + intensity // 4),
                            min(255, cur_val[2] + intensity)
                        ]

        return frame

    def close(self):
        """
        Releases camera handle.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.is_open = False
            logger.info("Camera stream closed.")


def get_camera_stream(cam_index: Optional[int] = None, force_mock: bool = False) -> CameraStream:
    """
    Factory: returns working CameraStream with auto-fallback to mock if physical camera is absent.
    """
    if force_mock:
        stream = CameraStream(is_mock=True)
        stream.open()
        return stream

    target_index = config.CAMERA_INDEX if cam_index is None else cam_index
    stream = CameraStream(cam_index=target_index, is_mock=False)
    if stream.open():
        return stream

    logger.warning(f"Physical camera at index {target_index} not found. Automatically switching to high-fidelity virtual camera.")
    mock_stream = CameraStream(is_mock=True)
    mock_stream.open()
    return mock_stream
