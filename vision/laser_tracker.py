"""
Analytical Subpixel Laser Line Extraction & Optical Triangulation Engine
Computes Center of Mass with gamma sharpening (execution time < 1 ms).
NO YOLO or discrete detectors permitted per Section 4.1 instructions.
"""

import logging
from typing import Dict, Tuple
import numpy as np
import config

logger = logging.getLogger("NeuralScan.Vision.LaserTracker")


class LaserTracker:
    """
    Subpixel laser stripe extractor using exponential weighted center of mass
    and metric optical triangulation.
    """

    def __init__(
        self,
        baseline: float = config.BASELINE_MM,
        focal_length: float = config.FOCAL_LENGTH_PX,
        alpha_rad: float = config.LASER_ANGLE_RAD,
        x0: float = config.PRINCIPAL_POINT_X,
        y0: float = config.PRINCIPAL_POINT_Y,
        gamma: float = config.LASER_GAMMA,
        window_radius: int = config.WINDOW_RADIUS,
        min_peak_intensity: int = config.MIN_PEAK_INTENSITY,
        z_min: float = config.Z_MIN_MM,
        z_max: float = config.Z_MAX_MM
    ):
        self.b = baseline
        self.f = focal_length
        self.alpha_rad = alpha_rad
        self.tan_alpha = np.tan(alpha_rad)
        self.x0 = x0
        self.y0 = y0
        self.gamma = gamma
        self.window_radius = window_radius
        self.min_peak_intensity = min_peak_intensity
        self.z_min = z_min
        self.z_max = z_max

    def extract_subpixel_laser(self, diff_frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Processes differential frame I_diff(x, y) to find subpixel laser peak coordinates
        and calculates physical 3D metric coordinates.

        Formula:
            x_c(y) = sum_{x in W} (x * (I_diff(x, y))^gamma) / sum_{x in W} ((I_diff(x, y))^gamma)
            Z_metric(x_c, y) = (b * f) / ((x_c - x0) + f * tan(alpha))

        Returns dictionary with:
            - 'y': row indices (int)
            - 'xc': subpixel horizontal center coordinates (float)
            - 'z_metric': metric depth in mm (float)
            - 'points_3d': 3D coordinates (X, Y, Z) in mm (M, 3)
        """
        h, w = diff_frame.shape[:2]
        
        # Ensure single-channel 2D difference image
        if diff_frame.ndim == 3:
            diff_gray = diff_frame[:, :, 2] # Use red channel if RGB
        else:
            diff_gray = diff_frame

        # Pre-allocate candidate storage
        valid_y = []
        valid_xc = []
        valid_z = []
        valid_x_3d = []
        valid_y_3d = []

        # Find maximum intensity across each row
        row_max_indices = np.argmax(diff_gray, axis=1)
        row_max_values = np.max(diff_gray, axis=1)

        # Vectorized candidate mask based on minimum peak intensity
        candidate_rows = np.where(row_max_values >= self.min_peak_intensity)[0]

        for y in candidate_rows:
            x_max = row_max_indices[y]
            
            # Form horizontal window W around peak
            w_left = max(0, x_max - self.window_radius)
            w_right = min(w, x_max + self.window_radius + 1)
            
            window_pixels = diff_gray[y, w_left:w_right].astype(np.float32)
            
            # Apply gamma sharpening: (I_diff)^gamma, gamma >= 2
            weights = np.power(window_pixels, self.gamma)
            weight_sum = np.sum(weights)

            if weight_sum < 1e-5:
                continue

            x_coords = np.arange(w_left, w_right, dtype=np.float32)
            
            # Subpixel Center of Mass
            xc = np.sum(x_coords * weights) / weight_sum

            # Optical Triangulation
            denominator = (xc - self.x0) + self.f * self.tan_alpha
            if np.abs(denominator) < 1e-4:
                continue

            z_metric = (self.b * self.f) / denominator

            # Reject physical outliers outside turntable work volume
            if self.z_min <= z_metric <= self.z_max:
                # Back-projection to metric camera frame
                x_metric = (xc - self.x0) * z_metric / self.f
                y_metric = (float(y) - self.y0) * z_metric / self.f

                valid_y.append(y)
                valid_xc.append(xc)
                valid_z.append(z_metric)
                valid_x_3d.append(x_metric)
                valid_y_3d.append(y_metric)

        if len(valid_y) == 0:
            return {
                "y": np.empty(0, dtype=np.int32),
                "xc": np.empty(0, dtype=np.float32),
                "z_metric": np.empty(0, dtype=np.float32),
                "points_3d": np.empty((0, 3), dtype=np.float32)
            }

        y_arr = np.array(valid_y, dtype=np.int32)
        xc_arr = np.array(valid_xc, dtype=np.float32)
        z_arr = np.array(valid_z, dtype=np.float32)
        pts_3d = np.column_stack((valid_x_3d, valid_y_3d, z_arr)).astype(np.float32)

        return {
            "y": y_arr,
            "xc": xc_arr,
            "z_metric": z_arr,
            "points_3d": pts_3d
        }
