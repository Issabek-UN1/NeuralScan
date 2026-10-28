"""
Metric Alignment Layer (Scale Ambiguity Resolution via OLS Regression)
Pairs relative monocular depth D_AI in [0, 1] with laser ground truth Z_metric.
Computes scale (s) and shift (t) to generate dense metric point clouds in mm.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import config

logger = logging.getLogger("NeuralScan.Fusion.MetricAligner")


class MetricAligner:
    """
    Solves monocular scale ambiguity by fitting a 1D linear regression:
        Z_dense = s * D_AI + t
    using laser triangulation ground-truth anchor coordinates.
    """

    def __init__(
        self,
        focal_length: float = config.FOCAL_LENGTH_PX,
        x0: float = config.PRINCIPAL_POINT_X,
        y0: float = config.PRINCIPAL_POINT_Y,
        min_laser_points: int = config.MIN_LASER_POINTS_FOR_ALIGNMENT,
        ransac_threshold_mm: float = config.RANSAC_RESIDUAL_THRESHOLD_MM
    ):
        self.f = focal_length
        self.x0 = x0
        self.y0 = y0
        self.min_laser_points = min_laser_points
        self.ransac_threshold_mm = ransac_threshold_mm

    def align_and_densify(
        self,
        depth_ai: np.ndarray,
        laser_data: Dict[str, np.ndarray],
        mask_object: np.ndarray,
        bgr_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Executes scale alignment and back-projects dense object surface into 3D camera coordinates.

        Returns:
            points_3d: (N, 3) float32 array in metric mm
            colors_rgb: (N, 3) float32 array in range [0.0, 1.0]
            s: scale factor (mm)
            t: translation shift (mm)
        """
        y_laser = laser_data["y"]
        xc_laser = laser_data["xc"]
        z_laser = laser_data["z_metric"]

        # 1. Collect paired samples (D_AI, Z_metric) along the laser line
        d_samples = []
        z_samples = []

        h, w = depth_ai.shape[:2]

        for y, xc, z_gt in zip(y_laser, xc_laser, z_laser):
            xi = int(round(xc))
            if 0 <= xi < w and 0 <= y < h:
                # Prioritize points belonging to the scanned object mask
                if mask_object[y, xi] > 0 or len(y_laser) < 50:
                    d_samples.append(depth_ai[y, xi])
                    z_samples.append(z_gt)

        # Fallback if laser points inside mask are scarce: take all valid laser points
        if len(d_samples) < self.min_laser_points and len(y_laser) >= self.min_laser_points:
            d_samples = [depth_ai[y, int(round(xc))] for y, xc in zip(y_laser, xc_laser) if 0 <= int(round(xc)) < w]
            z_samples = [z for y, xc, z in zip(y_laser, xc_laser, z_laser) if 0 <= int(round(xc)) < w]

        d_arr = np.array(d_samples, dtype=np.float64)
        z_arr = np.array(z_samples, dtype=np.float64)

        # 2. Ordinary Least Squares (OLS) with RANSAC-like outlier rejection
        s, t = self._fit_robust_ols(d_arr, z_arr)

        logger.debug(f"Metric alignment OLS: Scale s = {s:.3f} mm, Shift t = {t:.3f} mm across {len(d_arr)} points.")

        # 3. Dense depth reconstruction over Mask_object
        # Z_dense(x, y) = s * D_AI(x, y) + t
        obj_y, obj_x = np.where(mask_object > 0)
        
        if len(obj_y) == 0:
            logger.warning("Empty object mask detected. Returning empty point cloud.")
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), s, t

        d_vals = depth_ai[obj_y, obj_x].astype(np.float64)
        z_dense = s * d_vals + t

        # Filter points within valid turntable physical envelope
        valid_idx = (z_dense >= config.Z_MIN_MM) & (z_dense <= config.Z_MAX_MM)
        
        if not np.any(valid_idx):
            logger.warning("All points fell outside valid metric bounds [Z_min, Z_max].")
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), s, t

        y_valid = obj_y[valid_idx]
        x_valid = obj_x[valid_idx]
        z_valid = z_dense[valid_idx].astype(np.float32)

        # 4. Pinhole back-projection to 3D Camera Coordinates (mm)
        x_3d = (x_valid.astype(np.float32) - self.x0) * z_valid / self.f
        y_3d = (y_valid.astype(np.float32) - self.y0) * z_valid / self.f
        points_3d = np.column_stack((x_3d, y_3d, z_valid))

        # 5. Extract RGB point colors
        if bgr_image is not None:
            # OpenCV BGR -> RGB normalized to [0, 1]
            bgr_pixels = bgr_image[y_valid, x_valid]
            colors_rgb = (bgr_pixels[:, ::-1] / 255.0).astype(np.float32)
        else:
            colors_rgb = np.full_like(points_3d, 0.8, dtype=np.float32)

        return points_3d, colors_rgb, s, t

    def _fit_robust_ols(self, d_arr: np.ndarray, z_arr: np.ndarray) -> Tuple[float, float]:
        """
        Performs OLS regression: s = Cov(D, Z) / Var(D), t = mean(Z) - s * mean(D)
        with residual-based outlier elimination.
        """
        if len(d_arr) < 5 or np.var(d_arr) < 1e-7:
            # Default fallback based on working distance if data is degraded
            logger.warning("Degraded laser data for OLS. Using default geometric anchors.")
            return -60.0, float(config.WORKING_DISTANCE_MM)

        # Initial pass
        var_d = np.var(d_arr)
        cov_dz = np.cov(d_arr, z_arr)[0, 1]
        s = float(cov_dz / var_d)
        t = float(np.mean(z_arr) - s * np.mean(d_arr))

        # Second pass: trim 15% highest residuals for robust fit
        residuals = np.abs(z_arr - (s * d_arr + t))
        inlier_mask = residuals <= np.percentile(residuals, 85)

        if np.sum(inlier_mask) >= 5 and np.var(d_arr[inlier_mask]) > 1e-7:
            d_inl = d_arr[inlier_mask]
            z_inl = z_arr[inlier_mask]
            var_d = np.var(d_inl)
            cov_dz = np.cov(d_inl, z_inl)[0, 1]
            s = float(cov_dz / var_d)
            t = float(np.mean(z_inl) - s * np.mean(d_inl))

        return s, t
