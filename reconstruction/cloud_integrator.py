"""
3D Point Cloud Integrator & Multi-View Fusion Engine
Transforms 48 rotational angular frames into a coherent global 3D point cloud.
Applies Voxel Grid Homogenization, Statistical Outlier Removal (SOR), and Normal Estimation.
"""

import logging
from typing import List, Optional
import numpy as np
import open3d as o3d
import config

logger = logging.getLogger("NeuralScan.Reconstruction.CloudIntegrator")


class CloudIntegrator:
    """
    Accumulates, rotates, registers, and filters multi-view point clouds
    across the full 360-degree rotation (48 steps).
    """

    def __init__(
        self,
        voxel_size: float = config.VOXEL_SIZE_MM,
        sor_neighbors: int = config.SOR_NB_NEIGHBORS,
        sor_std_ratio: float = config.SOR_STD_RATIO,
        axis_x: float = config.TURNTABLE_AXIS_X,
        axis_z: float = config.TURNTABLE_AXIS_Z
    ):
        self.voxel_size = voxel_size
        self.sor_neighbors = sor_neighbors
        self.sor_std_ratio = sor_std_ratio
        self.axis_x = axis_x
        self.axis_z = axis_z

        self.all_points: List[np.ndarray] = []
        self.all_colors: List[np.ndarray] = []

    def add_view(self, points_cam: np.ndarray, colors_rgb: np.ndarray, step_index: int, total_steps: int = config.TOTAL_STEPS):
        """
        Transforms view point cloud at step_index into global world coordinates centered at turntable origin.
        """
        if len(points_cam) == 0:
            return

        theta_deg = step_index * (360.0 / max(total_steps, 1))
        # Reverse angle to map camera view back into static object reference frame
        theta_rad = -np.deg2rad(theta_deg)

        # 1. Translate relative to rotation axis
        p_rel = np.copy(points_cam)
        p_rel[:, 0] -= self.axis_x
        p_rel[:, 2] -= self.axis_z

        # 2. Rotation matrix R_y around vertical axis
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        r_y = np.array([
            [cos_t, 0.0, sin_t],
            [0.0,   1.0, 0.0  ],
            [-sin_t, 0.0, cos_t]
        ], dtype=np.float32)

        # 3. Rotate into global canonical frame
        p_world = np.dot(p_rel, r_y.T)

        # 4. Strict physical work volume boundary filtering (Anti-Hallucination)
        r_dist = np.sqrt(p_world[:, 0]**2 + p_world[:, 2]**2)
        valid_mask = (r_dist <= config.TURNTABLE_RADIUS_MAX_MM) & (np.abs(p_world[:, 1]) <= config.TURNTABLE_HEIGHT_MAX_MM)

        if np.any(valid_mask):
            self.all_points.append(p_world[valid_mask])
            self.all_colors.append(colors_rgb[valid_mask])

    def get_integrated_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        Merges all views, performs voxel downsampling, statistical outlier removal,
        and computes surface normal vectors.
        """
        if not self.all_points:
            logger.warning("No points accumulated in CloudIntegrator. Returning empty PointCloud.")
            return o3d.geometry.PointCloud()

        merged_pts = np.vstack(self.all_points).astype(np.float64)
        merged_colors = np.vstack(self.all_colors).astype(np.float64)

        logger.info(f"Merging raw point cloud: {len(merged_pts):,} total points.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_pts)
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        # 1. Voxel grid downsampling for spatial uniformity
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            logger.info(f"After voxel grid downsampling ({self.voxel_size} mm): {len(pcd.points):,} points.")

        # 2. Statistical Outlier Removal (SOR) to clean noise
        if len(pcd.points) > self.sor_neighbors:
            pcd_clean, inliers = pcd.remove_statistical_outlier(
                nb_neighbors=self.sor_neighbors,
                std_ratio=self.sor_std_ratio
            )
            logger.info(f"After Statistical Outlier Removal (SOR): {len(pcd_clean.points):,} points.")
            pcd = pcd_clean

        # 3. Surface normal estimation with consistent orientation
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(2.0, self.voxel_size * 3.0),
                max_nn=30
            )
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(k=15)
        except Exception:
            pass
        
        # Orient normals consistently outward from the turntable rotation axis (0, Y, 0)
        pts_arr = np.asarray(pcd.points)
        normals_arr = np.asarray(pcd.normals)
        radial_vecs = np.copy(pts_arr)
        radial_vecs[:, 1] = 0.0 # project onto horizontal XZ plane
        norm_lens = np.linalg.norm(radial_vecs, axis=1, keepdims=True) + 1e-6
        radial_unit = radial_vecs / norm_lens
        
        # Invert normals that point inward toward rotation axis
        inward_mask = np.sum(normals_arr * radial_unit, axis=1) < 0
        normals_arr[inward_mask] *= -1.0
        pcd.normals = o3d.utility.Vector3dVector(normals_arr)

        return pcd

    def clear(self):
        """Resets point buffer for fresh scan."""
        self.all_points.clear()
        self.all_colors.clear()
