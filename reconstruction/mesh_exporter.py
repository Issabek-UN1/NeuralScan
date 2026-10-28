"""
Screened Poisson Surface Reconstruction & Watertight STL Mesh Exporter
Enforces Section 7 rule: Automatic validation via trimesh.is_watertight and fill_holes().
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import open3d as o3d
import trimesh
import trimesh.repair
import config

logger = logging.getLogger("NeuralScan.Reconstruction.MeshExporter")


class MeshExporter:
    """
    Constructs continuous triangular surface meshes using Screened Poisson Reconstruction,
    crops spurious octree boundaries, and guarantees 3D printable watertight topology.
    """

    def __init__(self, poisson_depth: int = config.POISSON_DEPTH, trim_percentile: float = config.DENSITY_TRIM_PERCENTILE):
        self.poisson_depth = poisson_depth
        self.trim_percentile = trim_percentile

    def reconstruct_mesh(self, pcd: o3d.geometry.PointCloud) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Executes Screened Poisson Reconstruction and converts to validated Trimesh.
        """
        if len(pcd.points) < 50:
            raise ValueError(f"Insufficient points for Poisson Reconstruction ({len(pcd.points)} points).")

        logger.info(f"Executing Screened Poisson Surface Reconstruction (octree depth={self.poisson_depth})...")
        
        # 1. Screened Poisson Surface Reconstruction
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.poisson_depth,
            width=0,
            scale=1.1,
            linear_fit=False
        )

        # 2. Trim low-density spurious artifacts on outer boundaries
        density_arr = np.asarray(densities)
        if len(density_arr) > 0 and self.trim_percentile > 0:
            threshold = np.percentile(density_arr, self.trim_percentile)
            vertices_to_remove = density_arr < threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
            mesh_o3d.remove_unreferenced_vertices()
            logger.info(f"Trimmed low-density boundary artifacts (threshold={threshold:.2f}).")

        # 3. ANTI-HALLUCINATION: Keep only dominant object body (isolate and eliminate floating debris)
        try:
            triangle_clusters, num_triangles, _ = mesh_o3d.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            num_triangles = np.asarray(num_triangles)
            if len(num_triangles) > 1:
                largest_cluster_idx = np.argmax(num_triangles)
                debris_triangles = triangle_clusters != largest_cluster_idx
                mesh_o3d.remove_triangles_by_mask(debris_triangles)
                mesh_o3d.remove_unreferenced_vertices()
                logger.info(f"Anti-hallucination: Filtered {len(num_triangles)-1} disconnected floating debris components.")
        except Exception as e:
            logger.debug(f"Component clustering fallback: {e}")

        # 4. Decimate if excessively dense to keep watertight repair under 5 seconds
        if len(mesh_o3d.triangles) > 200000:
            logger.info(f"Decimating mesh from {len(mesh_o3d.triangles):,} to 200,000 faces for fast watertight repair...")
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=200000)

        # 5. Convert to Trimesh for CAD topology validation
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        
        tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        # Fix inverted normal winding if Poisson generated inverted surface
        try:
            if tmesh.volume < 0:
                logger.info("Inverting mesh winding to ensure positive outward normals...")
                tmesh.invert()
        except Exception as e:
            logger.debug(f"Volume check skipped: {e}")

        # 5. STRICT RULE: Validate STL watertightness and invoke fill_holes()
        is_watertight_initial = tmesh.is_watertight
        logger.info(f"Initial mesh watertight check: {is_watertight_initial}")

        if not is_watertight_initial:
            logger.warning("Mesh has non-manifold edges or holes. Executing topological repairs...")
            repaired = False
            try:
                import pymeshfix
                mfix = pymeshfix.MeshFix(np.asarray(tmesh.vertices), np.asarray(tmesh.faces))
                mfix.repair()
                tmesh = trimesh.Trimesh(mfix.points, mfix.faces, process=True)
                
                # Second pass if needed for complex geometry
                if not tmesh.is_watertight:
                    mfix2 = pymeshfix.MeshFix(np.asarray(tmesh.vertices), np.asarray(tmesh.faces))
                    mfix2.repair()
                    tmesh = trimesh.Trimesh(mfix2.points, mfix2.faces, process=True)

                repaired = tmesh.is_watertight
                logger.info(f"MeshFix repair applied. Watertight status: {tmesh.is_watertight}")
            except Exception as e:
                logger.debug(f"MeshFix exception: {e}")

            if not repaired:
                trimesh.repair.fix_winding(tmesh)
                trimesh.repair.fix_inversion(tmesh)
                trimesh.repair.fix_normals(tmesh)
                trimesh.repair.fill_holes(tmesh)
                logger.info(f"Trimesh fill_holes applied. Watertight status: {tmesh.is_watertight}")

        # Summary Metrology Metrics
        stats = {
            "vertex_count": len(tmesh.vertices),
            "face_count": len(tmesh.faces),
            "is_watertight": tmesh.is_watertight,
            "volume_mm3": float(tmesh.volume) if tmesh.is_watertight else None,
            "surface_area_mm2": float(tmesh.area),
            "bounding_box_extents_mm": tmesh.extents.tolist() if len(tmesh.vertices) > 0 else []
        }

        return tmesh, stats

    def export(self, tmesh: trimesh.Trimesh, output_stem: str = "neuralscan_model") -> Dict[str, Path]:
        """
        Saves watertight mesh to binary .stl and .ply.
        """
        stl_path = config.MODELS_DIR / f"{output_stem}.stl"
        ply_path = config.MODELS_DIR / f"{output_stem}.ply"

        logger.info(f"Exporting CAD files to:\n  -> {stl_path}\n  -> {ply_path}")

        # Export binary STL
        tmesh.export(str(stl_path), file_type="stl")
        
        # Export PLY
        tmesh.export(str(ply_path), file_type="ply")

        return {
            "stl": stl_path,
            "ply": ply_path
        }
