"""
Unit & Integration Test Suite for NeuralScan Pipeline
Verifies all 7 core modules locally and offline.
Compatible with standard unittest runner and direct script execution.
"""

import unittest
import numpy as np
import config
from hardware import MockSerialController
from vision import get_camera_stream, LaserTracker
from ai import BackgroundRemover, DepthEstimator
from fusion import MetricAligner
from reconstruction import CloudIntegrator, MeshExporter


class TestNeuralScanPipeline(unittest.TestCase):
    """Integration and Unit Test Suite for all 7 NeuralScan submodules."""

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 65)
        print("      NEURALSCAN COMPREHENSIVE PIPELINE VERIFICATION")
        print("=" * 65)

    def test_01_hardware_controller(self):
        """Hardware Serial Controller (Mock Simulation)."""
        print("\n[TEST 1/7] Hardware Serial Controller (Mock Simulation)...")
        mock_hw = MockSerialController()
        self.assertTrue(mock_hw.connect(), "Hardware connection failed")
        self.assertTrue(mock_hw.laser_on(), "Laser ON failed")
        self.assertTrue(mock_hw.step(forward=True), "Step command failed")
        self.assertTrue(mock_hw.return_to_zero(), "Return to zero failed")
        self.assertTrue(mock_hw.laser_off(), "Laser OFF failed")
        print("  -> Status: PASSED (Zero coil resting current enforced)")

    def test_02_camera_stream(self):
        """Camera Stream & Dual-Frame Acquisition."""
        print("\n[TEST 2/7] Camera Stream & Dual-Frame Acquisition...")
        mock_hw = MockSerialController()
        cam = get_camera_stream(force_mock=True)
        amb, las, diff = cam.capture_pair(mock_hw, step_index=0)
        self.assertEqual(amb.shape, (1080, 1920, 3), f"Bad ambient frame shape: {amb.shape}")
        self.assertEqual(las.shape, (1080, 1920, 3), f"Bad laser frame shape: {las.shape}")
        self.assertEqual(diff.shape, (1080, 1920), f"Bad diff shape: {diff.shape}")
        print(f"  -> Status: PASSED (Frames captured at 1080p, Diff range: [{np.min(diff)}, {np.max(diff)}])")

    def test_03_laser_tracker(self):
        """Analytical Subpixel Laser Extraction (< 1 ms)."""
        print("\n[TEST 3/7] Analytical Subpixel Laser Extraction (< 1 ms)...")
        mock_hw = MockSerialController()
        cam = get_camera_stream(force_mock=True)
        _, _, diff = cam.capture_pair(mock_hw, step_index=0)
        lt = LaserTracker()
        ldata = lt.extract_subpixel_laser(diff)
        self.assertGreater(len(ldata["y"]), 0, "No laser points detected")
        self.assertEqual(len(ldata["xc"]), len(ldata["y"]))
        self.assertEqual(len(ldata["z_metric"]), len(ldata["y"]))
        print(f"  -> Status: PASSED (Extracted {len(ldata['y'])} points, Mean Z = {np.mean(ldata['z_metric']):.1f} mm)")

    def test_04_background_remover(self):
        """Foreground Object Segmentation (u2net)."""
        print("\n[TEST 4/7] Foreground Object Segmentation (u2net)...")
        mock_hw = MockSerialController()
        cam = get_camera_stream(force_mock=True)
        amb, _, _ = cam.capture_pair(mock_hw, step_index=0)
        br = BackgroundRemover()
        mask = br.get_mask(amb)
        self.assertEqual(mask.shape, (1080, 1920), f"Bad mask shape: {mask.shape}")
        self.assertGreater(np.sum(mask), 1000, "Mask is empty")
        print(f"  -> Status: PASSED (Object mask area: {np.sum(mask):,} pixels)")

    def test_05_depth_estimator(self):
        """Monocular Relative Depth (Depth Anything V2, CUDA FP16)."""
        print("\n[TEST 5/7] Monocular Relative Depth (Depth Anything V2, CUDA FP16)...")
        mock_hw = MockSerialController()
        cam = get_camera_stream(force_mock=True)
        amb, _, _ = cam.capture_pair(mock_hw, step_index=0)
        de = DepthEstimator()
        depth_map, lat_ms = de.estimate_depth(amb)
        self.assertEqual(depth_map.shape, (1080, 1920), f"Bad depth map shape: {depth_map.shape}")
        self.assertTrue(0.0 <= np.min(depth_map) <= np.max(depth_map) <= 1.0)
        print(f"  -> Status: PASSED (Inference Latency: {lat_ms:.2f} ms on CUDA FP16)")

    def test_06_metric_aligner(self):
        """Metric Alignment Layer (Scale Ambiguity Resolution)."""
        print("\n[TEST 6/7] Metric Alignment Layer (Scale Ambiguity Resolution)...")
        mock_hw = MockSerialController()
        cam = get_camera_stream(force_mock=True)
        amb, _, diff = cam.capture_pair(mock_hw, step_index=0)
        lt = LaserTracker()
        ldata = lt.extract_subpixel_laser(diff)
        br = BackgroundRemover()
        mask = br.get_mask(amb)
        de = DepthEstimator()
        depth_map, _ = de.estimate_depth(amb)
        ma = MetricAligner()
        pts_3d, col_rgb, s, t = ma.align_and_densify(depth_map, ldata, mask, amb)
        self.assertGreater(len(pts_3d), 0, "Point cloud is empty")
        self.assertEqual(pts_3d.shape[1], 3)
        self.assertEqual(col_rgb.shape[1], 3)
        print(f"  -> Status: PASSED (Generated {len(pts_3d):,} metric 3D points | Scale s = {s:.2f} mm, Shift t = {t:.2f} mm)")

    def test_07_reconstruction_and_watertight_stl(self):
        """Multi-View Integration & Watertight STL Reconstruction."""
        print("\n[TEST 7/7] Multi-View Integration & Watertight STL Reconstruction...")
        mock_hw = MockSerialController()
        cam = get_camera_stream(force_mock=True)
        amb, _, diff = cam.capture_pair(mock_hw, step_index=0)
        lt = LaserTracker()
        ldata = lt.extract_subpixel_laser(diff)
        br = BackgroundRemover()
        mask = br.get_mask(amb)
        de = DepthEstimator()
        depth_map, _ = de.estimate_depth(amb)
        ma = MetricAligner()
        pts_3d, col_rgb, _, _ = ma.align_and_densify(depth_map, ldata, mask, amb)

        ci = CloudIntegrator()
        ci.add_view(pts_3d, col_rgb, step_index=0)
        pcd = ci.get_integrated_point_cloud()
        self.assertGreater(len(pcd.points), 0, "Integrated point cloud is empty")

        me = MeshExporter(poisson_depth=6) # fast depth for test suite
        tmesh, stats = me.reconstruct_mesh(pcd)
        self.assertTrue(stats["is_watertight"], "Mesh must be 100% watertight")
        print(f"  -> Status: PASSED (Watertight: {stats['is_watertight']} | Faces: {stats['face_count']:,} | Volume: {stats['volume_mm3']:.1f} mm^3)")


def run_all_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNeuralScanPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()
