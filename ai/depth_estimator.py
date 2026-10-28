"""
Depth Anything V2 Monocular Depth Inference Engine
Strictly optimized for FP16 CUDA Tensor Cores on RTX 4060 GPU (latency: 15-25 ms).
Complies with Section 7 strict prohibition on transformers.pipeline.
"""

import time
import logging
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import config

logger = logging.getLogger("NeuralScan.AI.DepthEstimator")


class DepthEstimator:
    """
    Direct model inference for Depth Anything V2 using native PyTorch FP16.
    Zero pipeline overhead, direct tensor memory allocation.
    """

    def __init__(self, model_id: str = config.DEPTH_MODEL_ID, device: str = config.DEVICE, use_fp16: bool = config.USE_FP16):
        self.model_id = model_id
        self.device = device
        self.use_fp16 = use_fp16 and (device == "cuda")

        logger.info(f"Loading Depth Estimator '{self.model_id}' on {self.device} (FP16={self.use_fp16})...")
        
        # Load with local cache priority for instant startup
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=True, use_fast=False)
            raw_model = AutoModelForDepthEstimation.from_pretrained(self.model_id, local_files_only=True)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=False)
            raw_model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        
        if self.use_fp16:
            self.model = raw_model.half().to(self.device)
        else:
            self.model = raw_model.to(self.device)

        self.model.eval()
        
        # Warm-up inference
        self._warmup()
        logger.info("Depth Estimator successfully loaded and warmed up.")

    def _warmup(self):
        """Warm up CUDA kernels for consistent low-latency execution."""
        dummy_img = Image.new("RGB", (518, 518), color="white")
        with torch.no_grad():
            inputs = self.processor(images=dummy_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            _ = self.model(**inputs)
            if self.device == "cuda":
                torch.cuda.synchronize()

    def estimate_depth(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Executes FP16 forward pass on input BGR frame and returns normalized depth map D_AI in [0, 1].
        
        Returns:
            depth_map: 2D np.ndarray (float32) of shape (H, W) in range [0.0, 1.0]
            latency_ms: float, execution time in milliseconds
        """
        h_orig, w_orig = bgr_image.shape[:2]
        
        # Convert BGR to RGB PIL image
        rgb_image = bgr_image[:, :, ::-1] # fast slice inversion
        pil_image = Image.fromarray(rgb_image)

        start_time = time.perf_counter()

        with torch.no_grad():
            # Process image to model input tensors
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

            # Forward pass
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth # shape (1, H_proc, W_proc)

            # Interpolate to native camera resolution (1920x1080) using bicubic sampling
            depth_resized = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h_orig, w_orig),
                mode="bicubic",
                align_corners=False
            ).squeeze()

            if self.device == "cuda":
                torch.cuda.synchronize()

            raw_depth = depth_resized.float().cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Normalize relative depth to range [0.0, 1.0]
        d_min = np.min(raw_depth)
        d_max = np.max(raw_depth)
        d_range = d_max - d_min

        if d_range > 1e-6:
            depth_normalized = (raw_depth - d_min) / d_range
        else:
            depth_normalized = np.zeros_like(raw_depth)

        return depth_normalized.astype(np.float32), latency_ms
