"""
Object Segmentation & Background Isolation Engine
Generates high-contrast binary masks Mask_object for downstream 3D fusion.
"""

import logging
import cv2
import numpy as np
import rembg
import config

logger = logging.getLogger("NeuralScan.AI.BackgroundRemover")


class BackgroundRemover:
    """
    Isolates foreground scanned object on the turntable using u2net segmentation.
    Reuses cached ONNX session for high throughput.
    """

    def __init__(self, model_name: str = config.REMBG_MODEL):
        self.model_name = model_name
        logger.info(f"Initializing BackgroundRemover session with model '{self.model_name}'...")
        try:
            self.session = rembg.new_session(self.model_name)
            logger.info("BackgroundRemover session initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize rembg session: {e}. Falling back to default.")
            self.session = rembg.new_session()

    def get_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Extracts binary foreground mask (0 for background, 1 for object).
        Applies morphological filtering to eliminate perimeter edge artifacts.
        """
        # Convert BGR to RGB for rembg
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Execute rembg segmentation
        rgba_result = rembg.remove(rgb_image, session=self.session)
        alpha = rgba_result[:, :, 3]

        # Initial thresholding
        _, binary = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)

        # Morphological refinement: close holes and smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # Filter out minor disconnected noise clusters (keep components > 500 px)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        clean_mask = np.zeros_like(opened)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 500:
                clean_mask[labels == i] = 1

        return clean_mask.astype(np.uint8)
