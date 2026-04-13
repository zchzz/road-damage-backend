import base64
import logging
import os
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoadDamageDetector:
    def __init__(self, model_path: str | None = None):
        """
        初始化道路病害检测器

        Args:
            model_path: YOLO 权重路径；为空时自动尝试项目内常见路径
        """
        if model_path is None:
            trained_model_paths = [
                "algorithm/weights/best.pt",
                "runs/RDD_Training/test_training/weights/best.pt",
                "runs/RDD_Training/test_training2/weights/best.pt",
                "runs/RDD_Training/YOLOv8_RDD_Model/weights/best.pt",
                "models/best.pt",
            ]

            for path in trained_model_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                "未找到道路病害检测模型权重，请确认 best.pt 路径是否正确。"
            )

        self.model = YOLO(model_path)
        logger.info("Loaded road damage model from %s", model_path)
        logger.info("Model classes: %s", self.model.names)

        self.damage_classes = [
            "Longitudinal Crack",
            "Transverse Crack",
            "Alligator Crack",
            "Potholes",
        ]