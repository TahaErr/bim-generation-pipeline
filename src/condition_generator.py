"""Condition map generator — Depth Anything V2 (primary) + legacy fallbacks."""

import cv2
import numpy as np
from PIL import Image

def _load_depth_anything_v2(model_size="Large"):
    from transformers import pipeline
    model_id = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
    print(f"  Loading Depth Anything V2 ({model_size}): {model_id}")
    return pipeline(task="depth-estimation", model=model_id, device=0)

def _depth_v2_inference(pipe, image):
    result = pipe(image)
    depth_map = result["depth"]
    arr = np.array(depth_map, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

class ConditionGenerator:
    SUPPORTED_MODES = ("depth_v2", "depth_midas", "hed", "canny", "combined")

    def __init__(self, mode="depth_v2", depth_v2_size="Large"):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unknown mode. Choose from: {self.SUPPORTED_MODES}")
        self.mode = mode
        print(f"Condition mode: {mode}")

        if mode in ("depth_v2", "combined"):
            self.depth_v2_pipe = _load_depth_anything_v2(depth_v2_size)
            print("  Depth Anything V2 loaded")
        if mode == "depth_midas":
            from controlnet_aux import MidasDetector
            self.depth_midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
            print("  MiDaS loaded")
        if mode in ("hed", "combined"):
            from controlnet_aux import HEDdetector
            self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
            print("  HED loaded")
        print("ConditionGenerator ready")

    def generate(self, image):
        img = image.convert("RGB")
        if self.mode == "depth_v2":    return _depth_v2_inference(self.depth_v2_pipe, img)
        if self.mode == "depth_midas": return self.depth_midas(img).convert("RGB")
        if self.mode == "hed":         return self.hed(img).convert("RGB")
        if self.mode == "canny":       return self._canny(img)
        if self.mode == "combined":    return self._combined(img)

    def _canny(self, img):
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    def _combined(self, img):
        d = np.array(_depth_v2_inference(self.depth_v2_pipe, img).convert("L"), dtype=np.float32)
        h = np.array(self.hed(img).convert("L"), dtype=np.float32)
        c = (d * 0.6 + h * 0.4).astype(np.uint8)
        return Image.fromarray(c).convert("RGB")
