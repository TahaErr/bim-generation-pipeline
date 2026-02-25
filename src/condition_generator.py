import cv2
import numpy as np
from PIL import Image
from controlnet_aux import MidasDetector, HEDdetector

class ConditionGenerator:
    def __init__(self, mode: str = "depth"):
        self.mode = mode
        print(f"Condition mode: {mode}")
        if mode in ("depth", "combined"):
            self.depth = MidasDetector.from_pretrained("lllyasviel/Annotators")
            print("  MiDaS loaded")
        if mode in ("hed", "combined"):
            self.hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
            print("  HED loaded")
        print("ConditionGenerator ready")

    def generate(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        if self.mode == "depth":    return self.depth(img).convert("RGB")
        if self.mode == "hed":      return self.hed(img).convert("RGB")
        if self.mode == "canny":    return self._canny(img)
        if self.mode == "combined": return self._combined(img)
        raise ValueError(f"Unknown mode: {self.mode}")

    def _canny(self, img):
        arr   = np.array(img)
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    def _combined(self, img):
        # 60% depth + 40% HED
        d = np.array(self.depth(img).convert("L"))
        h = np.array(self.hed(img).convert("L"))
        c = (d * 0.6 + h * 0.4).astype(np.uint8)
        return Image.fromarray(c).convert("RGB")
