"""4-layer quality filter for BIM generation pairs."""

import numpy as np
import cv2
from PIL import Image

STAGES = [
    "excavation and earthwork at construction site",
    "foundation and concrete pouring at construction site",
    "structural framing with columns and beams at construction site",
    "wall and envelope construction at construction site",
    "interior finishing and MEP installation at construction site",
    "completed building exterior at construction site",
]
STAGE_SHORT = ["excavation", "foundation", "framing", "envelope", "finishing", "completed"]


def edge_density(img_pil, low=50, high=150):
    arr = np.array(img_pil.convert("L"))
    edges = cv2.Canny(arr, low, high)
    return edges.mean() / 255.0


def bim_style_score(img_pil):
    arr = np.array(img_pil.convert("HSV"))
    sat = arr[:, :, 1].mean() / 255.0
    hue_ratio = len(np.unique(arr[:, :, 0])) / 256.0
    return float(1.0 - (sat * 0.6 + hue_ratio * 0.4))
