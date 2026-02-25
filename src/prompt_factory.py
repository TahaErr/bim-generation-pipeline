VIEW_PROMPTS = {
    "elevation view":   "front elevation view",
    "plan view":        "top-down plan view",
    "perspective view": "perspective view",
    "aerial view":      "aerial perspective view",
    "close-up view":    "close-up detail view",
    "interior view":    "interior section view",
}
DISTANCE_HINTS = {
    "close distance": "detailed close-up",
    "mid distance":   "medium distance",
    "far distance":   "wide-angle overview",
}

def build_prompt(sample: dict) -> tuple:
    """Returns (positive, negative) prompt for a dataset sample."""
    view     = VIEW_PROMPTS.get(sample.get("view", ""), "perspective view")
    distance = DISTANCE_HINTS.get(sample.get("camera_distance", ""), "medium distance")
    positive = (
        f"BIM render, {view}, {distance}, "
        f"3D building model, flat colors, structural elements, "
        f"Revit style, no texture, technical CAD drawing, "
        f"concrete steel structure, clean background"
    )
    negative = (
        "(workers:2.0), (people:2.0), (humans:2.0), "
        "(hard hat:1.5), (watermark:2.0), (text:2.0), "
        "photorealistic, photo, dirt, vegetation, "
        "shadows, blur, low quality"
    )
    return positive, negative
