"""Backend-aware prompt factory for BIM generation."""

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

_NEG_SD15 = (
    "(workers:2.0), (people:2.0), (humans:2.0), "
    "(hard hat:1.5), (watermark:2.0), (text:2.0), "
    "photorealistic, photo, dirt, vegetation, shadows, blur, low quality"
)

_NEG_SDXL = (
    "(workers:2.0), (people:2.0), (humans:2.0), (hard hat:1.5), "
    "(watermark:2.0), (text:2.0), (logo:1.5), "
    "photorealistic, photograph, real photo, "
    "dirt, vegetation, grass, trees, "
    "deep shadows, motion blur, low quality, jpeg artifacts, noise, grain"
)


def _base_prompt(sample):
    view = VIEW_PROMPTS.get(sample.get("view", ""), "perspective view")
    dist = DISTANCE_HINTS.get(sample.get("camera_distance", ""), "medium distance")
    return view, dist


def _prompt_sd15(sample):
    view, dist = _base_prompt(sample)
    pos = (
        f"BIM render, {view}, {dist}, "
        f"3D building model, flat colors, structural elements, "
        f"Revit style, no texture, technical CAD drawing, "
        f"concrete steel structure, clean background"
    )
    return pos, _NEG_SD15


def _prompt_sdxl(sample):
    view, dist = _base_prompt(sample)
    caption = sample.get("image_caption", "")
    ctx = f", scene context: {caption[:60]}" if caption else ""
    pos = (
        f"Professional BIM render, {view}, {dist}, "
        f"high-quality 3D building information model, "
        f"flat matte colors, visible structural elements, "
        f"Autodesk Revit rendering style, untextured surfaces, "
        f"technical architectural CAD visualization, "
        f"concrete and steel structural frame, "
        f"clean neutral background, sharp edges, "
        f"architectural blueprint aesthetic, "
        f"no people, no vegetation, no shadows"
        f"{ctx}"
    )
    return pos, _NEG_SDXL


def _prompt_flux(sample):
    view, dist = _base_prompt(sample)
    caption = sample.get("image_caption", "")
    ctx = f" The original scene shows: {caption[:80]}." if caption else ""
    pos = (
        f"A professional BIM render showing a "
        f"{view} of a building at {dist}. "
        f"Autodesk Revit 3D model export with flat matte colors, "
        f"visible concrete and steel structural elements, "
        f"clean untextured surfaces, sharp geometric edges. "
        f"Clean neutral background. "
        f"No people, no vegetation, no realistic textures or shadows. "
        f"Technical architectural visualization."
        f"{ctx}"
    )
    return pos, ""


_BUILDERS = {"sd15": _prompt_sd15, "sdxl": _prompt_sdxl, "flux": _prompt_flux}


def build_prompt(sample, backend="sdxl"):
    """Return (positive, negative) prompts for a dataset sample."""
    return _BUILDERS.get(backend, _prompt_sdxl)(sample)
