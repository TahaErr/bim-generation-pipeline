import os
os.environ["PYTORCH_JIT"] = "0"

import torch, torch.jit

# JIT compatibility patch
_orig_script = torch.jit.script
def _safe_script(fn, *args, **kwargs):
    try:    return _orig_script(fn, *args, **kwargs)
    except: return fn
torch.jit.script = _safe_script
try:
    from torch.jit._script import ScriptMeta
    _orig_init = ScriptMeta.__init__
    def _patched_init(cls, name, bases, attrs):
        try:    return _orig_init(cls, name, bases, attrs)
        except TypeError: pass
    ScriptMeta.__init__ = _patched_init
except Exception: pass

from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

CONTROLNET_IDS = {
    "depth":    "lllyasviel/sd-controlnet-depth",
    "hed":      "lllyasviel/sd-controlnet-hed",
    "canny":    "lllyasviel/sd-controlnet-canny",
    "combined": "lllyasviel/sd-controlnet-depth",
}

class BIMGenerator:
    def __init__(self, condition_mode: str = "depth", seed: int = 42):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen    = torch.Generator(self.device).manual_seed(seed)
        print(f"Loading ControlNet: {CONTROLNET_IDS[condition_mode]}")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_IDS[condition_mode], torch_dtype=torch.float16)
        print("Loading SD 1.5...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        print(f"BIMGenerator ready | device: {self.device}")

    def generate(self, condition_map, prompt, negative_prompt,
                 steps=35, guidance=12.0, cn_scale=1.2, size=512):
        cond = condition_map.resize((size, size))
        with torch.inference_mode():
            out = self.pipe(
                prompt=prompt, negative_prompt=negative_prompt, image=cond,
                num_inference_steps=steps, guidance_scale=guidance,
                controlnet_conditioning_scale=cn_scale,
                generator=self.gen, width=size, height=size,
            )
        return out.images[0]
