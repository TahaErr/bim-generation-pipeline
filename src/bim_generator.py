"""Multi-backend BIM generator: SD1.5 / SDXL / Flux.1-dev."""

import os
os.environ["PYTORCH_JIT"] = "0"

import torch, torch.jit

# JIT patch (simpler — xformers removed, only torch.jit issues remain)
_orig_script = torch.jit.script
def _safe_script(fn=None, *a, **kw):
    if fn is None:
        return lambda f: f
    return fn
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

# ── ControlNet model IDs ──────────────────────────────────────────
CN_SD15 = {
    "depth":    "lllyasviel/sd-controlnet-depth",
    "depth_v2": "lllyasviel/sd-controlnet-depth",
    "hed":      "lllyasviel/sd-controlnet-hed",
    "canny":    "lllyasviel/sd-controlnet-canny",
}
CN_SDXL  = {"depth": "diffusers/controlnet-depth-sdxl-1.0"}
CN_FLUX  = {"depth": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"}

DEFAULTS = {
    "sd15": dict(steps=35, guidance=12.0, cn_scale=1.2, size=512),
    "sdxl": dict(steps=30, guidance=10.0, cn_scale=0.8, size=1024),
    "flux": dict(steps=24, guidance=3.5,  cn_scale=0.5, size=1024),
}

class SD15Backend:
    def __init__(self, condition_mode="depth_v2", seed=42):
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen = torch.Generator(self.device).manual_seed(seed)
        cn_id = CN_SD15.get(condition_mode, CN_SD15["depth"])
        print(f"[SD1.5] Loading ControlNet: {cn_id}")
        cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=torch.float16)
        print("[SD1.5] Loading SD 1.5...")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=cn,
            torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        print(f"[SD1.5] Ready | device: {self.device}")

    def generate(self, condition_map, prompt, negative_prompt, steps=35, guidance=12.0, cn_scale=1.2, size=512):
        cond = condition_map.resize((size, size))
        with torch.inference_mode():
            out = self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=cond,
                           num_inference_steps=steps, guidance_scale=guidance,
                           controlnet_conditioning_scale=cn_scale,
                           generator=self.gen, width=size, height=size)
        return out.images[0]

class SDXLBackend:
    def __init__(self, condition_mode="depth_v2", seed=42):
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, EulerAncestralDiscreteScheduler
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen = torch.Generator(self.device).manual_seed(seed)
        cn_id = CN_SDXL["depth"]
        print(f"[SDXL] Loading ControlNet: {cn_id}")
        cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=torch.float16)
        print("[SDXL] Loading SDXL base...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        print(f"[SDXL] Ready | device: {self.device}")

    def generate(self, condition_map, prompt, negative_prompt, steps=30, guidance=10.0, cn_scale=0.8, size=1024):
        cond = condition_map.resize((size, size))
        with torch.inference_mode():
            out = self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=cond,
                           num_inference_steps=steps, guidance_scale=guidance,
                           controlnet_conditioning_scale=cn_scale,
                           generator=self.gen, width=size, height=size)
        return out.images[0]

class FluxBackend:
    def __init__(self, seed=42):
        from diffusers import FluxControlNetModel, FluxControlNetPipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen = torch.Generator(self.device).manual_seed(seed)
        cn_id = CN_FLUX["depth"]
        print(f"[Flux] Loading ControlNet: {cn_id}")
        cn = FluxControlNetModel.from_pretrained(cn_id, torch_dtype=torch.bfloat16)
        print("[Flux] Loading Flux.1-dev...")
        self.pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", controlnet=cn, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()
        print(f"[Flux] Ready | device: {self.device}")

    def generate(self, condition_map, prompt, negative_prompt="", steps=24, guidance=3.5, cn_scale=0.5, size=1024):
        cond = condition_map.resize((size, size))
        with torch.inference_mode():
            out = self.pipe(prompt=prompt, control_image=cond,
                           num_inference_steps=steps, guidance_scale=guidance,
                           controlnet_conditioning_scale=cn_scale,
                           generator=self.gen, width=size, height=size)
        return out.images[0]

_BACKENDS = {"sd15": SD15Backend, "sdxl": SDXLBackend, "flux": FluxBackend}

class BIMGenerator:
    def __init__(self, backend="sdxl", condition_mode="depth_v2", seed=42):
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend. Choose: {list(_BACKENDS)}")
        self.backend_name = backend
        self.defaults = DEFAULTS[backend]
        print(f"\n{'='*60}")
        print(f"  BIMGenerator | backend={backend}")
        print(f"{'='*60}\n")
        if backend == "flux":
            self._engine = FluxBackend(seed=seed)
        else:
            self._engine = _BACKENDS[backend](condition_mode=condition_mode, seed=seed)

    def generate(self, condition_map, prompt, negative_prompt="", **kwargs):
        params = {**self.defaults, **kwargs}
        return self._engine.generate(
            condition_map=condition_map, prompt=prompt,
            negative_prompt=negative_prompt, **params)
