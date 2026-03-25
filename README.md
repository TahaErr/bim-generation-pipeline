# BIM Generation Pipeline v5

Real construction photos → ControlNet → BIM-style renders

## Features

- **Depth Anything V2** (AbsRel 0.074 vs MiDaS 0.127)
- **Multi-backend**: SD 1.5 / SDXL / Flux.1-dev
- **Up to 1024×1024** resolution
- **Backend-aware prompts**

## Latest Run (50 renders, sdxl + depth_v2)

| Metric | Value |
|---|---|
| Backend | sdxl |
| Condition | depth_v2 |
| Resolution | 1024×1024 |
| Guidance | 10.0 |
| CN scale | 0.8 |
| Steps | 30 |

## Next Steps

- [ ] YOLOv8 human removal from depth maps
- [ ] Full 7K dataset generation
- [ ] A/B comparison across backends
- [ ] VLM fine-tuning on paired data
