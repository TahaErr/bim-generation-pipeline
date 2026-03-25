# BIM Generation Pipeline v4

Real construction photos → ControlNet → BIM-style renders

## v4 Upgrades

- **Depth Anything V2** replaces MiDaS (AbsRel 0.074 vs 0.127)
- **Multi-backend**: SD 1.5 / SDXL / Flux.1-dev selectable
- **1024×1024** resolution with SDXL and Flux
- **Backend-aware prompts** (Flux uses T5-XXL natural language)

## Results (50 renders, sdxl + depth_v2)

| Metric | Value |
|---|---|
| Backend | sdxl |
| Condition mode | depth_v2 |
| Resolution | 1024×1024 |
| Guidance | 10.0 |
| CN scale | 0.8 |
| Steps | 30 |

## Next Steps

- [ ] YOLOv8 human removal from depth maps
- [ ] Full 7K dataset generation (SDXL + Depth Anything V2)
- [ ] A/B comparison: SD1.5 vs SDXL vs Flux renders
- [ ] VLM fine-tuning on paired data
