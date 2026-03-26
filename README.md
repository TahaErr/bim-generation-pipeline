# BIM Generation Pipeline v5

Real construction photos → ControlNet + IP-Adapter → BIM-style renders

## Features

- **Depth Anything V2** (AbsRel 0.074 vs MiDaS 0.127)
- **SDXL + IP-Adapter** — Revit reference images as style guide
- **1024×1024** resolution
- **19 Revit reference images** (10 filtered solid/shaded)

## Latest Run (343 renders)

| Metric | Value |
|---|---|
| Backend | SDXL + IP-Adapter |
| Condition | Depth Anything V2 (Large) |
| IP-Adapter scale | 0.5 |
| CN scale | 0.8 |
| BIM Style Score | 0.71 (mean) |
| Resolution | 1024×1024 |
| Steps | 30 |

## Pipeline

```
Real Photo → Depth Anything V2 → ControlNet (depth)
                                      +
            Revit Ref Images → IP-Adapter (style)
                                      ↓
                                 BIM Render
```

## Next Steps

- [ ] Dataset quality filtering for VLM fine-tuning
- [ ] Full 7K+ dataset generation
- [ ] VLM fine-tuning on paired (real, BIM) data
- [ ] Construction progress estimation
