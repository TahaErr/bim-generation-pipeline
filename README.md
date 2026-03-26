# BIM Generation Pipeline v5

Real construction photos → ControlNet + IP-Adapter → BIM-style renders

## Features

- **Depth Anything V2** (AbsRel 0.074 vs MiDaS 0.127)
- **SDXL + IP-Adapter** — Revit reference images as style guide
- **4-layer quality filter** — CLIP-based source/pair/stage filtering
- **1024×1024** resolution

## Latest Run

| Metric | Value |
|---|---|
| Total renders | 343 |
| After filtering | 257 (75%) |
| BIM Style Score | 0.71 (mean) |
| Backend | SDXL + IP-Adapter |
| IP-Adapter scale | 0.5 |
| CN scale | 0.8 |

## Quality Filter (4 layers)

1. **Source quality** — CLIP construction relevance
2. **BIM render quality** — style score + edge density
3. **Pair consistency** — CLIP embedding similarity + SSIM
4. **Stage classification** — excavation/foundation/framing/envelope/finishing

## Stage Distribution (filtered)

| Stage | Count | % |
|---|---|---|
| Excavation | 137 | 53% |
| Foundation | 98 | 38% |
| Envelope | 15 | 6% |
| Finishing | 6 | 2% |
| Framing | 1 | 0% |

## Next Steps

- [ ] Pre-classify full 7K dataset by stage (CLIP)
- [ ] Balanced sampling across stages before BIM generation
- [ ] Full dataset generation (~5K balanced pairs)
- [ ] VLM fine-tuning with progress estimation labels
