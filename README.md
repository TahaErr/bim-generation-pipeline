# BIM Generation Pipeline

Real construction photos -> ControlNet -> BIM-style renders

## Results (50 renders generated)

| Metric | Value |
|---|---|
| BIM Style Score | 0.61 (target > 0.55) |
| Condition mode  | depth (MiDaS) |
| Guidance scale  | 12.0 |
| ControlNet scale| 1.2 |
| Steps           | 35 |

## Next Steps

- [ ] YOLOv8 human removal from depth maps
- [ ] Full 7K dataset generation
- [ ] VLM fine-tuning on paired data
