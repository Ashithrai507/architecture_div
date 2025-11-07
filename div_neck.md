# NECK

## The Neck connects the Backbone (feature extractor) to the Detection Head (predictor).

## The Neck helps the model understand both fine details (small colonies) and global context (large colonies).

**Takes multiple scales of features from the backbone (small, medium, large)**

**Merges them in a way that each output layer gets context from all scales**

## STEP 2: Why You Need a Neck

Without the Neck:

Small-scale layers can detect only tiny colonies

Deep layers can detect only large colonies

No communication happens between them

The Neck (like FPN/PAN) fuses them together, improving accuracy for all object sizes.

## STEP 3: Choose a Neck Type

| Type                              | Description                       | Used In         |
| --------------------------------- | --------------------------------- | --------------- |
| **FPN (Feature Pyramid Network)** | Top-down merging (deep → shallow) | YOLOv3          |
| **PANet**                         | Both top-down and bottom-up       | YOLOv4 / YOLOv5 |
| **BiFPN**                         | Weighted fusion                   | EfficientDet    |

We’ll build a simplified FPN-style Neck, perfect for your custom backbone.

```sql

Backbone Outputs:
   P3 = 80×80 (small)
   P4 = 40×40 (medium)
   P5 = 20×20 (large)

FPN Operations:
   P5 → Upsample → Merge with P4 → Fused(40×40)
   Fused(40×40) → Upsample → Merge with P3 → Fused(80×80)

```
## Test the Neck

After testing the neck we get three outputs

```python

p3: torch.Size([1, 128, 80, 80])
p4: torch.Size([1, 256, 40, 40])
p5: torch.Size([1, 512, 20, 20])

```

### What Each Neck Output Represents

| Output | Feature Map Size | Channels | Role            |
| ------ | ---------------- | -------- | --------------- |
| `p3`   | 80×80            | 128      | Small colonies  |
| `p4`   | 40×40            | 256      | Medium colonies |
| `p5`   | 20×20            | 512      | Large colonies  |
