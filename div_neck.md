# NECK

## The Neck connects the Backbone (feature extractor) to the Detection Head (predictor).

## The Neck helps the model understand both fine details (small colonies) and global context (large colonies).

### What the Neck Does

The neck combines and enhances these maps using two major ideas:

#### Feature Pyramid Network (FPN)
→ Combines low-level (detailed) and high-level (contextual) features.

#### Path Aggregation Network (PAN)
→ Strengthens information flow bottom-up (from small to large features).

### The layer flow in simplified way
``` scss
             Backbone Output
     ┌────────────┬────────────┬────────────┐
     │     P3     │     P4     │     P5     │
     │(80×80×128) │(40×40×256) │(20×20×512) │
     └─────┬──────┴─────┬──────┴─────┬──────┘
           │              │              │
           ▼              ▼              ▼
    1x1 Convs → Reduce Channels → Upsample → Merge
           │              │              │
           ▼              ▼              ▼
     Fused Feature Maps (multi-scale)
           │
           ▼
   Output → [small, medium, large feature maps]

```

**Takes multiple scales of features from the backbone (small, medium, large)**

**Merges them in a way that each output layer gets context from all scales**

##  Why You Need a Neck

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

### What Happens to an Image Here

| Stage                   | Feature Map Example | What’s Happening                                   |
| ----------------------- | ------------------- | -------------------------------------------------- |
| From Backbone           | P3, P4, P5          | The image is represented as abstract feature maps  |
| 1×1 Convs               | Channel reduction   | Keeps important data, reduces redundancy           |
| Upsample P5             | 20×20 → 40×40       | Aligns with medium scale features                  |
| Concatenate P5+P4       | 512 channels        | Combines large + medium colony info                |
| Upsample + concat again | 80×80               | Adds small colony details                          |
| Conv layers             | Refinement          | Sharpens colony features, removes background noise |
| Output                  | F3, F4, F5          | Clean, multi-scale features ready for detection    |

| Layer Type       | Input              | Output                | Purpose                |
| ---------------- | ------------------ | --------------------- | ---------------------- |
| 1×1 Conv         | P3, P4, P5         | Reduce channels       | Align depth            |
| Upsample         | P5 → P4, P4 → P3   | Match resolutions     | Combine scales         |
| Concat           | Stack feature maps | Fuse detail + context |                        |
| ConvBlocks       | 3×3 Convs          | Refine and filter     | Remove noise           |
| Downsample (PAN) | F3 → F4 → F5       | Bottom-up path        | Pass small info upward |
| Output           | F3, F4, F5         | 80×80, 40×40, 20×20   | To detection head      |

