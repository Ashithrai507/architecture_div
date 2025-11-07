# HEAD

## What the Detection Head Does

The Head takes your fused feature maps (p3, p4, p5) from the Neck
and predicts objects (colonies) in each cell of those maps.

| Output                  | Meaning                                        |
| ----------------------- | ---------------------------------------------- |
| **x, y**                | center of colony (relative to cell)            |
| **w, h**                | width and height of the bounding box           |
| **objectness**          | confidence score (how likely there’s a colony) |
| **class probabilities** | (if multiple bacteria types — optional)        |

### Next step after the head is loss function

| Step                        | Description                                   |
| --------------------------- | --------------------------------------------- |
| 1️⃣ **Loss Function**       | Implement CIoU + BCE loss for training        |
| 2️⃣ **Post-Processing**     | Decode predictions (x, y, w, h) and apply NMS |
| 3️⃣ **Dataset Preparation** | Annotate colonies and load using DataLoader   |
| 4️⃣ **Training Loop**       | Train end-to-end on your labeled data         |
| 5️⃣ **Evaluation**          | Measure mAP, precision, recall                |

# Concept Recap

``` scss

Input Image (640×640)
   ↓
[Backbone]
   ↓
  f3(80x80) ─────────┐
  f4(40x40) ─────┐   │
  f5(20x20) ──┐  │   │
               │  ↓   ↓
            [FPN Neck] → p3,p4,p5
               ↓
           [Detection Head]
               ↓
     Predictions (x,y,w,h,conf)

```

## What the Loss Function Does

**The loss function tells the network how wrong its predictions are and how to adjust during training.**

**Each predicted bounding box is compared to a ground truth box (the actual colony location from your dataset).**

We calculate three parts of loss per prediction:

| Component            | Description                                                    | Function                                  |
| -------------------- | -------------------------------------------------------------- | ----------------------------------------- |
| **CIoU Loss**        | Measures how well the predicted box overlaps the true box      | Box Regression                            |
| **BCE (Objectness)** | Measures how confident the model is that a colony exists there | Object Confidence                         |
| **BCE (Class)**      | Measures if the model classified the colony correctly          | Classification (for multi-class datasets) |

