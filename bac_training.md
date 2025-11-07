##The Training Loop 

``` java
for each epoch:
    for each batch:
        1. Load image + labels
        2. Forward pass (model → predictions)
        3. Compute loss (CIoU + BCE)
        4. Backpropagate gradients
        5. Update weights (optimizer)
```

## Project Folder Structure
```
bacterial_detection/
│
├── images/
│   ├── train/
│   ├── val/
│
├── labels/
│   ├── train/
│   ├── val/
│
├── train.py           ← training loop script
├── model.py           ← backbone, neck, head, loss
└── utils.py           ← helper functions
