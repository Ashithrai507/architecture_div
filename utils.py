import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ======================================
# 🔹 Bounding Box Conversion Utilities
# ======================================

def xywh_to_xyxy(boxes):
    """
    Convert [x_center, y_center, w, h] → [x1, y1, x2, y2]
    """
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def xyxy_to_xywh(boxes):
    """
    Convert [x1, y1, x2, y2] → [x_center, y_center, w, h]
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


# ======================================
# 🔹 IoU and CIoU Computation
# ======================================

def bbox_iou(box1, box2, eps=1e-7):
    """
    Compute IoU between two sets of boxes.
    box1: [N, 4] (x1, y1, x2, y2)
    box2: [M, 4]
    """
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter + eps
    return inter / union


def bbox_ciou(box1, box2, eps=1e-7):
    """
    Compute CIoU between two sets of boxes.
    Input format: [x_center, y_center, w, h]
    """
    # Convert to corner coordinates
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + eps
    iou = inter / union

    # Distance between box centers
    center_dist = (box1[:, 0] - box2[:, 0]) ** 2 + (box1[:, 1] - box2[:, 1]) ** 2

    # Enclosing box diagonal
    c_x1, c_y1 = torch.min(b1_x1, b2_x1), torch.min(b1_y1, b2_y1)
    c_x2, c_y2 = torch.max(b1_x2, b2_x2), torch.max(b1_y2, b2_y2)
    c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Aspect ratio term
    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(box2[:, 2] / (box2[:, 3] + eps)) -
                                       torch.atan(box1[:, 2] / (box1[:, 3] + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / c_diag + alpha * v)
    return ciou


# ======================================
# 🔹 Non-Maximum Suppression (NMS)
# ======================================

def nms(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression (NMS) to remove overlapping boxes.
    boxes: [N, 4] (x1, y1, x2, y2)
    scores: [N]
    """
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())

        if idxs.numel() == 1:
            break

        ious = bbox_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][ious < iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


# ======================================
# 🔹 Visualization Helpers
# ======================================

def draw_boxes(image, boxes, color=(0, 255, 0), labels=None, thickness=2):
    """
    Draw bounding boxes on the image.
    image: OpenCV image (BGR)
    boxes: [[x1, y1, x2, y2], ...]
    labels: optional list of text labels
    """
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels:
            cv2.putText(img, str(labels[i]), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def show_image(img, title="Image"):
    """
    Display image using matplotlib.
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# ======================================
# 🔹 Metric Logger (AverageMeter)
# ======================================

class AverageMeter:
    """
    Keeps track of average values (e.g., loss, accuracy)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
