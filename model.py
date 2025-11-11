import torch
import torch.nn as nn
import torch.nn.functional as F



#Basic Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



#Residual Block

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer1 = ConvBlock(channels, channels)
        self.layer2 = ConvBlock(channels, channels)

    def forward(self, x):
        return x + self.layer2(self.layer1(x))



#Backbone (Feature Extractor)#

class MyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ConvBlock(3, 32, stride=2)
        self.layer2 = nn.Sequential(ConvBlock(32, 64, stride=2), ResidualBlock(64))
        self.layer3 = nn.Sequential(ConvBlock(64, 128, stride=2), ResidualBlock(128), ResidualBlock(128))
        self.layer4 = nn.Sequential(ConvBlock(128, 256, stride=2), ResidualBlock(256), ResidualBlock(256))
        self.layer5 = nn.Sequential(ConvBlock(256, 512, stride=2), ResidualBlock(512), ResidualBlock(512))

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x3, x4, x5 



#Neck (Feature Pyramid Network)

class FPNNeck(nn.Module):
    def __init__(self, channels=[128, 256, 512]):
        super().__init__()
        self.reduce_c3 = nn.Conv2d(channels[0], 128, 1)
        self.reduce_c4 = nn.Conv2d(channels[1], 256, 1)
        self.reduce_c5 = nn.Conv2d(channels[2], 512, 1)

        self.conv_c4 = ConvBlock(512 + 256, 256)
        self.conv_c3 = ConvBlock(256 + 128, 128)

    def forward(self, c3, c4, c5):
        c3 = self.reduce_c3(c3)
        c4 = self.reduce_c4(c4)
        c5 = self.reduce_c5(c5)

        up_c5 = F.interpolate(c5, scale_factor=2, mode="nearest")
        fused_c4 = torch.cat([up_c5, c4], dim=1)
        p4 = self.conv_c4(fused_c4)

        up_p4 = F.interpolate(p4, scale_factor=2, mode="nearest")
        fused_c3 = torch.cat([up_p4, c3], dim=1)
        p3 = self.conv_c3(fused_c3)

        return p3, p4, c5



#Detection Head

class DetectionHead(nn.Module):
    def __init__(self, num_classes=1, anchors_per_scale=3):
        super().__init__()
        self.num_outputs = 5 + num_classes  # (x, y, w, h, obj, class)
        self.head_small = nn.Conv2d(128, anchors_per_scale * self.num_outputs, 1)
        self.head_medium = nn.Conv2d(256, anchors_per_scale * self.num_outputs, 1)
        self.head_large = nn.Conv2d(512, anchors_per_scale * self.num_outputs, 1)

    def forward(self, p3, p4, p5):
        out_small = self.head_small(p3)
        out_medium = self.head_medium(p4)
        out_large = self.head_large(p5)
        return [out_small, out_medium, out_large]


#Full Bacteria Detector Model#

class BacteriaDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = MyBackbone()
        self.neck = FPNNeck()
        self.head = DetectionHead(num_classes=num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        preds = self.head(p3, p4, p5)
        return preds



#Loss Function (CIoU + BCE)

def detection_loss(preds, targets, anchors, device, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0):
    """
    Custom loss function combining CIoU + BCE for object detection.
    """
    bce = nn.BCELoss(reduction="sum")
    total_box_loss, total_obj_loss, total_cls_loss = 0, 0, 0

    for scale_i, pred in enumerate(preds):
        B, C, H, W = pred.shape
        pred = pred.view(B, 3, (5 + 1), H, W).permute(0, 1, 3, 4, 2)

        # Placeholder targets for now
        obj_target = torch.zeros_like(pred[..., 4], device=device)
        cls_target = torch.zeros_like(pred[..., 5], device=device)

        box_loss = torch.tensor(0.0, device=device)
        obj_loss = bce(torch.sigmoid(pred[..., 4]), obj_target)
        cls_loss = bce(torch.sigmoid(pred[..., 5]), cls_target)

        total_box_loss += box_loss
        total_obj_loss += obj_loss
        total_cls_loss += cls_loss

    total_loss = lambda_box * total_box_loss + lambda_obj * total_obj_loss + lambda_cls * total_cls_loss
    return total_loss
