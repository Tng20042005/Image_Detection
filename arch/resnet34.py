import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .common import ConvBlock, DecoupledHead

class YOLO_ResNet34(nn.Module):
    def __init__(self, num_classes=5, num_anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_scale

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone_base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.layer2 = resnet.layer2 
        self.layer3 = resnet.layer3 
        self.layer4 = resnet.layer4 

        # --- Top-down FPN (Giữ nguyên của bạn) ---
        self.lateral_p5 = nn.Conv2d(512, 128, kernel_size=1)
        self.lateral_p4 = nn.Conv2d(256, 128, kernel_size=1)
        self.lateral_p3 = nn.Conv2d(128, 128, kernel_size=1)
        
        self.smooth_p4 = ConvBlock(128, 128, kernel_size=3, padding=1)
        self.smooth_p3 = ConvBlock(128, 128, kernel_size=3, padding=1)

        # --- BỔ SUNG: Luồng Bottom-up PANet ---
        self.downsample_n3 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1)
        self.pan_n4_conv = ConvBlock(128, 128, kernel_size=3, padding=1)
        
        self.downsample_n4 = ConvBlock(128, 128, kernel_size=3, stride=2, padding=1)
        self.pan_n5_conv = ConvBlock(128, 128, kernel_size=3, padding=1)

        # Heads dự đoán
        self.pred_large = DecoupledHead(128, self.num_anchors, self.num_classes)
        self.pred_medium = DecoupledHead(128, self.num_anchors, self.num_classes)
        self.pred_small = DecoupledHead(128, self.num_anchors, self.num_classes)

    def forward(self, x):
        # 1. Backbone
        x = self.backbone_base(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 2. Top-down FPN
        p5_feat = self.lateral_p5(c5)
        
        p4_up = F.interpolate(p5_feat, scale_factor=2, mode='nearest')
        p4_feat = self.lateral_p4(c4) + p4_up
        p4_feat = self.smooth_p4(p4_feat)
        
        p3_up = F.interpolate(p4_feat, scale_factor=2, mode='nearest')
        p3_feat = self.lateral_p3(c3) + p3_up
        p3_feat = self.smooth_p3(p3_feat)

        # 3. Bottom-up PANet (Nâng cấp)
        n3_feat = p3_feat
        
        n4_feat = p4_feat + self.downsample_n3(n3_feat)
        n4_feat = self.pan_n4_conv(n4_feat)
        
        n5_feat = p5_feat + self.downsample_n4(n4_feat)
        n5_feat = self.pan_n5_conv(n5_feat)

        # 4. Truyền vào Heads (Sử dụng đặc trưng đã qua PANet)
        out_large = self.pred_large(n3_feat)
        out_medium = self.pred_medium(n4_feat)
        out_small = self.pred_small(n5_feat)

        return {"large": out_large, "medium": out_medium, "small": out_small}