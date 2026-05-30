import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .common import ConvBlock, DecoupledHead

class Bottleneck(nn.Module):
    """Khối kiến trúc Residual cơ bản giúp tăng độ sâu mạng không bị vanishing gradient"""
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3Block(nn.Module):
    """
    CSP Block (Cross Stage Partial) - Trái tim của Neck YOLOv5.
    Giúp chia đôi luồng tính toán, tăng khả năng học đặc trưng sâu sắc, giảm False Positives.
    """
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2  # Kênh ẩn (hidden channels)
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, kernel_size=1, bias=False), nn.BatchNorm2d(c_), nn.ReLU(inplace=True))
        self.cv2 = nn.Sequential(nn.Conv2d(c1, c_, kernel_size=1, bias=False), nn.BatchNorm2d(c_), nn.ReLU(inplace=True))
        
        # 🔥 ĐÃ FIX: Sửa nn.BatchNorm2d(c_) thành nn.BatchNorm2d(c2) để khớp với output của Conv2d
        self.cv3 = nn.Sequential(
            nn.Conv2d(2 * c_, c2, kernel_size=1, bias=False), 
            nn.BatchNorm2d(c2), 
            nn.ReLU(inplace=True)
        )
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast: Mở rộng vùng nhìn ngữ cảnh ở đỉnh mạng"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = nn.Sequential(nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = nn.Sequential(nn.Conv2d(hidden * 4, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class YOLO_EfficientNetB0(nn.Module):
    def __init__(self, num_classes=5, num_anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_scale

        # Backbone EfficientNet-B0 lấy 3 tầng đặc trưng
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        self.layer2 = eff[0:4]   # P3: 40 channels
        self.layer3 = eff[4:6]   # P4: 112 channels
        self.layer4 = eff[6:8]   # P5: 320 channels

        # Đỉnh mạng tích hợp SPPF
        self.sppf = SPPF(320, 128)
        
        # Nhánh ép kênh Top-down FPN
        self.lateral_p4 = nn.Conv2d(112, 128, kernel_size=1)
        self.lateral_p3 = nn.Conv2d(40, 128, kernel_size=1)
        
        # 🔥 THAY THẾ: Sử dụng C3Block dày dặn thay cho ConvBlock đơn lẻ
        self.smooth_p4 = C3Block(128, 128, n=1, shortcut=False)
        self.smooth_p3 = C3Block(128, 128, n=1, shortcut=False)

        # Nhánh Bottom-up PANet
        self.downsample_n3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pan_n4_conv = C3Block(128, 128, n=1, shortcut=False)
        
        self.downsample_n4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pan_n5_conv = C3Block(128, 128, n=1, shortcut=False)

        # Predict Heads chuyên biệt cho từng scale
        self.pred_large = DecoupledHead(128, self.num_anchors, self.num_classes)
        self.pred_medium = DecoupledHead(128, self.num_anchors, self.num_classes)
        self.pred_small = DecoupledHead(128, self.num_anchors, self.num_classes)

    def forward(self, x):
        # 1. Trích xuất đặc trưng Backbone
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 2. Cổ mạng Top-down FPN với SPPF
        p5_feat = self.sppf(c5)
        
        p4_up = F.interpolate(p5_feat, scale_factor=2, mode='nearest')
        p4_feat = self.lateral_p4(c4) + p4_up
        p4_feat = self.smooth_p4(p4_feat)
        
        p3_up = F.interpolate(p4_feat, scale_factor=2, mode='nearest')
        p3_feat = self.lateral_p3(c3) + p3_up
        p3_feat = self.smooth_p3(p3_feat)

        # 3. Cổ mạng Bottom-up PANet (CSP-PANet nâng cấp)
        n3_feat = p3_feat
        
        n4_feat = p4_feat + self.downsample_n3(n3_feat)
        n4_feat = self.pan_n4_conv(n4_feat)
        
        n5_feat = p5_feat + self.downsample_n4(n4_feat)
        n5_feat = self.pan_n5_conv(n5_feat)

        # 4. Đưa ra các đầu dự đoán
        out_large = self.pred_large(n3_feat)
        out_medium = self.pred_medium(n4_feat)
        out_small = self.pred_small(n5_feat)

        return {"large": out_large, "medium": out_medium, "small": out_small}