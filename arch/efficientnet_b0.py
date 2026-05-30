import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .common import ConvBlock, DecoupledHead

# --- [GIỮ NGUYÊN BỘ 3 KHỐI BOTTLENECK, C3BLOCK, SPPF CỦA BẠN] ---
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.cv2 = nn.Sequential(nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2  
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, kernel_size=1, bias=False), nn.BatchNorm2d(c_), nn.ReLU(inplace=True))
        self.cv2 = nn.Sequential(nn.Conv2d(c1, c_, kernel_size=1, bias=False), nn.BatchNorm2d(c_), nn.ReLU(inplace=True))
        self.cv3 = nn.Sequential(nn.Conv2d(2 * c_, c2, kernel_size=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SPPF(nn.Module):
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

        # Backbone EfficientNet-B0 chuẩn chỉnh
        eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
        self.layer2 = eff[0:4]   # P3: 40 channels
        self.layer3 = eff[4:6]   # P4: 112 channels
        self.layer4 = eff[6:8]   # P5: 320 channels

        self.sppf = SPPF(320, 128)
        
        self.lateral_p4 = nn.Conv2d(112, 128, kernel_size=1)
        self.lateral_p3 = nn.Conv2d(40, 128, kernel_size=1)
        
        self.smooth_p4 = C3Block(128, 128, n=3, shortcut=False)
        self.smooth_p3 = C3Block(128, 128, n=3, shortcut=False)

        self.downsample_n3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pan_n4_conv = C3Block(128, 128, n=3, shortcut=False)
        
        self.downsample_n4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pan_n5_conv = C3Block(128, 128, n=3, shortcut=False)

        # 🔥 CẢI TIẾN 1: Thêm Gate Gating để triệt tiêu kích hoạt nhiễu background yếu
        self.gate_large = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1), nn.Sigmoid())
        self.gate_medium = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1), nn.Sigmoid())
        self.gate_small = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1), nn.Sigmoid())

        # 🔥 CẢI TIẾN 2: Ép Dropout mạnh (0.3) trước Head để phá vỡ sự overconfidence của các neuron đoán mò
        self.drop_large = nn.Dropout2d(p=0.3)
        self.drop_medium = nn.Dropout2d(p=0.3)
        self.drop_small = nn.Dropout2d(p=0.3)

        self.pred_large = DecoupledHead(128, self.num_anchors, self.num_classes)
        self.pred_medium = DecoupledHead(128, self.num_anchors, self.num_classes)
        self.pred_small = DecoupledHead(128, self.num_anchors, self.num_classes)

    def forward(self, x):
        # 1. Backbone
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 2. Top-down FPN
        p5_feat = self.sppf(c5)
        
        p4_up = F.interpolate(p5_feat, scale_factor=2, mode='nearest')
        p4_feat = self.lateral_p4(c4) + p4_up
        p4_feat = self.smooth_p4(p4_feat)
        
        p3_up = F.interpolate(p4_feat, scale_factor=2, mode='nearest')
        p3_feat = self.lateral_p3(c3) + p3_up
        p3_feat = self.smooth_p3(p3_feat)

        # 3. Bottom-up PANet
        n3_feat = p3_feat
        n4_feat = p4_feat + self.downsample_n3(n3_feat)
        n4_feat = self.pan_n4_conv(n4_feat)
        
        n5_feat = p5_feat + self.downsample_n4(n4_feat)
        n5_feat = self.pan_n5_conv(n5_feat)

        # 🔥 Thực hiện cơ chế lọc cổng (Gating Mechanism) để ép các giá trị nhiễu về hẳn mức 0
        n3_feat = n3_feat * self.gate_large(n3_feat)
        n4_feat = n4_feat * self.gate_medium(n4_feat)
        n5_feat = n5_feat * self.gate_small(n5_feat)

        # 🔥 Áp dụng Dropout giải quyết triệt để bệnh tự tin thái quá khi train
        n3_feat = self.drop_large(n3_feat)
        n4_feat = self.drop_medium(n4_feat)
        n5_feat = self.drop_small(n5_feat)

        # 4. Predict Heads
        out_large = self.pred_large(n3_feat)
        out_medium = self.pred_medium(n4_feat)
        out_small = self.pred_small(n5_feat)

        return {"large": out_large, "medium": out_medium, "small": out_small}