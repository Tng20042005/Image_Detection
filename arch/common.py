import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.reg_conv = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        
        self.cls_conv = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * (1 + num_classes), kernel_size=1)

    def forward(self, x):
        reg = self.reg_pred(self.reg_conv(x))
        cls_and_obj = self.cls_pred(self.cls_conv(x))
        
        B, _, H, W = x.shape
        reg = reg.view(B, self.num_anchors, 4, H, W)
        cls_and_obj = cls_and_obj.view(B, self.num_anchors, 1 + self.num_classes, H, W)
        
        obj = cls_and_obj[:, :, :1, :, :]
        cls_prob = cls_and_obj[:, :, 1:, :, :]
        
        out = torch.cat([reg[:, :, :2, :, :], reg[:, :, 2:, :, :], obj, cls_prob], dim=2)
        return out.view(B, -1, H, W)