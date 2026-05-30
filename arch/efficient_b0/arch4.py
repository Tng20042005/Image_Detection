import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .common import ConvBlock, DecoupledHead


# ── Các khối cơ sở (giữ nguyên) ───────────────────────────────────────────────
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.cv2 = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False), nn.BatchNorm2d(c_), nn.ReLU(inplace=True))
        self.cv2 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False), nn.BatchNorm2d(c_), nn.ReLU(inplace=True))
        self.cv3 = nn.Sequential(
            nn.Conv2d(2*c_, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden*4, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.maxpool(x); y2 = self.maxpool(y1); y3 = self.maxpool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ── SE Block: thay thế Gate đơn giản ─────────────────────────────────────────
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Hu et al., 2018).

    Ưu điểm so với Gate cũ (single Conv2d + Sigmoid):
    - Gate cũ: weight per-pixel → gây checkerboard artifacts, không học được
      global context của feature map.
    - SE Block: squeeze toàn bộ spatial → Global Average Pool → 1×1 vector,
      rồi học channel importance qua 2-layer MLP. Model biết KÊNH NÀO quan trọng
      ở scale này, không phải pixel nào → suppress background hiệu quả hơn.

    ratio=4: bottleneck MLP nhỏ để tránh overfit trên số channel ít (128).
    """
    def __init__(self, channels, ratio=4):
        super().__init__()
        reduced = max(channels // ratio, 8)
        self.squeeze  = nn.AdaptiveAvgPool2d(1)
        self.excite   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.excite(self.squeeze(x))          # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)  # (B, C, H, W)


# ── Task-Aligned Detection Head ───────────────────────────────────────────────
class TaskAlignedHead(nn.Module):
    """
    Tách biệt nhánh Classification và Localization (như TOOD, 2021).
    Thêm "task-aligned" scoring: output conf = cls_score * loc_score
    trước khi đưa vào prediction head.

    Điều này giải quyết vấn đề: Box predict đúng vị trí nhưng conf cao vì
    class score cao, không liên quan đến IoU. Task alignment buộc model chỉ
    confident khi CẢ HAI nhánh đều confident.

    align_corners=False đảm bảo không thêm computation tại inference.
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Nhánh classification
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Nhánh localization
        self.loc_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Prediction layers
        self.pred_box  = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.pred_obj  = nn.Conv2d(in_channels, num_anchors * 1, 1)
        self.pred_cls  = nn.Conv2d(in_channels, num_anchors * num_classes, 1)

        # Task-alignment score: học cách blend cls và loc signal
        self.alignment_cls = nn.Conv2d(in_channels, num_anchors, 1)
        self.alignment_loc = nn.Conv2d(in_channels, num_anchors, 1)

        self._init_bias()

    def _init_bias(self):
        """
        Khởi tạo bias objectness âm: giả sử trước khi train, xác suất object
        tại bất kỳ anchor nào là thấp (prior ~0.01).
        Điều này cực kỳ quan trọng: ngăn model bắn confidence cao ngay từ đầu
        train, làm tăng precision từ epoch đầu.
        prior = 0.01 → bias = log(0.01/0.99) ≈ -4.6
        """
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.pred_obj.bias, bias_value.item())
        nn.init.constant_(self.pred_cls.bias, bias_value.item())

    def forward(self, x):
        cls_feat = self.cls_branch(x)
        loc_feat = self.loc_branch(x)

        # Prediction
        box_pred = self.pred_box(loc_feat)
        obj_pred = self.pred_obj(loc_feat)
        cls_pred = self.pred_cls(cls_feat)

        # Task-aligned score: độ tự tin chỉ cao khi cả classification
        # và localization đều mạnh ở cùng vị trí
        align_cls = torch.sigmoid(self.alignment_cls(cls_feat))  # (B, A, H, W)
        align_loc = torch.sigmoid(self.alignment_loc(loc_feat))  # (B, A, H, W)
        align_score = (align_cls * align_loc) ** 0.5  # geometric mean

        # Nhân alignment score vào obj_pred (logit space: cộng log)
        # Sử dụng logit để không phá vỡ gradient flow
        align_logit = torch.log(align_score.clamp(1e-6) / (1 - align_score).clamp(1e-6))
        obj_pred = obj_pred + 0.5 * align_logit  # blend nhẹ để không overpower

        B, _, H, W = x.shape
        A = self.num_anchors
        C = self.num_classes

        box_pred = box_pred.view(B, A, 4,  H, W).permute(0, 1, 3, 4, 2).contiguous()
        obj_pred = obj_pred.view(B, A, 1,  H, W).permute(0, 1, 3, 4, 2).contiguous()
        cls_pred = cls_pred.view(B, A, C,  H, W).permute(0, 1, 3, 4, 2).contiguous()

        out = torch.cat([box_pred, obj_pred, cls_pred], dim=-1)  # (B, A, H, W, 5+C)
        out = out.permute(0, 1, 4, 2, 3).contiguous()            # (B, A*(5+C), H, W)
        return out.view(B, A * (5 + C), H, W)


# ── Main Model ────────────────────────────────────────────────────────────────
class YOLO_EfficientNetB0(nn.Module):
    """
    YOLO với EfficientNet-B0 backbone + FPN-PAN neck + SE attention + Task-aligned head.

    Thay đổi so với v1:
    1. SEBlock thay Gate cũ (channel-wise squeeze-excitation > pixel-wise gate)
    2. TaskAlignedHead thay DecoupledHead (task-alignment score + prior bias init)
    3. Dropout giữ nguyên nhưng giảm p=0.2 (0.3 quá aggressive gây underfit)
    4. Bias init âm ở head: ngăn false positive từ epoch 0
    """
    def __init__(self, num_classes=5, num_anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_scale

        # ── Backbone EfficientNet-B0 ──────────────────────────────────────────
        eff = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT).features
        self.layer2 = eff[0:4]   # P3: stride 8,  40ch
        self.layer3 = eff[4:6]   # P4: stride 16, 112ch
        self.layer4 = eff[6:8]   # P5: stride 32, 320ch

        # ── Neck ─────────────────────────────────────────────────────────────
        self.sppf = SPPF(320, 128)

        self.lateral_p4 = nn.Conv2d(112, 128, 1)
        self.lateral_p3 = nn.Conv2d(40, 128, 1)

        self.smooth_p4 = C3Block(128, 128, n=3, shortcut=False)
        self.smooth_p3 = C3Block(128, 128, n=3, shortcut=False)

        self.downsample_n3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pan_n4_conv = C3Block(128, 128, n=3, shortcut=False)

        self.downsample_n4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pan_n5_conv = C3Block(128, 128, n=3, shortcut=False)

        # ── SE Attention (thay Gate cũ) ───────────────────────────────────────
        # SEBlock học channel importance globally → suppress background channels
        # tốt hơn pixel-wise gating. ratio=4 cho channels=128.
        self.se_large  = SEBlock(128, ratio=4)
        self.se_medium = SEBlock(128, ratio=4)
        self.se_small  = SEBlock(128, ratio=4)

        # ── Dropout (giảm xuống 0.2 — 0.3 trước đó quá mạnh, gây underfit) ──
        self.drop_large  = nn.Dropout2d(p=0.2)
        self.drop_medium = nn.Dropout2d(p=0.2)
        self.drop_small  = nn.Dropout2d(p=0.2)

        # ── Detection Heads ───────────────────────────────────────────────────
        self.pred_large  = TaskAlignedHead(128, self.num_anchors, self.num_classes)
        self.pred_medium = TaskAlignedHead(128, self.num_anchors, self.num_classes)
        self.pred_small  = TaskAlignedHead(128, self.num_anchors, self.num_classes)

    def forward(self, x):
        # 1. Backbone
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # 2. Top-down FPN
        p5_feat = self.sppf(c5)

        p4_up   = F.interpolate(p5_feat, scale_factor=2, mode='nearest')
        p4_feat = self.smooth_p4(self.lateral_p4(c4) + p4_up)

        p3_up   = F.interpolate(p4_feat, scale_factor=2, mode='nearest')
        p3_feat = self.smooth_p3(self.lateral_p3(c3) + p3_up)

        # 3. Bottom-up PANet
        n3_feat = p3_feat
        n4_feat = self.pan_n4_conv(p4_feat + self.downsample_n3(n3_feat))
        n5_feat = self.pan_n5_conv(p5_feat + self.downsample_n4(n4_feat))

        # 4. SE Attention: channel-wise recalibration
        n3_feat = self.se_large(n3_feat)
        n4_feat = self.se_medium(n4_feat)
        n5_feat = self.se_small(n5_feat)

        # 5. Regularization
        n3_feat = self.drop_large(n3_feat)
        n4_feat = self.drop_medium(n4_feat)
        n5_feat = self.drop_small(n5_feat)

        # 6. Prediction
        return {
            "large":  self.pred_large(n3_feat),
            "medium": self.pred_medium(n4_feat),
            "small":  self.pred_small(n5_feat),
        }