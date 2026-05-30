import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """
    Focal Loss dạng BCE - ép model tập trung vào các ví dụ khó.
    Giảm contribution của các background dễ phân loại (easy negatives).
    γ=2 là giá trị chuẩn từ paper RetinaNet.
    """
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, pred_logits, target):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
        p_t = torch.exp(-bce)                          # xác suất đúng
        focal_weight = (1 - p_t) ** self.gamma         # giảm easy examples
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class YoloLoss(nn.Module):
    """
    Multi-scale YOLOv3 Loss v2 — Tối ưu Precision/Recall balance

    Thay đổi so với v1:
      1. Focal Loss cho objectness → tự động giảm gradient từ easy background
      2. lambda_noobj tăng từ 0.5 → 2.0 → phạt nặng hơn khi bắn nhầm background
      3. lambda_obj giảm từ 5.0 → 3.0 → cân bằng lại với noobj
      4. CIoU loss thay MSE cho box regression → học hình dạng box tốt hơn

    Quy ước encode (nhất quán dataset.py & predict.py):
      tx, ty : offset trong cell [0,1] → BCEWithLogitsLoss
      tw, th : log(w/anchor_w)         → dùng trong CIoU reconstruction
      obj    : BCE / Focal
      cls    : BCEWithLogitsLoss
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        self.focal_bce = FocalBCELoss(gamma=2.0, reduction="mean")
        self.bce       = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse       = nn.MSELoss(reduction="mean")

        # ── Trọng số loss ──────────────────────────────────────────────
        # Vấn đề hiện tại: 108K predictions → precision=0.015
        # Nguyên nhân: model không bị phạt đủ khi predict nhầm background
        # Giải pháp: tăng lambda_noobj lên 2.0, dùng Focal cho obj
        self.lambda_class  = 1.0
        self.lambda_obj    = 3.0   # Giảm từ 5 → 3
        self.lambda_noobj  = 2.0   # Tăng từ 0.5 → 2.0 (quan trọng nhất!)
        self.lambda_box    = 5.0

        # Anchors để reconstruct box khi tính CIoU (khớp dataset.py)
        self.anchors_per_scale = {
            0: torch.tensor([[0.05,0.05],[0.08,0.08],[0.12,0.12]]),  # large
            1: torch.tensor([[0.15,0.15],[0.25,0.25],[0.35,0.35]]),  # medium
            2: torch.tensor([[0.40,0.40],[0.55,0.55],[0.70,0.70]]),  # small
        }

    def _ciou_loss(self, pred_box_raw, target_box_raw, anchor_tensor, grid_size):
        """
        CIoU loss — học cả vị trí, kích thước và tỉ lệ khung hình.

        pred_box_raw   : (N, 4) — [tx_logit, ty_logit, tw_raw, th_raw]
        target_box_raw : (N, 4) — [tx_01, ty_01, tw_log, th_log]
        anchor_tensor  : (N, 2) — [anchor_w, anchor_h] cho mỗi prediction
        grid_size      : int    — kích thước lưới (28, 14 hoặc 7)
        """
        S = grid_size

        # Decode pred → normalized [0,1] center-xywh
        tx_pred = torch.sigmoid(pred_box_raw[:, 0]) / S
        ty_pred = torch.sigmoid(pred_box_raw[:, 1]) / S
        tw_pred = anchor_tensor[:, 0] * torch.exp(pred_box_raw[:, 2].clamp(-4, 4))
        th_pred = anchor_tensor[:, 1] * torch.exp(pred_box_raw[:, 3].clamp(-4, 4))

        # Decode target → normalized [0,1] center-xywh
        tx_tgt  = target_box_raw[:, 0] / S
        ty_tgt  = target_box_raw[:, 1] / S
        tw_tgt  = anchor_tensor[:, 0] * torch.exp(target_box_raw[:, 2])
        th_tgt  = anchor_tensor[:, 1] * torch.exp(target_box_raw[:, 3])

        # Chuyển về xyxy
        px1 = tx_pred - tw_pred / 2;  py1 = ty_pred - th_pred / 2
        px2 = tx_pred + tw_pred / 2;  py2 = ty_pred + th_pred / 2
        gx1 = tx_tgt  - tw_tgt  / 2;  gy1 = ty_tgt  - th_tgt  / 2
        gx2 = tx_tgt  + tw_tgt  / 2;  gy2 = ty_tgt  + th_tgt  / 2

        # IoU
        ix1 = torch.max(px1, gx1);  iy1 = torch.max(py1, gy1)
        ix2 = torch.min(px2, gx2);  iy2 = torch.min(py2, gy2)
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        pred_area   = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
        union = pred_area + target_area - inter + 1e-7
        iou   = inter / union

        # Distance penalty (DIoU term)
        cx_pred = (px1 + px2) / 2;  cy_pred = (py1 + py2) / 2
        cx_tgt  = (gx1 + gx2) / 2;  cy_tgt  = (gy1 + gy2) / 2
        d2 = (cx_pred - cx_tgt) ** 2 + (cy_pred - cy_tgt) ** 2

        # Enclosing box diagonal
        ex1 = torch.min(px1, gx1);  ey1 = torch.min(py1, gy1)
        ex2 = torch.max(px2, gx2);  ey2 = torch.max(py2, gy2)
        c2  = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

        # Aspect ratio consistency (v term)
        with torch.no_grad():
            atan_tgt  = torch.atan(tw_tgt  / (th_tgt  + 1e-7))
            atan_pred = torch.atan(tw_pred / (th_pred + 1e-7))
            v = (4 / (torch.pi ** 2)) * (atan_tgt - atan_pred) ** 2
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - d2 / c2 - alpha * v
        return (1 - ciou).mean()

    def forward(self, predictions, targets):
        scales     = ['large', 'medium', 'small']
        total_loss = 0.0

        for i, scale in enumerate(scales):
            pred   = predictions[scale]
            target = targets[i].to(pred.device)

            B, _, H, W = pred.shape
            pred = pred.view(B, 3, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # (B,3,H,W,5+C)

            pred = torch.nan_to_num(pred, nan=0.0, posinf=15.0, neginf=-15.0)
            pred = torch.clamp(pred, -15.0, 15.0)

            obj_mask   = target[..., 4] == 1.0
            noobj_mask = target[..., 4] == 0.0

            # ── 1. No-object loss (Focal) — phạt nặng false positive ──
            if noobj_mask.sum() > 0:
                noobj_loss = self.focal_bce(
                    pred[..., 4:5][noobj_mask],
                    target[..., 4:5][noobj_mask]
                )
            else:
                noobj_loss = torch.tensor(0.0, device=pred.device)

            if obj_mask.sum() == 0:
                total_loss += self.lambda_noobj * noobj_loss
                continue

            pred_obj   = pred[obj_mask]
            target_obj = target[obj_mask]

            # ── 2. Object confidence loss ──────────────────────────────
            obj_loss = self.bce(pred_obj[:, 4:5], target_obj[:, 4:5])

            # ── 3. CIoU Box loss ───────────────────────────────────────
            anchors_base = self.anchors_per_scale[i].to(pred.device)  # (3,2)

            # Tìm anchor index cho mỗi positive (theo dim anchor)
            # obj_mask shape: (B,3,H,W) → nonzero → cột 1 là anchor idx
            anchor_indices = torch.nonzero(obj_mask)[:, 1]             # (N,)
            anchor_tensor  = anchors_base[anchor_indices]               # (N,2)

            box_loss = self._ciou_loss(
                pred_obj[:, 0:4], target_obj[:, 0:4], anchor_tensor, H
            )

            # ── 4. Class loss ──────────────────────────────────────────
            class_loss = self.bce(pred_obj[:, 5:], target_obj[:, 5:])

            # ── 5. Tổng hợp ────────────────────────────────────────────
            scale_loss = (
                self.lambda_noobj * noobj_loss
                + self.lambda_obj   * obj_loss
                + self.lambda_box   * box_loss
                + self.lambda_class * class_loss
            )
            total_loss += scale_loss

        return total_loss / len(scales)