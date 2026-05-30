import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloLoss(nn.Module):
    """
    YoloLoss v7 — Sạch, ổn định, loss không explode.

    Các thay đổi so với v6 (fix loss=78,000):
      Bug A: Varifocal noobj normalize sai → loss.sum()/1 thay vì /n_cells
             → Fix: dùng reduction='mean' cho noobj, focal weight thuần túy
      Bug B: lambda_noobj=8.0 quá lớn khi loss chưa normalized → loss explode
             → Fix: lambda_noobj=2.0, tăng dần sau khi training ổn định
      Bug C: Quality score target từ IoU ngay epoch 1 = 0 (box chưa học)
             → Fix: dùng target=1.0 cứng cho obj trước, sau đó mới dùng IoU
      
    Chiến lược mới (đơn giản, dễ debug):
      - noobj: Focal BCE, mean reduction, lambda=2.0
      - obj  : BCE với target=1.0, mean reduction, lambda=1.0  
      - box  : CIoU, lambda=5.0
      - class: BCE + label smoothing, lambda=1.0
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes   = num_classes
        self.lambda_class  = 1.0
        self.lambda_obj    = 1.0
        self.lambda_noobj  = 2.0   # KHÔNG tăng quá 3.0 trước khi loss ổn định
        self.lambda_box    = 5.0
        self.label_smooth  = 0.01
        self.focal_gamma   = 1.5   # focal weight cho noobj, không quá cao

        self.anchors_per_scale = {
            0: torch.tensor([[0.05, 0.05], [0.08, 0.08], [0.12, 0.12]]),
            1: torch.tensor([[0.15, 0.15], [0.25, 0.25], [0.35, 0.35]]),
            2: torch.tensor([[0.40, 0.40], [0.55, 0.55], [0.70, 0.70]]),
        }

    def _ciou_loss(self, pred_raw, target_raw, anchor_tensor, S):
        tx_p = torch.sigmoid(pred_raw[:, 0]) / S
        ty_p = torch.sigmoid(pred_raw[:, 1]) / S
        tw_p = anchor_tensor[:, 0] * torch.exp(pred_raw[:, 2].clamp(-4, 4))
        th_p = anchor_tensor[:, 1] * torch.exp(pred_raw[:, 3].clamp(-4, 4))

        tx_t = target_raw[:, 0] / S
        ty_t = target_raw[:, 1] / S
        tw_t = anchor_tensor[:, 0] * torch.exp(target_raw[:, 2])
        th_t = anchor_tensor[:, 1] * torch.exp(target_raw[:, 3])

        px1 = tx_p - tw_p / 2; py1 = ty_p - th_p / 2
        px2 = tx_p + tw_p / 2; py2 = ty_p + th_p / 2
        gx1 = tx_t - tw_t / 2; gy1 = ty_t - th_t / 2
        gx2 = tx_t + tw_t / 2; gy2 = ty_t + th_t / 2

        inter = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(0) * \
                (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(0)
        pred_area   = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
        union = pred_area + target_area - inter + 1e-7
        iou   = inter / union

        d2 = ((px1+px2)/2 - (gx1+gx2)/2)**2 + ((py1+py2)/2 - (gy1+gy2)/2)**2
        c2 = ((torch.min(px1, gx1) - torch.max(px2, gx2))**2 +
              (torch.min(py1, gy1) - torch.max(py2, gy2))**2).clamp(1e-7)

        with torch.no_grad():
            v = (4 / (torch.pi ** 2)) * (
                torch.atan(tw_t / (th_t + 1e-7)) -
                torch.atan(tw_p / (th_p + 1e-7))
            ) ** 2
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - d2 / c2 - alpha * v
        return (1 - ciou).mean()

    def forward(self, predictions, targets):
        scales     = ["large", "medium", "small"]
        total_loss = 0.0

        for i, scale in enumerate(scales):
            pred   = predictions[scale]
            target = targets[i].to(pred.device)

            B, _, H, W = pred.shape
            pred = pred.view(B, 3, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            pred = torch.nan_to_num(pred, nan=0.0, posinf=15.0, neginf=-15.0)
            pred = pred.clamp(-15.0, 15.0)

            obj_mask   = target[..., 4] == 1.0
            noobj_mask = target[..., 4] == 0.0
            pred_conf  = pred[..., 4]

            # ── No-Object Loss: Focal BCE, mean reduction ─────────────────────
            # reduction='mean' → loss tự normalize trên n_noobj_cells
            # KHÔNG dùng .sum() / clamp(1) vì khi target=0, sum=0 → /1 → explode
            if noobj_mask.sum() > 0:
                noobj_logits = pred_conf[noobj_mask]
                noobj_target = torch.zeros_like(noobj_logits)

                # Focal weight: tập trung vào những cell confident sai
                with torch.no_grad():
                    p_sigmoid    = torch.sigmoid(noobj_logits)
                    focal_weight = (p_sigmoid ** self.focal_gamma)   # high conf → high penalty

                bce_noobj  = F.binary_cross_entropy_with_logits(
                    noobj_logits, noobj_target, reduction='none')
                noobj_loss = (focal_weight * bce_noobj).mean()       # mean trên n_noobj
            else:
                noobj_loss = pred_conf.new_tensor(0.0)

            # ── Object Loss: BCE đơn giản, target=1.0 ────────────────────────
            # Dùng target=1.0 cứng thay vì IoU-quality để ổn định epoch đầu
            if obj_mask.sum() > 0:
                obj_logits = pred_conf[obj_mask]
                obj_target = torch.ones_like(obj_logits)
                obj_loss   = F.binary_cross_entropy_with_logits(
                    obj_logits, obj_target, reduction='mean')

                # ── Box Loss: CIoU ────────────────────────────────────────────
                pred_obj   = pred[obj_mask]
                target_obj = target[obj_mask]

                anchors_base  = self.anchors_per_scale[i].to(pred.device)
                anchor_idx    = torch.nonzero(obj_mask)[:, 1]
                anchor_tensor = anchors_base[anchor_idx]
                box_loss      = self._ciou_loss(
                    pred_obj[:, :4], target_obj[:, :4], anchor_tensor, H)

                # ── Class Loss: BCE + label smoothing ─────────────────────────
                n   = self.num_classes
                eps = self.label_smooth
                cls_tgt   = target_obj[:, 5:].clamp(0, 1)
                cls_tgt   = cls_tgt * (1 - eps) + eps / n   # smooth
                class_loss = F.binary_cross_entropy_with_logits(
                    pred_obj[:, 5:], cls_tgt, reduction='mean')

            else:
                obj_loss   = pred_conf.new_tensor(0.0)
                box_loss   = pred_conf.new_tensor(0.0)
                class_loss = pred_conf.new_tensor(0.0)

            # ── Tổng hợp ──────────────────────────────────────────────────────
            scale_loss = (
                self.lambda_noobj * noobj_loss +
                self.lambda_obj   * obj_loss   +
                self.lambda_box   * box_loss   +
                self.lambda_class * class_loss
            )
            total_loss += scale_loss

        return total_loss / len(scales)