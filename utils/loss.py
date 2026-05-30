import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloLoss(nn.Module):
    """
    YoloLoss v5 — Bản giải cứu Recall bằng cơ chế tách biệt mẫu số Object/No-Object
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

        # Trọng số cân bằng mới khi các mẫu số đã được cô lập biệt lập
        self.lambda_class  = 1.0
        self.lambda_obj    = 1.0   # Đứng độc lập trên số lượng positive
        self.lambda_noobj  = 1.0   # Đứng độc lập trên số lượng negative + Focal
        self.lambda_box    = 5.0

        self.anchors_per_scale = {
            0: torch.tensor([[0.05,0.05],[0.08,0.08],[0.12,0.12]]),  
            1: torch.tensor([[0.15,0.15],[0.25,0.25],[0.35,0.35]]),  
            2: torch.tensor([[0.40,0.40],[0.55,0.55],[0.70,0.70]]),  
        }

    def _ciou_loss(self, pred_box_raw, target_box_raw, anchor_tensor, grid_size):
        S = grid_size
        tx_pred = torch.sigmoid(pred_box_raw[:, 0]) / S
        ty_pred = torch.sigmoid(pred_box_raw[:, 1]) / S
        tw_pred = anchor_tensor[:, 0] * torch.exp(pred_box_raw[:, 2].clamp(-4, 4))
        th_pred = anchor_tensor[:, 1] * torch.exp(pred_box_raw[:, 3].clamp(-4, 4))

        tx_tgt  = target_box_raw[:, 0] / S
        ty_tgt  = target_box_raw[:, 1] / S
        tw_tgt  = anchor_tensor[:, 0] * torch.exp(target_box_raw[:, 2])
        th_tgt  = anchor_tensor[:, 1] * torch.exp(target_box_raw[:, 3])

        px1 = tx_pred - tw_pred / 2;  py1 = ty_pred - th_pred / 2
        px2 = tx_pred + tw_pred / 2;  py2 = ty_pred + th_pred / 2
        gx1 = tx_tgt  - tw_tgt  / 2;  gy1 = ty_tgt  - th_tgt  / 2
        gx2 = tx_tgt  + tw_tgt  / 2;  gy2 = ty_tgt  + th_tgt  / 2

        ix1 = torch.max(px1, gx1);  iy1 = torch.max(py1, gy1)
        ix2 = torch.min(px2, gx2);  iy2 = torch.min(py2, gy2)
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        pred_area   = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
        union = pred_area + target_area - inter + 1e-7
        iou   = inter / union

        cx_pred = (px1 + px2) / 2;  cy_pred = (py1 + py2) / 2
        cx_tgt  = (gx1 + gx2) / 2;  cy_tgt  = (gy1 + gy2) / 2
        d2 = (cx_pred - cx_tgt) ** 2 + (cy_pred - cy_tgt) ** 2

        ex1 = torch.min(px1, gx1);  ey1 = torch.min(py1, gy1)
        ex2 = torch.max(px2, gx2);  ey2 = torch.max(py2, gy2)
        c2  = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

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
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  

            pred = torch.nan_to_num(pred, nan=0.0, posinf=15.0, neginf=-15.0)
            pred = torch.clamp(pred, -15.0, 15.0)

            obj_mask   = target[..., 4] == 1.0
            noobj_mask = target[..., 4] == 0.0

            pred_conf = pred[..., 4]   
            target_conf = target[..., 4] 

            # ── 🔥 ĐÃ FIX CHÍ MẠNG V5: TÁCH BIỆT HOÀN TOÀN MẪU SỐ CONFIDENCE ──
            # 1. Tính loss cho vùng nền (No-Object) và chia trung bình trên chính số lượng ô nền
            if noobj_mask.sum() > 0:
                bce_noobj = F.binary_cross_entropy_with_logits(pred_conf[noobj_mask], target_conf[noobj_mask], reduction="none")
                p_t = torch.exp(-bce_noobj)
                focal_weight = (1 - p_t) ** 2.0  # Khôi phục gamma=2.0 chuẩn RetinaNet
                noobj_loss = (focal_weight * bce_noobj).mean()
            else:
                noobj_loss = torch.tensor(0.0, device=pred.device)

            # 2. Tính loss cho vùng có vật thể (Object) và chia trung bình CHỈ trên các ô có vật thể
            if obj_mask.sum() > 0:
                obj_loss = F.binary_cross_entropy_with_logits(pred_conf[obj_mask], target_conf[obj_mask], reduction="mean")
            else:
                obj_loss = torch.tensor(0.0, device=pred.device)

            # Tổng hợp conf_loss từ hai luồng độc lập
            conf_loss = self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss

            if obj_mask.sum() == 0:
                total_loss += conf_loss
                continue

            # ── Tính Box Loss và Class Loss ──
            pred_obj   = pred[obj_mask]
            target_obj = target[obj_mask]

            anchors_base = self.anchors_per_scale[i].to(pred.device)  
            anchor_indices = torch.nonzero(obj_mask)[:, 1]             
            anchor_tensor  = anchors_base[anchor_indices]               

            box_loss = self._ciou_loss(pred_obj[:, 0:4], target_obj[:, 0:4], anchor_tensor, H)
            class_loss = self.bce(pred_obj[:, 5:], target_obj[:, 5:])

            scale_loss = conf_loss + self.lambda_box * box_loss + self.lambda_class * class_loss
            total_loss += scale_loss

        return total_loss / len(scales)