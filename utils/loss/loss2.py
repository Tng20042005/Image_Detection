import torch
import torch.nn as nn
import torch.nn.functional as F

class YoloLoss(nn.Module):
    """
    Multi-scale YOLOv3 Loss v3 — Đã sửa lỗi lệch mẫu số và mất kết nối không gian CIoU
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

        # Trọng số loss chuẩn hóa cân bằng
        self.lambda_class  = 1.0
        self.lambda_obj    = 3.0   
        self.lambda_noobj  = 2.0   
        self.lambda_box    = 5.0

        # Anchors tương ứng từng scale (khớp dataset.py)
        self.anchors_per_scale = {
            0: torch.tensor([[0.05,0.05],[0.08,0.08],[0.12,0.12]]),  # large
            1: torch.tensor([[0.15,0.15],[0.25,0.25],[0.35,0.35]]),  # medium
            2: torch.tensor([[0.40,0.40],[0.55,0.55],[0.70,0.70]]),  # small
        }

    def _ciou_loss(self, pred_box_raw, target_box_raw, anchor_tensor, grid_size, grid_x, grid_y):
        """
        CIoU loss chuẩn hóa — Đã thêm grid_x, grid_y để định vị tuyệt đối trên ảnh
        """
        S = grid_size

        # 🔥 ĐÃ FIX: Cộng thêm tọa độ Grid Cell (grid_x, grid_y) vào phép giải mã hình học
        tx_pred = (grid_x + torch.sigmoid(pred_box_raw[:, 0])) / S
        ty_pred = (grid_y + torch.sigmoid(pred_box_raw[:, 1])) / S
        tw_pred = anchor_tensor[:, 0] * torch.exp(pred_box_raw[:, 2].clamp(-4, 4))
        th_pred = anchor_tensor[:, 1] * torch.exp(pred_box_raw[:, 3].clamp(-4, 4))

        tx_tgt  = (grid_x + target_box_raw[:, 0]) / S
        ty_tgt  = (grid_y + target_box_raw[:, 1]) / S
        tw_tgt  = anchor_tensor[:, 0] * torch.exp(target_box_raw[:, 2])
        th_tgt  = anchor_tensor[:, 1] * torch.exp(target_box_raw[:, 3])

        # Chuyển đổi về xyxy toàn cục
        px1 = tx_pred - tw_pred / 2;  py1 = ty_pred - th_pred / 2
        px2 = tx_pred + tw_pred / 2;  py2 = ty_pred + th_pred / 2
        gx1 = tx_tgt  - tw_tgt  / 2;  gy1 = ty_tgt  - th_tgt  / 2
        gx2 = tx_tgt  + tw_tgt  / 2;  gy2 = ty_tgt  + th_tgt  / 2

        # Tính toán IoU lý thuyết
        ix1 = torch.max(px1, gx1);  iy1 = torch.max(py1, gy1)
        ix2 = torch.min(px2, gx2);  iy2 = torch.min(py2, gy2)
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        pred_area   = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
        union = pred_area + target_area - inter + 1e-7
        iou   = inter / union

        # Khoảng cách tâm (DIoU term)
        cx_pred = (px1 + px2) / 2;  cy_pred = (py1 + py2) / 2
        cx_tgt  = (gx1 + gx2) / 2;  cy_tgt  = (gy1 + gy2) / 2
        d2 = (cx_pred - cx_tgt) ** 2 + (cy_pred - cy_tgt) ** 2

        # Khung bao nhỏ nhất bao quanh cả 2 box
        ex1 = torch.min(px1, gx1);  ey1 = torch.min(py1, gy1)
        ex2 = torch.max(px2, gx2);  ey2 = torch.max(py2, gy2)
        c2  = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

        # Tỷ lệ khung hình (CIoU aspect ratio term)
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

            # ── 🔥 ĐÃ FIX 1: TÍNH CONFIDENCE LOSS TRÊN TOÀN BỘ LƯỚI CHUNG MẪU SỐ ──
            pred_conf = pred[..., 4]   # shape: (B, 3, H, W)
            target_conf = target[..., 4] # shape: (B, 3, H, W)

            # Tính BCE raw không lấy mean vội
            bce_conf = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction="none")
            
            # Áp dụng cơ chế Focal Loss lên phần background để triệt tiêu easy negative
            p_t = torch.exp(-bce_conf)
            focal_weight = (1 - p_t) ** 2.0
            
            # Khởi tạo ma trận loss rỗng để phân phối trọng số
            loss_conf_grid = torch.zeros_like(bce_conf)
            
            # Dòng dương (vật thể): giữ nguyên BCE nguyên bản * lambda_obj
            loss_conf_grid[obj_mask] = bce_conf[obj_mask] * self.lambda_obj
            # Dòng âm (nền): ép Focal BCE * lambda_noobj để phạt nặng các vùng ảo giác huyễn hoặc
            loss_conf_grid[noobj_mask] = (focal_weight * bce_conf)[noobj_mask] * self.lambda_noobj
            
            # Lấy mean trên toàn bộ grid scale này để mẫu số không bị loãng
            conf_loss = loss_conf_grid.mean()

            # Nếu grid hiện tại hoàn toàn không chứa vật thể dương nào, cộng conf_loss và bỏ qua tính box/class
            if obj_mask.sum() == 0:
                total_loss += conf_loss
                continue

            # Trích xuất các ô lưới chứa vật thể thật
            pred_obj   = pred[obj_mask]
            target_obj = target[obj_mask]

            # ── 2. CIoU Box loss ───────────────────────────────────────
            anchors_base = self.anchors_per_scale[i].to(pred.device)  # (3,2)

            # Lấy chỉ mục vị trí cell thực tế từ mặt nạ obj_mask
            nonzero = torch.nonzero(obj_mask)  # (N, 4) -> [batch, anchor_idx, grid_y, grid_x]
            anchor_indices = nonzero[:, 1]
            anchor_tensor  = anchors_base[anchor_indices] # (N, 2)
            
            # Lấy tọa độ lưới thực tế đưa vào tính toán CIoU hình học
            grid_y = nonzero[:, 2].to(pred.device).float()
            grid_x = nonzero[:, 3].to(pred.device).float()

            box_loss = self._ciou_loss(
                pred_obj[:, 0:4], target_obj[:, 0:4], anchor_tensor, H, grid_x, grid_y
            )

            # ── 3. Class loss ──────────────────────────────────────────
            class_loss = self.bce(pred_obj[:, 5:], target_obj[:, 5:])

            # ── 4. Tổng hợp loss của scale hiện tại ────────────────────────
            scale_loss = conf_loss + self.lambda_box * box_loss + self.lambda_class * class_loss
            total_loss += scale_loss

        return total_loss / len(scales)