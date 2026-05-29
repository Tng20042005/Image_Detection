import os
import json
import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from models.detector import YOLO_Lite
from utils.box_utils import nms

CLASSES = ["person", "car", "dog", "cat", "chair"]

# Anchors (khớp 100% với dataset.py)
ANCHORS_CFG = {
    'large':  [(0.05, 0.05), (0.08, 0.08), (0.12, 0.12)],   # grid 60x60 (stride 8)
    'medium': [(0.15, 0.15), (0.25, 0.25), (0.35, 0.35)],   # grid 30x30 (stride 16)
    'small':  [(0.40, 0.40), (0.55, 0.55), (0.70, 0.70)],   # grid 15x15 (stride 32)
}


def decode_multiscale_predictions(
    predictions_dict, scale, pad_left, pad_top,
    orig_w, orig_h, conf_thresh=0.25
):
    """
    Decode dự đoán đa tỷ lệ về tọa độ ảnh gốc.

    Encode convention (phải khớp dataset.py & loss.py):
      - tx, ty : sigmoid(pred) → offset trong cell [0,1]
      - tw, th : pred (log-space) → w = anchor_w * exp(pred_w)
      - obj    : sigmoid(pred)
      - cls    : sigmoid(pred)
    """
    boxes  = []
    device = next(iter(predictions_dict.values())).device

    for scale_name in ['large', 'medium', 'small']:
        pred = predictions_dict[scale_name]   # (1, A*(5+C), H, W)
        B, C, H, W = pred.shape

        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred = pred.view(B, H, W, 3, 5 + len(CLASSES))[0]   # (H, W, 3, 5+C)

        # ── Tính score tổng hợp = sigmoid(obj) * max sigmoid(cls) ──
        obj_conf       = torch.sigmoid(pred[..., 4])
        class_probs    = torch.sigmoid(pred[..., 5:])
        max_cls_conf, class_ids = torch.max(class_probs, dim=-1)
        final_scores   = obj_conf * max_cls_conf

        mask = final_scores >= conf_thresh
        if not mask.any():
            continue

        filtered_pred     = pred[mask]
        filtered_scores   = final_scores[mask]
        filtered_class_ids = class_ids[mask]

        indices    = torch.nonzero(mask)
        i_idx      = indices[:, 0].float()   # row (y)
        j_idx      = indices[:, 1].float()   # col (x)
        anchor_idx = indices[:, 2]

        # ── Decode tọa độ tâm ──────────────────────────────────────
        x_cell   = torch.sigmoid(filtered_pred[:, 0])
        y_cell   = torch.sigmoid(filtered_pred[:, 1])
        x_center = (j_idx + x_cell) / W   # [0,1] trong ảnh 480
        y_center  = (i_idx + y_cell) / H

        # ── Decode kích thước (anchor * exp(pred)) ─────────────────
        scale_anchors = torch.tensor(ANCHORS_CFG[scale_name], device=device)
        anchor_w = scale_anchors[anchor_idx, 0]
        anchor_h = scale_anchors[anchor_idx, 1]

        # Clamp pred để tránh exp bùng nổ
        tw = torch.clamp(filtered_pred[:, 2], min=-4.0, max=4.0)
        th = torch.clamp(filtered_pred[:, 3], min=-4.0, max=4.0)

        w_norm = anchor_w * torch.exp(tw)   # normalized [0,1] trên 480px
        h_norm = anchor_h * torch.exp(th)

        # ── Quy đổi về tọa độ pixel trong ảnh 480x480 ─────────────
        IMG = 480
        x_px = x_center * IMG
        y_px  = y_center  * IMG
        w_px  = w_norm    * IMG
        h_px  = h_norm    * IMG

        xmin_px = x_px - w_px / 2
        xmax_px = x_px + w_px / 2
        ymin_px  = y_px  - h_px / 2
        ymax_px  = y_px  + h_px / 2

        # ── Loại bỏ padding, đưa về kích thước ảnh gốc ────────────
        xmin = (xmin_px - pad_left) / scale
        xmax = (xmax_px - pad_left) / scale
        ymin  = (ymin_px  - pad_top)  / scale
        ymax  = (ymax_px  - pad_top)  / scale

        xmin = torch.clamp(xmin, min=0.0, max=float(orig_w))
        xmax = torch.clamp(xmax, min=0.0, max=float(orig_w))
        ymin  = torch.clamp(ymin,  min=0.0, max=float(orig_h))
        ymax  = torch.clamp(ymax,  min=0.0, max=float(orig_h))

        # ── Gom kết quả ────────────────────────────────────────────
        xmin_np = xmin.cpu().numpy()
        ymin_np  = ymin.cpu().numpy()
        xmax_np = xmax.cpu().numpy()
        ymax_np  = ymax.cpu().numpy()
        scr_np  = filtered_scores.cpu().numpy()
        id_np   = filtered_class_ids.cpu().numpy()

        for k in range(len(xmin_np)):
            if xmax_np[k] <= xmin_np[k] or ymax_np[k] <= ymin_np[k]:
                continue
            if (xmax_np[k] - xmin_np[k]) < 2 or (ymax_np[k] - ymin_np[k]) < 2:
                continue
            boxes.append([
                float(xmin_np[k]), float(ymin_np[k]),
                float(xmax_np[k]), float(ymax_np[k]),
                float(scr_np[k]),  int(id_np[k])
            ])

    return boxes


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Thiết bị dự đoán: {device}")

    model = YOLO_Lite(num_classes=5).to(device)
    model_path = './models/best.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results     = []
    valid_ext   = ('.jpg', '.jpeg', '.png')
    image_files = sorted([f for f in os.listdir(args.image_dir)
                          if f.lower().endswith(valid_ext)])
    print(f"📦 Tìm thấy {len(image_files)} ảnh.")

    # ── Ngưỡng inference ────────────────────────────────────────────────
    # Phân tích kết quả: cat/dog AP cao (0.7/0.56) nhưng person/car/chair thấp
    # → dùng threshold cao hơn cho class dễ bị false positive
    CONF_THRESH = 0.45    # Tăng mạnh từ 0.35 → cắt bớt false positive
    IOU_THRESH  = 0.40    # Giảm nhẹ IOU threshold → NMS aggressive hơn

    # Per-class confidence threshold (tinh chỉnh sau khi nhìn AP per class)
    # person/car/chair precision rất thấp → dùng threshold cao hơn
    CLASS_THRESH = {
        0: 0.45,  # person  — AP thấp, FP nhiều
        1: 0.45,  # car     — AP thấp, FP nhiều
        2: 0.35,  # dog     — AP trung bình
        3: 0.30,  # cat     — AP cao nhất, giữ threshold thấp hơn
        4: 0.45,  # chair   — AP thấp, FP nhiều
    }

    with torch.no_grad():
        for idx, filename in enumerate(image_files):
            print(f"⏳ [{idx + 1}/{len(image_files)}] {filename}", end="\r", flush=True)

            img_path = os.path.join(args.image_dir, filename)
            image    = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            # Letterbox
            sc     = 480 / max(orig_w, orig_h)
            new_w  = int(orig_w * sc)
            new_h  = int(orig_h * sc)
            img_r  = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            pad_l  = (480 - new_w) // 2
            pad_t  = (480 - new_h) // 2
            canvas = Image.new("RGB", (480, 480), (128, 128, 128))
            canvas.paste(img_r, (pad_l, pad_t))

            img_t = TF.to_tensor(canvas)
            img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            img_t = img_t.unsqueeze(0).to(device)

            predictions = model(img_t)

            # ── Test-Time Augmentation: flip ngang ──────────────────────
            img_flip  = TF.hflip(img_t)
            pred_flip = model(img_flip)

            raw_boxes = decode_multiscale_predictions(
                predictions, sc, pad_l, pad_t, orig_w, orig_h, CONF_THRESH
            )

            # Merge flip predictions (lật lại tọa độ x)
            flip_boxes = decode_multiscale_predictions(
                pred_flip, sc, pad_l, pad_t, orig_w, orig_h, CONF_THRESH
            )
            for b in flip_boxes:
                xmin, ymin, xmax, ymax, conf, cls_id = b
                # Đảo ngược x
                raw_boxes.append([orig_w - xmax, ymin, orig_w - xmin, ymax, conf, cls_id])

            # ── Giới hạn trước NMS để tránh treo máy ───────────────────
            if len(raw_boxes) > 300:
                raw_boxes = sorted(raw_boxes, key=lambda x: x[4], reverse=True)[:300]

            nms_boxes = nms(raw_boxes, iou_threshold=IOU_THRESH,
                            conf_threshold=CONF_THRESH)

            img_result = {"image_id": filename, "boxes": []}
            for box in nms_boxes:
                xmin, ymin, xmax, ymax, conf, cls_id = box
                # Lọc thêm bằng per-class threshold
                if conf < CLASS_THRESH.get(cls_id, CONF_THRESH):
                    continue
                x0 = max(0, int(round(xmin)));  y0 = max(0, int(round(ymin)))
                x1 = min(orig_w, int(round(xmax))); y1 = min(orig_h, int(round(ymax)))
                if x1 <= x0 or y1 <= y0:
                    continue
                img_result["boxes"].append({
                    "class":      CLASSES[cls_id],
                    "confidence": round(float(conf), 4),
                    "bbox":       [x0, y0, x1, y1]
                })
            results.append(img_result)

    print(f"\n✅ Xong! Đang lưu kết quả...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"🎉 Kết quả lưu tại: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output',    type=str, required=True)
    args = parser.parse_args()
    main(args)