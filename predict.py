
import os
import json
import argparse
import time

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

from models.detector import YOLO_Lite
from utils.box_utils import nms

# ================= CONFIG =================
CLASSES = ["person", "car", "dog", "cat", "chair"]

ANCHORS_CFG = {
    'large':  [(0.05, 0.05), (0.08, 0.08), (0.12, 0.12)],
    'medium': [(0.15, 0.15), (0.25, 0.25), (0.35, 0.35)],
    'small':  [(0.40, 0.40), (0.55, 0.55), (0.70, 0.70)],
}

# ================= DATASET =================
class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.image_dir = image_dir

    def __len__(self):
        return len(self.files)

    def letterbox(self, image, size=480):
        w, h = image.size
        scale = size / max(w, h)

        nw, nh = int(w * scale), int(h * scale)

        img = image.resize((nw, nh), Image.BILINEAR)

        pad_l = (size - nw) // 2
        pad_t = (size - nh) // 2

        canvas = Image.new("RGB", (size, size), (128, 128, 128))
        canvas.paste(img, (pad_l, pad_t))

        return canvas, w, h, scale, pad_l, pad_t

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.image_dir, fname)

        img = Image.open(path).convert("RGB")
        img, w, h, scale, pl, pt = self.letterbox(img)

        img = TF.to_tensor(img)
        img = TF.normalize(img,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

        return img, fname, w, h, scale, pl, pt

# ================= COLLATE =================
def custom_collate_fn(batch):
    imgs, names = zip(*batch)
    return list(imgs), list(names)

# ================= LOAD MODEL =================
def load_model(device):
    print("⚙️  Loading model...")

    model = YOLO_Lite(num_classes=5)
    
    if not os.path.exists(args.model_path):
        print(f"⚠️ Cảnh báo: Không tìm thấy '{args.model_path}'. Vui lòng kiểm tra lại đường dẫn!")
    else:
        state = torch.load(args.model_path, map_location="cpu")
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)

    model = model.to(device)
    model.eval()

    print("✅ Model loaded on", device)
    return model

# ================= DECODE =================
def decode_multiscale_predictions(predictions_dict, scale, pad_left, pad_top,
                                   orig_w, orig_h, conf_thresh=0.25):
    boxes = []
    device = next(iter(predictions_dict.values())).device

    for scale_name in ['large', 'medium', 'small']:
        pred = predictions_dict[scale_name]
        B, C, H, W = pred.shape

        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred = pred.view(B, H, W, 3, 5 + len(CLASSES))[0]

        obj = torch.sigmoid(pred[..., 4])
        cls = torch.sigmoid(pred[..., 5:])

        max_cls, cls_id = torch.max(cls, dim=-1)
        score = obj * max_cls

        mask = score >= conf_thresh
        if not mask.any():
            continue

        p = pred[mask]
        s = score[mask]
        c = cls_id[mask]

        idx = torch.nonzero(mask)
        yi = idx[:, 0].float()
        xi = idx[:, 1].float()
        ai = idx[:, 2]

        x = torch.sigmoid(p[:, 0])
        y = torch.sigmoid(p[:, 1])

        x_center = (xi + x) / W
        y_center = (yi + y) / H

        anchors = torch.tensor(ANCHORS_CFG[scale_name], device=device)
        aw = anchors[ai, 0]
        ah = anchors[ai, 1]

        tw = torch.clamp(p[:, 2], -4, 4)
        th = torch.clamp(p[:, 3], -4, 4)

        IMG = 480

        w = aw * torch.exp(tw) * IMG
        h = ah * torch.exp(th) * IMG

        x_px = x_center * IMG
        y_px = y_center * IMG

        xmin = (x_px - w / 2 - pad_left) / scale
        xmax = (x_px + w / 2 - pad_left) / scale
        ymin = (y_px - h / 2 - pad_top) / scale
        ymax = (y_px + h / 2 - pad_top) / scale

        xmin = torch.clamp(xmin, 0, orig_w)
        xmax = torch.clamp(xmax, 0, orig_w)
        ymin = torch.clamp(ymin, 0, orig_h)
        ymax = torch.clamp(ymax, 0, orig_h)

        # 🔥 Giữ nguyên thuật toán Vectorization để chạy 1 GPU với tốc độ bàn thờ
        valid_mask = (xmax > xmin) & (ymax > ymin)
        if valid_mask.any():
            xmin_v = xmin[valid_mask]
            ymin_v = ymin[valid_mask]
            xmax_v = xmax[valid_mask]
            ymax_v = ymax[valid_mask]
            s_v = s[valid_mask]
            c_v = c[valid_mask].float()
            
            boxes_tensor = torch.stack([xmin_v, ymin_v, xmax_v, ymax_v, s_v, c_v], dim=1)
            boxes.extend(boxes_tensor.cpu().tolist())

    return boxes

# ================= MAIN =================
def main(args):

    # 🔥 Chỉ sử dụng 1 GPU mặc định
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("🚀 Device:", device)

    # Tắt benchmark nếu vẫn bị crash trên 1 GPU, nhưng thường 1 GPU thì không sao
    torch.backends.cudnn.benchmark = True 

    model = load_model(device)

    dataset = ImageDataset(args.image_dir)

    loader = DataLoader(
        dataset,
        batch_size=32, # Trả về batch_size=16 như ban đầu cho 1 GPU
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    CONF_THRESH = 0.5
    CLASS_THRESH = {
        0: 0.45,
        1: 0.45,
        2: 0.35,
        3: 0.30,
        4: 0.45,
    }

    results = []
    total = len(dataset)
    processed = 0
    start_time = time.time()

    with torch.no_grad():
        for imgs, names, ws, hs, scales, pls, pts in loader:

            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs)

            B = len(names)

            for i in range(B):
                # Khi chạy 1 GPU không bọc DataParallel, cấu trúc output có thể cần truy cập trực tiếp
                pred_i = {k: v[i:i+1] for k, v in preds.items()}

                raw_boxes = decode_multiscale_predictions(
                    pred_i,
                    scales[i].item() if isinstance(scales[i], torch.Tensor) else scales[i],
                    pls[i].item() if isinstance(pls[i], torch.Tensor) else pls[i],
                    pts[i].item() if isinstance(pts[i], torch.Tensor) else pts[i],
                    ws[i].item() if isinstance(ws[i], torch.Tensor) else ws[i],
                    hs[i].item() if isinstance(hs[i], torch.Tensor) else hs[i],
                    conf_thresh=CONF_THRESH
                )

                nms_boxes = nms(raw_boxes, 0.5, CONF_THRESH)

                img_result = {"image_id": names[i], "boxes": []}

                for b in nms_boxes:
                    x1, y1, x2, y2, conf, cls = b
                    cls = int(cls)

                    if conf < CLASS_THRESH.get(cls, CONF_THRESH):
                        continue

                    # Tự động sắp xếp lại: cái nhỏ làm min, cái lớn làm max
                    ix1, ix2 = min(int(x1), int(x2)), max(int(x1), int(x2))
                    iy1, iy2 = min(int(y1), int(y2)), max(int(y1), int(y2))

                    # Loại bỏ các box bị dẹp lép (chiều rộng hoặc chiều cao = 0)
                    if ix1 == ix2 or iy1 == iy2:
                        continue

                    img_result["boxes"].append({
                        "class": CLASSES[cls],
                        "confidence": round(float(conf), 4),
                        "bbox": [ix1, iy1, ix2, iy2]
                    })

                results.append(img_result)

            # ================= PROGRESS BAR =================
            processed += len(names)
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / speed if speed > 0 else 0

            print(
                f"⏳ {processed}/{total} "
                f"({processed/total*100:.1f}%) "
                f"- {speed:.2f} img/s "
                f"- ETA: {eta:.1f}s",
                end="\r",
                flush=True
            )

    print("\n🎉 Inference done!")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("💾 Saved:", args.output)

# ================= RUN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    main(args)

