import json
import os
import torch
import random
import math
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torchvision.transforms.functional as TF


class CustomDetectionDataset(Dataset):
    """
    Dataset cho object detection với:
      - Letterbox padding (tránh méo)
      - Augmentation mạnh hơn (flip, color jitter, mosaic-lite, scale jitter)
      - Encode target nhất quán với loss.py và predict.py
    """

    def __init__(self, json_path, img_dir, img_size=480, is_train=True):
        self.img_dir  = img_dir
        self.is_train = is_train
        self.img_size = img_size

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.classes      = data['classes']
        self.num_classes  = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.img_data = {}
        for img in data['images']:
            self.img_data[img['id']] = {'info': img, 'annos': []}
        for anno in data['annotations']:
            self.img_data[anno['image_id']]['annos'].append(anno)

        self.image_ids = list(self.img_data.keys())

        # Grid và anchors (khớp predict.py và loss.py)
        # img_size=480: stride 8→60x60, stride 16→30x30, stride 32→15x15
        self.grids       = [60, 30, 15]
        self.num_anchors = 3
        self.anchors = torch.tensor([
            [0.05, 0.05], [0.08, 0.08], [0.12, 0.12],   # scale large (grid 60x60, stride 8)
            [0.15, 0.15], [0.25, 0.25], [0.35, 0.35],   # scale medium (grid 30x30, stride 16)
            [0.40, 0.40], [0.55, 0.55], [0.70, 0.70],   # scale small  (grid 15x15, stride 32)
        ])

    def __len__(self):
        return len(self.image_ids)

    # ─────────────────────────────── helpers ──────────────────────────────

    def _iou_wh(self, box_w, box_h):
        """IoU giữa ground-truth và 9 anchors (chỉ xét w/h)."""
        inter = torch.min(box_w, self.anchors[:, 0]) * torch.min(box_h, self.anchors[:, 1])
        box_a = box_w * box_h
        anc_a = self.anchors[:, 0] * self.anchors[:, 1]
        return inter / (box_a + anc_a - inter + 1e-6)

    def _letterbox(self, image, annos, orig_w, orig_h):
        """Resize giữ tỉ lệ + pad xám."""
        scale  = self.img_size / max(orig_w, orig_h)
        new_w  = int(orig_w * scale)
        new_h  = int(orig_h * scale)
        image  = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        pad_l  = (self.img_size - new_w) // 2
        pad_t  = (self.img_size - new_h) // 2
        canvas = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        canvas.paste(image, (pad_l, pad_t))

        new_annos = []
        for a in annos:
            xmin, ymin, xmax, ymax = a['bbox']
            xmin = xmin * scale + pad_l
            xmax = xmax * scale + pad_l
            ymin = ymin * scale + pad_t
            ymax = ymax * scale + pad_t
            new_annos.append({**a, 'bbox': [xmin, ymin, xmax, ymax]})

        return canvas, new_annos, scale, pad_l, pad_t

    # ─────────────────────────────── augmentation ─────────────────────────

    def _aug_flip(self, image, annos):
        """Lật ngang."""
        image = ImageOps.mirror(image)
        S = self.img_size
        new_annos = []
        for a in annos:
            xmin, ymin, xmax, ymax = a['bbox']
            new_annos.append({**a, 'bbox': [S - xmax, ymin, S - xmin, ymax]})
        return image, new_annos

    def _aug_color_jitter(self, image):
        """Thay đổi màu sắc ngẫu nhiên."""
        if random.random() > 0.5:
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if random.random() > 0.5:
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if random.random() > 0.5:
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Color(image).enhance(factor)
        return image

    def _aug_scale_jitter(self, image, annos):
        """Scale jitter: random zoom in/out nhẹ."""
        factor = random.uniform(0.75, 1.1)
        orig_s = self.img_size
        new_s  = int(orig_s * factor)

        image = image.resize((new_s, new_s), Image.Resampling.BILINEAR)

        if factor > 1.0:
            # Crop về img_size (random crop)
            max_off = new_s - orig_s
            off_x = random.randint(0, max_off)
            off_y = random.randint(0, max_off)
            image = image.crop((off_x, off_y, off_x + orig_s, off_y + orig_s))
            new_annos = []
            for a in annos:
                xmin, ymin, xmax, ymax = a['bbox']
                xmin = xmin * factor - off_x
                xmax = xmax * factor - off_x
                ymin = ymin * factor - off_y
                ymax = ymax * factor - off_y
                # Clip và bỏ box quá nhỏ
                xmin = max(0.0, xmin); ymin = max(0.0, ymin)
                xmax = min(float(orig_s), xmax); ymax = min(float(orig_s), ymax)
                if xmax - xmin >= 4 and ymax - ymin >= 4:
                    new_annos.append({**a, 'bbox': [xmin, ymin, xmax, ymax]})
        else:
            # Pad về img_size (center)
            canvas = Image.new("RGB", (orig_s, orig_s), (128, 128, 128))
            off_x  = (orig_s - new_s) // 2
            off_y  = (orig_s - new_s) // 2
            canvas.paste(image, (off_x, off_y))
            image  = canvas
            new_annos = []
            for a in annos:
                xmin, ymin, xmax, ymax = a['bbox']
                new_annos.append({**a, 'bbox': [
                    xmin * factor + off_x, ymin * factor + off_y,
                    xmax * factor + off_x, ymax * factor + off_y
                ]})

        return image, new_annos

    # ─────────────────────────────── encode target ────────────────────────

    def _encode_targets(self, annos):
        """
        Encode annotations → multi-scale target tensors.
        Nhất quán với loss.py (log-space w/h) và predict.py (anchor * exp(pred)).
        """
        targets = [torch.zeros((self.num_anchors, S, S, 5 + self.num_classes))
                   for S in self.grids]

        for a in annos:
            xmin, ymin, xmax, ymax = a['bbox']
            cls_idx = self.class_to_idx[a['class']]

            # Chuẩn hóa về [0,1]
            S = self.img_size
            x_center = ((xmin + xmax) / 2.0) / S
            y_center  = ((ymin + ymax) / 2.0) / S
            w_norm    = (xmax - xmin) / S
            h_norm    = (ymax - ymin) / S

            if w_norm <= 0 or h_norm <= 0:
                continue

            # Tìm anchor phù hợp nhất
            iou             = self._iou_wh(torch.tensor(w_norm), torch.tensor(h_norm))
            best_anchor_idx = iou.argmax().item()

            scale_idx          = best_anchor_idx // 3
            anchor_idx_in_scale = best_anchor_idx % 3
            grid_s             = self.grids[scale_idx]

            i = min(int(grid_s * y_center), grid_s - 1)
            j = min(int(grid_s * x_center), grid_s - 1)

            if targets[scale_idx][anchor_idx_in_scale, i, j, 4] != 0:
                continue  # ô đã bị chiếm

            # Offset trong cell [0,1]
            x_cell = grid_s * x_center - j
            y_cell  = grid_s * y_center - i

            # ── encode w/h thành log-space để khớp với loss.py ──
            # loss.py dùng MSE(pred_wh, log(target_wh))
            # predict.py decode: w = anchor_w * exp(pred_w)
            # Để nhất quán:
            #   target lưu log(w_norm / anchor_w)  → exp() → w_norm khi decode
            anchor_w = self.anchors[best_anchor_idx, 0].item()
            anchor_h = self.anchors[best_anchor_idx, 1].item()
            tw = math.log(w_norm / anchor_w + 1e-6)
            th = math.log(h_norm / anchor_h + 1e-6)

            targets[scale_idx][anchor_idx_in_scale, i, j, 0] = x_cell
            targets[scale_idx][anchor_idx_in_scale, i, j, 1] = y_cell
            targets[scale_idx][anchor_idx_in_scale, i, j, 2] = tw
            targets[scale_idx][anchor_idx_in_scale, i, j, 3] = th
            targets[scale_idx][anchor_idx_in_scale, i, j, 4] = 1.0
            targets[scale_idx][anchor_idx_in_scale, i, j, 5 + cls_idx] = 1.0

        return targets

    # ─────────────────────────────── mosaic ──────────────────────────────

    def _load_sample(self, idx):
        """Load 1 sample đã qua letterbox (dùng cho mosaic)."""
        img_id   = self.image_ids[idx]
        img_info = self.img_data[img_id]['info']
        annos    = self.img_data[img_id]['annos']
        filename = os.path.basename(img_info['file_name'])
        image    = Image.open(os.path.join(self.img_dir, filename)).convert("RGB")
        orig_w, orig_h = img_info['width'], img_info['height']
        image, annos, *_ = self._letterbox(image, annos, orig_w, orig_h)
        return image, annos

    def _aug_mosaic(self, idx):
        """
        Mosaic 2x2: ghép 4 ảnh thành 1 ảnh img_size x img_size.
        Giúp model học nhiều object trong 1 ảnh, tăng context diversity.
        """
        S   = self.img_size
        cx  = random.randint(S // 4, 3 * S // 4)
        cy  = random.randint(S // 4, 3 * S // 4)

        indices = [idx] + random.choices(range(len(self.image_ids)), k=3)
        canvas  = Image.new("RGB", (S, S), (128, 128, 128))
        all_annos = []

        placements = [
            (0,  0,  cx, cy),
            (cx, 0,  S,  cy),
            (0,  cy, cx, S),
            (cx, cy, S,  S),
        ]

        for i, (x1, y1, x2, y2) in enumerate(placements):
            img_i, annos_i = self._load_sample(indices[i])
            cell_w = x2 - x1
            cell_h = y2 - y1
            img_i  = img_i.resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            canvas.paste(img_i, (x1, y1))

            scale_x = cell_w / S
            scale_y = cell_h / S
            for a in annos_i:
                bx1, by1, bx2, by2 = a['bbox']
                bx1 = bx1 * scale_x + x1
                bx2 = bx2 * scale_x + x1
                by1 = by1 * scale_y + y1
                by2 = by2 * scale_y + y1
                bx1 = max(x1, bx1); by1 = max(y1, by1)
                bx2 = min(x2, bx2); by2 = min(y2, by2)
                if bx2 - bx1 >= 4 and by2 - by1 >= 4:
                    all_annos.append({**a, 'bbox': [bx1, by1, bx2, by2]})

        return canvas, all_annos

    # ─────────────────────────────── __getitem__ ──────────────────────────

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.img_data[img_id]['info']
        annos    = self.img_data[img_id]['annos']

        filename = os.path.basename(img_info['file_name'])
        image    = Image.open(os.path.join(self.img_dir, filename)).convert("RGB")
        orig_w, orig_h = img_info['width'], img_info['height']

        if self.is_train:
            # Mosaic (50% xac suat) - thay the letterbox thong thuong
            if random.random() > 0.5:
                image, annos = self._aug_mosaic(idx)
            else:
                image, annos, *_ = self._letterbox(image, annos, orig_w, orig_h)

            # Color jitter
            image = self._aug_color_jitter(image)

            # Scale jitter (30% xac suat)
            if random.random() > 0.7:
                image, annos = self._aug_scale_jitter(image, annos)

            # Flip ngang (50%)
            if random.random() > 0.5:
                image, annos = self._aug_flip(image, annos)
        else:
            image, annos, *_ = self._letterbox(image, annos, orig_w, orig_h)

        # Tensor + normalize ImageNet
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        # Encode targets
        targets = self._encode_targets(annos)

        return image, targets, str(img_id), torch.tensor([orig_w, orig_h])