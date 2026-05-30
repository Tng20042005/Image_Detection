import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from utils.dataset import CustomDetectionDataset
from models.detector import YOLO_Lite
from utils.loss import YoloLoss


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Bắt đầu huấn luyện trên: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── 1. DATALOADER ───────────────────────────────────────────────────
    print("📦 Tải dữ liệu...")
    train_dataset = CustomDetectionDataset(args.train_data, args.image_dir,    is_train=True)
    val_dataset   = CustomDetectionDataset(args.val_data,   args.val_image_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )

    # ── 2. MODEL + LOSS ─────────────────────────────────────────────────
    print("🧠 Khởi tạo YOLO_Lite (ResNet-34)...")
    model     = YOLO_Lite(num_classes=5).to(device)
    criterion = YoloLoss(num_classes=5)

    # ── 3. OPTIMIZER (differential LR) ──────────────────────────────────
    backbone_params   = []
    head_fpn_params   = []
    for name, param in model.named_parameters():
        if any(k in name for k in ("pred_", "lateral_", "smooth_")):
            head_fpn_params.append(param)
        else:
            backbone_params.append(param)

    BASE_LR = args.lr
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': BASE_LR * 0.1},
        {'params': head_fpn_params, 'lr': BASE_LR},
    ], weight_decay=1e-4)

    # ── 4. SCHEDULER: Warmup + Cosine ───────────────────────────────────
    WARMUP_EPOCHS = 5    

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - WARMUP_EPOCHS)

    # ── 5. AMP ──────────────────────────────────────────────────────────
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss   = float('inf')
    patience        = 10
    patience_counter = 0

    print("🔥 Bắt đầu training...")
    for epoch in range(args.epochs):
        # ── TRAIN ────────────────────────────────────────────────────────
        model.train()
        train_loss  = 0.0
        valid_steps = 0

        for batch_idx, (images, targets, img_ids, orig_sizes) in enumerate(train_loader):
            images  = images.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with autocast('cuda'):
                    preds = model(images)
                    loss  = criterion(preds, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  NaN/Inf loss at batch {batch_idx}, skip")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images)
                loss  = criterion(preds, targets)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠️  NaN/Inf loss at batch {batch_idx}, skip")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss  += loss.item()
            valid_steps += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  [Ep {epoch+1}/{args.epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_train = train_loss / max(valid_steps, 1)

        # ── VALIDATION ───────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        val_steps  = 0

        with torch.no_grad():
            for images, targets, img_ids, orig_sizes in val_loader:
                images  = images.to(device)
                targets = [t.to(device) for t in targets]

                if scaler is not None:
                    with autocast('cuda'):
                        preds = model(images)
                        loss  = criterion(preds, targets)
                else:
                    preds = model(images)
                    loss  = criterion(preds, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                val_loss  += loss.item()
                val_steps += 1

        avg_val = val_loss / max(val_steps, 1)
        scheduler.step()

        lr0 = optimizer.param_groups[0]['lr']
        lr1 = optimizer.param_groups[1]['lr']
        print(f"✅ Epoch [{epoch+1}/{args.epochs}] "
              f"LR={lr0:.2e}/{lr1:.2e} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # ── SAVE BEST ────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            path = os.path.join(args.checkpoint_dir, 'best.pth')
            torch.save(model.state_dict(), path)
            print(f"  🌟 Lưu best model (val={best_val_loss:.4f})")
        else:
            if epoch >= WARMUP_EPOCHS + 5:   # Không early-stop quá sớm
                patience_counter += 1
                print(f"  ⏸  Không cải thiện ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"🛑 Early stopping tại epoch {epoch+1}")
                    break

        # Lưu checkpoint định kỳ mỗi 10 epoch
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'ckpt_ep{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)


def collate_fn(batch):
    """Custom collate để xử lý targets là list of lists."""
    images     = torch.stack([b[0] for b in batch])
    img_ids    = [b[2] for b in batch]
    orig_sizes = torch.stack([b[3] for b in batch])

    # targets: mỗi sample là list[3 tensors], stack theo batch
    num_scales = len(batch[0][1])
    targets    = []
    for s in range(num_scales):
        scale_targets = torch.stack([b[1][s] for b in batch])
        targets.append(scale_targets)

    return images, targets, img_ids, orig_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',     type=str, default="public/annotations/train.json")
    parser.add_argument('--val_data',       type=str, default="public/annotations/val.json")
    parser.add_argument('--image_dir',      type=str, default="public/train/images/")
    parser.add_argument('--val_image_dir',  type=str, default="public/val/images/")
    parser.add_argument('--checkpoint_dir', type=str, default="models/")
    parser.add_argument('--epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',     type=int,   default=32)
    parser.add_argument('--lr',             type=float, default=5e-4)
    parser.add_argument('--num_workers',    type=int,   default=16)
    args = parser.parse_args()
    main(args)