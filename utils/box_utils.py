import torch

def nms(predictions, iou_threshold=0.5, conf_threshold=0.5):
    """
    Phiên bản NMS đã được tối ưu hóa bằng Vectorization của PyTorch.
    Không cần dùng vòng lặp Python duyệt từng box nữa.
    """
    if not predictions:
        return []

    # 1. Chuyển toàn bộ list thành Tensor ĐÚNG 1 LẦN (nếu đầu vào là list)
    if isinstance(predictions, list):
        preds = torch.tensor(predictions, dtype=torch.float32)
    else:
        preds = predictions.clone() # Tránh lỗi tham chiếu nếu truyền tensor gốc
        
    if preds.numel() == 0:
        return []

    # 2. Lọc bỏ các box có độ tin cậy thấp (sử dụng Boolean Mask)
    mask = preds[:, 4] >= conf_threshold
    preds = preds[mask]

    if preds.size(0) == 0:
        return []

    # 3. Sắp xếp tất cả các box theo độ tin cậy giảm dần
    scores = preds[:, 4]
    order = scores.argsort(descending=True)
    preds = preds[order]

    keep_boxes = []

    # 4. Vòng lặp NMS nhưng xử lý mảng (Vectorized IoU)
    while preds.size(0) > 0:
        # Rút box có điểm cao nhất ra
        best_box = preds[0]
        keep_boxes.append(best_box.tolist())
        
        if preds.size(0) == 1:
            break

        # Tách tọa độ của best_box và [TẤT CẢ các box còn lại]
        best_bbox = best_box[:4].unsqueeze(0)  # Shape: [1, 4]
        rest_bboxes = preds[1:, :4]            # Shape: [N-1, 4]
        
        best_class = best_box[5]
        rest_classes = preds[1:, 5]            # Shape: [N-1]

        # --- TÍNH IOU SONG SONG CHO TẤT CẢ CÁC BOX CÙNG LÚC ---
        x1 = torch.max(best_bbox[:, 0], rest_bboxes[:, 0])
        y1 = torch.max(best_bbox[:, 1], rest_bboxes[:, 1])
        x2 = torch.min(best_bbox[:, 2], rest_bboxes[:, 2])
        y2 = torch.min(best_bbox[:, 3], rest_bboxes[:, 3])

        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter_area = inter_w * inter_h

        area1 = (best_bbox[:, 2] - best_bbox[:, 0]) * (best_bbox[:, 3] - best_bbox[:, 1])
        area2 = (rest_bboxes[:, 2] - rest_bboxes[:, 0]) * (rest_bboxes[:, 3] - rest_bboxes[:, 1])
        union_area = area1 + area2 - inter_area + 1e-6

        iou = inter_area / union_area

        # --- ĐIỀU KIỆN GIỮ LẠI BẰNG BOOLEAN LOGIC ---
        # Giữ lại nếu: Khác class HOẶC (Cùng class VÀ IoU < ngưỡng)
        different_class = rest_classes != best_class
        low_iou = iou < iou_threshold
        
        # Phép OR trên ma trận boolean
        keep_mask = different_class | low_iou

        # Cắt gọt lại danh sách preds bằng mask cho vòng lặp tiếp theo
        preds = preds[1:][keep_mask]

    return keep_boxes