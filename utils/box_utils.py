import torch

def calculate_iou(box1, box2):
    """
    Tính Intersection over Union (IoU) giữa 2 hộp bao.
    Format của box: [xmin, ymin, xmax, ymax]
    """
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    # Diện tích phần giao
    inter_width = torch.clamp(x2 - x1, min=0)
    inter_height = torch.clamp(y2 - y1, min=0)
    inter_area = inter_width * inter_height

    # Diện tích phần hợp
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area + 1e-6

    return inter_area / union_area

def nms(predictions, iou_threshold=0.5, conf_threshold=0.5):
    """
    Non-Maximum Suppression theo từng lớp.
    predictions: list chứa dict hoặc list [xmin, ymin, xmax, ymax, conf, class_id]
    """
    # Lọc bỏ các box có độ tin cậy thấp
    valid_boxes = [box for box in predictions if box[4] >= conf_threshold]
    
    # Sắp xếp theo độ tin cậy giảm dần
    valid_boxes = sorted(valid_boxes, key=lambda x: x[4], reverse=True)
    
    keep_boxes = []
    while valid_boxes:
        best_box = valid_boxes.pop(0)
        keep_boxes.append(best_box)
        
        # Giữ lại các box khác class HOẶC cùng class nhưng IoU < ngưỡng
        valid_boxes = [
            box for box in valid_boxes
            if box[5] != best_box[5] or calculate_iou(torch.tensor(best_box[:4]), torch.tensor(box[:4])) < iou_threshold
        ]
        
    return keep_boxes