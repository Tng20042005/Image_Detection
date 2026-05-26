from arch.efficientnet_b0 import YOLO_EfficientNetB0
from arch.resnet34 import YOLO_ResNet34

def YOLO_Lite(arch="resnet34", num_classes=5):
    """
    Hàm Factory giúp khởi tạo mạng động.
    Các script train.py và predict.py vẫn gọi YOLO_Lite() như cũ mà không bị lỗi.
    """
    print(f"⚙️  Đang khởi tạo kiến trúc mạng: {arch.upper()}")
    
    if arch == "efficientnet_b0":
        return YOLO_EfficientNetB0(num_classes=num_classes)
    elif arch == "resnet34":
        return YOLO_ResNet34(num_classes=num_classes)
    else:
        raise ValueError(f"❌ Không hỗ trợ kiến trúc: {arch}. Chỉ nhận 'efficientnet_b0' hoặc 'resnet34'")