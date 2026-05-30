import os
import subprocess

def main():
    # Cấu hình thư mục
    preds_dir = "prediction_output"
    output_dir = "val_score_output"
    gt_file = "public/annotations/val.json"
    eval_script = "public/tools/evaluate_predictions.py"

    # Tạo thư mục chứa điểm
    os.makedirs(output_dir, exist_ok=True)

    # Kiểm tra đầu vào
    if not os.path.exists(preds_dir):
        print(f"❌ Không tìm thấy thư mục {preds_dir}. Hãy chạy inference_grid.py trước!")
        return

    # Lọc lấy tất cả các file json sinh ra
    pred_files = [f for f in os.listdir(preds_dir) if f.endswith(".json")]
    pred_files.sort() # Sắp xếp tăng dần theo ngưỡng

    print(f"🚀 Bắt đầu chấm điểm {len(pred_files)} file cấu hình...\n")

    for pred_file in pred_files:
        pred_path = os.path.join(preds_dir, pred_file)
        
        # Tạo tên file output tương ứng (vd: score_preds_0.45.json)
        score_filename = f"score_{pred_file}"
        score_path = os.path.join(output_dir, score_filename)

        # Build câu lệnh
        cmd = [
            "python", eval_script,
            "--ground_truth", gt_file,
            "--predictions", pred_path,
            "--output", score_path
        ]

        print(f"Đang chấm: {pred_file} -> {score_filename} ...", end=" ")
        
        # Chạy câu lệnh
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Xong")
        else:
            print("❌ Lỗi!")
            print(result.stderr)

    print(f"\n🎉 Đã chấm xong toàn bộ! Hãy vào thư mục '{output_dir}' để xem kết quả mAP tương ứng.")

if __name__ == "__main__":
    main()