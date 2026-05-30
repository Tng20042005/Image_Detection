import os
from PIL import Image

img_dir = "public/train/images"  # đổi nếu khác

files = [
    f for f in os.listdir(img_dir)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
][:]

widths = []
heights = []

for f in files:
    w, h = Image.open(os.path.join(img_dir, f)).size
    widths.append(w)
    heights.append(h)

widths.sort()
heights.sort()

print(f"Số ảnh mẫu: {len(files)}")
print(f"Width  — min:{min(widths)} median:{widths[len(widths)//2]} max:{max(widths)}")
print(f"Height — min:{min(heights)} median:{heights[len(heights)//2]} max:{max(heights)}")

print(
    f"Width  phân vị 25/75/90: "
    f"{widths[len(widths)//4]} / "
    f"{widths[3*len(widths)//4]} / "
    f"{widths[int(len(widths)*0.9)]}"
)

print(
    f"Height phân vị 25/75/90: "
    f"{heights[len(heights)//4]} / "
    f"{heights[3*len(heights)//4]} / "
    f"{heights[int(len(heights)*0.9)]}"
)