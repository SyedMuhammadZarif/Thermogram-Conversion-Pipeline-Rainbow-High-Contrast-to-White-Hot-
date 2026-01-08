import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# -----------------------------
# Local paths (change these)
# -----------------------------
input_folder = r"STEP 1 Data Processing\IEEE\IEE_RESIZED"
output_folder = r"STEP 1 Data Processing\IEEE\IEEE_Grayscaled"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Functions
# -----------------------------
def enhanced_hue_to_gray(hue):
    
    shifted_hue = (hue + 30) % 360
    nh = shifted_hue / 360.0  # normalize [0,1]

    center = 0.1
    width = 0.1
    amplitude = 0.5

    gaussian = amplitude * np.exp(-0.5 * ((nh - center) / width) ** 2)
    base = 1.0 - nh
    boosted = np.clip(base + gaussian, 0, 1)
    gray = (boosted * 255).astype(np.uint8)
    return gray

def is_black(pixel, threshold=20):
    return np.all(pixel <= threshold)

# -----------------------------
# Processing loop
# -----------------------------
for root, dirs, files in os.walk(input_folder):
    rel_path = os.path.relpath(root, input_folder)
    out_dir = os.path.join(output_folder, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for file in tqdm(files, desc=f"Processing {rel_path}"):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        in_path = os.path.join(root, file)
        out_path = os.path.join(out_dir, file)

        img_bgr = cv2.imread(in_path)
        if img_bgr is None:
            print(f"⚠️ Could not read {in_path}")
            continue

        mask_black = np.apply_along_axis(is_black, 2, img_bgr)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_360 = hsv[:, :, 0] * 2.0

        gray = enhanced_hue_to_gray(hue_360)
        gray[mask_black] = 0

        # Optional: mild blur
        gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

        Image.fromarray(gray_blur).save(out_path)

print("✅ All images processed and saved!")
