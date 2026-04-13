"""
frame_0029에 머리카락 합성 → data/test/bad/synthetic_hair.png

test1.png 기준 두 위치:
  - 앞부분: 드론 상단 앞쪽 몸체
  - 중앙:   드론 몸체 가운데

실행:
    python make_synthetic.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
src_path  = os.path.join(BASE_DIR, "data", "train", "good", "frame_0029.png")
save_path = os.path.join(BASE_DIR, "data", "test", "bad", "synthetic_hair.png")

# 2배 해상도로 작업 후 축소 → 0.5px 수준의 얇은 선
SCALE = 2
img_orig = Image.open(src_path).convert('RGB')
W, H = img_orig.size   # 1920 x 1080
img = img_orig.resize((W * SCALE, H * SCALE), Image.LANCZOS)
draw = ImageDraw.Draw(img)


def bezier_hair(draw, x0, y0, x1, y1, cx, cy, color=(20, 20, 20), width=1):
    """2차 베지어 곡선으로 머리카락 그리기"""
    steps = 80
    points = []
    for t in np.linspace(0, 1, steps):
        bx = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
        by = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
        bx += np.random.uniform(-0.8, 0.8)
        by += np.random.uniform(-0.8, 0.8)
        points.append((bx, by))
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=color, width=width)


# image.png(1656x928) → frame_0029(1920x1080) 좌표 스케일
sx = 1920 / 1656
sy = 1080 / 928

def scale(x, y):
    return int(x * sx * SCALE), int(y * sy * SCALE)

# ── 빨간 선 기준: 왼쪽 몸체에 위에서 아래로 굽은 곡선 ──
# 시작(상단): image.png 약 (310, 195) → 끝(하단): 약 (370, 390)
# 제어점: 왼쪽으로 휘게 (240, 300)
x0, y0 = scale(490, 265)
x1, y1 = scale(545, 420)
cx, cy = scale(430, 345)

bezier_hair(draw, x0, y0, x1, y1, cx, cy, color=(18, 18, 18), width=1)

# 2배 해상도 → 원본 크기로 다운샘플 (얇은 선 효과)
img = img.resize((W, H), Image.LANCZOS)

os.makedirs(os.path.dirname(save_path), exist_ok=True)
img.save(save_path)
print(f"저장: {save_path}")

# 미리보기용 (원본과 나란히)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

orig = np.array(Image.open(src_path).convert('RGB'))
synth = np.array(img)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(orig);   axes[0].set_title("원본 (frame_0029)"); axes[0].axis('off')
axes[1].imshow(synth);  axes[1].set_title("합성 결함 (머리카락)"); axes[1].axis('off')
plt.tight_layout()

preview_path = os.path.join(BASE_DIR, "static", "synthetic_preview.png")
os.makedirs(os.path.dirname(preview_path), exist_ok=True)
plt.savefig(preview_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"미리보기: {preview_path}")
