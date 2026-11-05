"""흰 배경 캐릭터 PNG를 투명 PNG로 바꾼다.
테두리에서 이어진 흰 영역만 지우므로 윤곽선 안쪽의 흰색은 남는다.
실행: python cutout.py <입력폴더> <출력폴더> [--size 512]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

WHITE_MIN = 232  # 이보다 밝으면 배경 후보
BAND = 2  # 배경 경계에서 부드럽게 처리할 픽셀 수


def cut(path):
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    near_white = rgb.min(axis=2) >= WHITE_MIN
    labels, _ = ndimage.label(near_white)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    bg = np.isin(labels, edge[edge > 0])
    # 윤곽선과 경계 사이에 갇힌 작은 얼룩은 그대로 둔다
    alpha = np.where(bg, 0.0, 1.0)
    band = ndimage.binary_dilation(bg, iterations=BAND) & ~bg
    # 경계 픽셀: 흰색과 섞인 정도만큼 투명하게 하고 색에서 흰색을 뺀다
    a = np.clip((255 - rgb.min(axis=2)) / 150.0, 0, 1)
    alpha[band] = a[band]
    out = rgb.copy()
    safe = np.maximum(alpha, 0.05)[..., None]
    fixed = np.clip((rgb - 255 * (1 - safe)) / safe, 0, 255)
    out[band] = fixed[band]
    rgba = np.dstack([out, alpha * 255]).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for p in sorted(args.src.glob("*.png")):
        im = cut(p)
        im.thumbnail((args.size, args.size), Image.LANCZOS)
        im.save(args.out / p.name, optimize=True)
        print(p.name, im.size)


if __name__ == "__main__":
    main()
